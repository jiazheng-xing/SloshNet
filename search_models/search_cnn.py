import torch
import torch.nn as nn
import torch.nn.functional as F
from search_models.search_cell import SearchCell
import search_models.genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C, PRIMITIVES, n_nodes=4,):

        super().__init__()
        self.C = C
        self.PRIMITIVES = PRIMITIVES

        self.cells = nn.ModuleList()

        cell = SearchCell(n_nodes,C, self.PRIMITIVES)
        self.cells.append(cell)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, weights_normal):
        # for i in range(len(x)):
        #     N,T,C,W,H = x[i].size()   
        #     x[i]=x[i].reshape(-1,C,W,H)
        for cell in self.cells:
            weights = weights_normal
            out_feature = cell(x, weights)

        # out = self.avgpool(out_feature)
        # out = out.view(out.size(0), -1) # flatten
        return out_feature


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C, criterion, n_nodes=4, device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.PRIMITIVES = gt.PRIMITIVES_FEWSHOT
        
        # initialize architect parameters: alphas
        n_ops = len(self.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()

        for i in range(n_nodes-1):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+1, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C,self.PRIMITIVES, n_nodes)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal)
        # print(len((self.device_ids)))
        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def euclidean_dist( self, x, y, normalize=False):
    # x: N x D
    # y: M x D
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        n = x.size(0) #5
        m = y.size(0) #5
        d = x.size(1) #2048
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def loss(self, X, y,nway,n_support,n_query):
        logits = self.forward(X)
        sq = n_support + n_query
        t = 8 #todo
        logits = logits.reshape(nway,sq,t,-1) # 5 x 2 x 8 x 2048
        logits = logits.mean(2) # average
        z_support   = logits[:, :n_support]
        z_query     = logits[:, n_support:]
        z_proto     = z_support.reshape(nway, n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.reshape(nway* n_query, -1 )
        dists = self.euclidean_dist(z_query, z_proto)
        scores = -dists
        return self.criterion(scores, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, self.PRIMITIVES,k=1)
        concat = range(1, 1+self.n_nodes) # concat all intermediate nodes
        return gt.Genotype(normal=gene_normal, normal_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
