from sys import implementation
from matplotlib.style import context
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
from itertools import combinations 
import torch.nn.functional as F

from torch.autograd import Variable
from Modules.resnet import resnet50
import torchvision.models as models

from search_models.search_cnn import SearchCNNController
from Modules.cmmformer import CMM
from Modules.cstmformer import TemporalTransformer
from Modules.align import SpatialModulation

NUM_SAMPLES=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)

        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) 
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way).cuda()

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            support_labels = support_labels.to(device = mh_support_set_ks.device)
            c = c.to(device = mh_support_set_ks.device)
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3).contiguous()
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
        
class SloshNet(nn.Module):
    """
    Standard Resnet connected to a Temporal Cross Transformer.
    
    """
    def __init__(self, args):
        super(SloshNet, self).__init__()


        beta_part = 0.6 
        beta  = 1e-3*torch.randn(1, 5) 
        beta_exp_sum = torch.sum(torch.exp(beta),dim=-1)
        beta[:,-1] = beta_exp_sum[:][0] * beta_part
        beta[:,:-1] = beta_exp_sum[:][0] * (1-beta_part) / int(beta.shape[1] -1)
        beta = torch.log(beta)
        self.beta1 = nn.Parameter(beta)

        gamma_part = 0.5
        gamma1  = 1e-3*torch.randn(1, 2)
        gamma_exp_sum = torch.sum(torch.exp(gamma1),dim=-1)
        gamma1[:,-1] = gamma_exp_sum[:][0] * gamma_part
        gamma1[:,:-1] = gamma_exp_sum[:][0] * (1-gamma_part) / int(gamma1.shape[1] -1)
        gamma1 = torch.log(gamma1)
        self.gamma1 = nn.Parameter(gamma1)

        self.train()
        self.args = args

        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif self.args.method == "resnet50_darts":
            new_model = resnet50(pretrained=True)
            resnet = new_model
        
        device = torch.device("cuda")
        C = [256, 512, 1024, 2048]
        net_crit = nn.CrossEntropyLoss().to(device)
        darts_model = SearchCNNController(C, net_crit, n_nodes=4, device_ids=None)
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.resnet = resnet
        self.darts_model = darts_model
        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 
        self.cmm = CMM(C[-1], self.args.seq_len)
        self.cstm = TemporalTransformer(width=C[-1], layers=1, heads=8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_patches = 49
        self.spatial_modulation = SpatialModulation(inplanes=[256,512, 1024, 2048],planes=2048)

    def forward(self, context_images, context_labels, target_images):

        beta1 = [F.softmax(alpha, dim=-1) for alpha in self.beta1]
        gamma1 = [F.softmax(alpha, dim=-1) for alpha in self.gamma1]

        context_features_hid = self.resnet(context_images)# ([40, 256, 56, 56])
        target_features_hid = self.resnet(target_images)

        context_features_hid = self.spatial_modulation(context_features_hid)
        target_features_hid = self.spatial_modulation(target_features_hid)
        context_stats = self.darts_model(context_features_hid)
        target_stats = self.darts_model(target_features_hid) #([40, 2048, 7, 7])

        context_features = context_stats[0]*beta1[0][0] +context_stats[1]*beta1[0][1]+context_stats[2]*beta1[0][2]+context_stats[3]*beta1[0][3] +context_features_hid[-1]*beta1[0][4]
        target_features =  target_stats[0]*beta1[0][0]  +target_stats[1]*beta1[0][1]+target_stats[2]*beta1[0][2]+target_stats[3]*beta1[0][3]  +target_features_hid[-1]*beta1[0][4]
        
        NT,D,W,H = context_features.size()

        context_labels = context_labels.to(device = context_features[0].device)

        context_features = context_features.reshape(-1, self.args.seq_len, D,W,H) 
        target_features = target_features.reshape(-1, self.args.seq_len, D,W,H)

        context_features = self.cmm(context_features)*gamma1[0][1]*2 +self.cstm(context_features)*gamma1[0][0]*2
        target_features = self.cmm(target_features)*gamma1[0][1]*2 +self.cstm(target_features)*gamma1[0][0]*2

        context_features = self.avgpool(context_features)
        context_features = context_features.view(context_features.size(0), -1).view(-1,self.args.seq_len,D)
        target_features = self.avgpool(target_features)   
        target_features = target_features.view(target_features.size(0), -1).view(-1,self.args.seq_len,D)
        all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])
        
        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]])}
        return return_dict

    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list  
        
    def loss_fn(self,test_logits_sample, test_labels, device):

        size = test_logits_sample.size()
        sample_count = size[0]  # scalar for the loop counter
        num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

        log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
        for sample in range(sample_count):
            log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
        score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
        return -torch.sum(score, dim=0)

    def loss(self,task_dict):
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

        model_dict = self.forward(context_images, context_labels, target_images)
        target_logits = model_dict['logits'].to(device)
        target_logits_post_pat = model_dict['logits_post_pat'].to(self.device)
        task_loss_post_pat = self.loss_fn(target_logits_post_pat, target_labels, self.device) / self.args.tasks_per_batch
        task_loss = self.loss_fn(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_loss = task_loss + 0.1 * task_loss_post_pat

        return task_loss

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.cmm.cuda(0)
            self.cmm = torch.nn.DataParallel(self.cmm, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.cstm.cuda(0)
            self.cstm  = torch.nn.DataParallel(self.cstm, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.spatial_modulation.cuda(0)
            self.spatial_modulation = torch.nn.DataParallel(self.spatial_modulation, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.darts_model.cuda(0)
            self.avgpool.cuda(0)
            self.transformers.cuda(0)