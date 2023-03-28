import torch
import torch.nn as nn
from search_models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes,  C , PRIMITIVES):
        
        super().__init__()
 
        self.n_nodes = n_nodes 
        # generate dag
        self.dag = nn.ModuleList()
        self.PRIMITIVES =  PRIMITIVES
        for i in range(self.n_nodes - 1):
            self.dag.append(nn.ModuleList())
            for j in range(1 + i):
                stride = 2**(i+1-j)
                op = ops.MixedOp(C[j],C[i+1],stride, self.PRIMITIVES)
                self.dag[i].append(op)
        print("yes!!")

    def forward(self, input_features, w_dag):
        layer1_feature = input_features[0].cuda()
        alpha_prune_threshold = 0.01
        states = [layer1_feature]
        k = 0
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = 0
            k += 1
            for i, (s, w) in enumerate(zip(states, w_list)):
                cur = edges[i](s,input_features[k].cuda(), w, alpha_prune_threshold=alpha_prune_threshold)
                s_cur += cur
            
            states.append(s_cur)

        s_out = states
        return s_out
