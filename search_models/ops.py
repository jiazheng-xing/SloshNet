import torch.nn.functional as F

import torch
import torch.nn as nn
# from darts.genotypes import PRIMITIVES


OPS = {
    "GP1":lambda  C_in, C_out ,stride, affine:GP1(),
    "GP2":lambda  C_in, C_out ,stride, affine:GP2(),
    "Sum":lambda  C_in, C_out ,stride, affine:Sum(),
}

class GP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1

class GP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x2, x1):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1


class Sum(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return x1+x2

class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C_in, C_out, stride, PRIMITIVES):
        super().__init__()
        self._ops = nn.ModuleList()
        self.PRIMITIVES = PRIMITIVES
        for primitive in self.PRIMITIVES:
            op = OPS[primitive](C_in, C_out,stride, affine=False)

            self._ops.append(op)

    def forward(self, x1, x2,weights, alpha_prune_threshold=0.0):
        """
        Args:
            x: input
            weights: weight for each operation
            alpha_prune_threshold: prune ops during forward pass if alpha below threshold
        """

        return sum(
            w * op(x1,x2) for w, op in zip(weights, self._ops) if w > alpha_prune_threshold
        )
