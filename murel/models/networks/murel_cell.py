from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import block
from .pairwise import Pairwise

class MuRelCell(nn.Module):

    def __init__(self,
            residual=False,
            fusion={},
            pairwise={}):
        super(MuRelCell, self).__init__()
        self.residual = residual
        self.fusion = fusion
        self.pairwise = pairwise
        #
        self.fusion_module = block.factory_fusion(self.fusion)
        if self.pairwise:
            self.pairwise_module = Pairwise(**pairwise)

    def forward(self, q_expand, mm, coords=None):
        mm_new = self.process_fusion(q_expand, mm)

        if self.pairwise:
            mm_new = self.pairwise_module(mm_new, coords)

        if self.residual:
            mm_new = mm_new + mm
        return mm_new

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]
        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm
