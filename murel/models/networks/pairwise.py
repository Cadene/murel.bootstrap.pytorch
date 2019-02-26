import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import block

def make_pairs_ids(n_regions, bsize):
    pairs_ids = []
    for batch_id in range(bsize):
        pairs_id = torch.tensor([
            (batch_id,i,j) for i,j in \
            itertools.product(range(n_regions),repeat=2)],
            requires_grad=False,
            dtype=torch.long)
        pairs_ids.append(pairs_id)
    out = torch.cat(pairs_ids).contiguous()
    return out

class Pairwise(nn.Module):

    def __init__(self,
            residual=True,
            fusion_coord={},
            fusion_feat={},
            agg={}):
        super(Pairwise, self).__init__()
        self.residual = residual
        self.fusion_coord = fusion_coord
        self.fusion_feat = fusion_feat
        self.agg = agg
        #
        if self.fusion_coord:
            self.f_coord_module = block.factory_fusion(self.fusion_coord)
        if self.fusion_feat:
            self.f_feat_module = block.factory_fusion(self.fusion_feat)
        #
        self.buffer = None
        self.pairs_ids = None
        self.pairs_ids_n_regions = None
        self.pairs_ids_bsize = None

    def set_pairs_ids(self, n_regions, bsize, device):
        self.pairs_ids = make_pairs_ids(n_regions, bsize)
        self.pairs_ids = self.pairs_ids.to(device=device, non_blocking=True)
        self.pairs_ids_n_regions = n_regions
        self.pairs_ids_bsize = bsize

    def set_buffer(self):
        self.buffer = {}

    def forward(self, mm, coords=None):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        if self.pairs_ids is None \
           or self.pairs_ids_n_regions != n_regions \
           or self.pairs_ids_bsize != bsize:
            self.set_pairs_ids(n_regions, bsize, device=mm.device)

        Rij = 0
        if self.fusion_coord:
            assert coords is not None
            pair_coords = coords[self.pairs_ids[:,0][:,None], self.pairs_ids[:,1:]]
            pair_coords.detach_() # REALLY IMPORTANT
            Rij += self.f_coord_module([pair_coords[:,0,:], pair_coords[:,1,:]])
        if self.fusion_feat:
            pair_mm = mm[self.pairs_ids[:,0][:,None], self.pairs_ids[:,1:]]
            pair_mm.detach_() # REALLY IMPORTANT
            Rij += self.f_feat_module([pair_mm[:,0,:], pair_mm[:,1,:]])

        Rij = Rij.view(bsize, n_regions, n_regions, -1)

        if self.agg['type'] == 'max':
            mm_new, argmax = Rij.max(2)
        else:
            mm_new = getattr(Rij, self.agg['type'])(2)

        if self.buffer is not None:
            self.buffer['mm'] = mm.data.cpu() # bx36x2048
            self.buffer['mm_new'] = mm.data.cpu() # bx36x2048
            self.buffer['argmax'] = argmax.data.cpu() # bx36x2048
            L1_regions = torch.norm(mm_new.data, 1, 2) # bx36
            L2_regions = torch.norm(mm_new.data, 2, 2) # bx36
            self.buffer['L1_max'] = L1_regions.max(1)[0].cpu() # b
            self.buffer['L2_max'] = L2_regions.max(1)[0].cpu() # b

        if self.residual:
            mm_new += mm

        return mm_new
