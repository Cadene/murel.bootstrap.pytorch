import torch
import torch.nn as nn
import torch.nn.functional as F
import block


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

    def set_buffer(self):
        self.buffer = {}

    def forward(self, mm, coords=None):
        bsize = mm.shape[0]
        nregion = mm.shape[1]

        Rij = 0
        if self.fusion_coord:
            assert coords is not None
            coords_l = coords[:,:,None,:]
            coords_l = coords_l.expand(bsize,nregion,nregion,coords.shape[-1])
            coords_l = coords_l.contiguous()
            coords_l = coords_l.view(bsize*nregion*nregion,coords.shape[-1])
            coords_r = coords[:,None,:,:]
            coords_r = coords_r.expand(bsize,nregion,nregion,coords.shape[-1])
            coords_r = coords_r.contiguous()
            coords_r = coords_r.view(bsize*nregion*nregion,coords.shape[-1])
            Rij += self.f_coord_module([coords_l, coords_r])
        if self.fusion_feat:
            mm_l = mm[:,:,None,:]
            mm_l = mm_l.expand(bsize,nregion,nregion,mm.shape[-1])
            mm_l = mm_l.contiguous()
            mm_l = mm_l.view(bsize*nregion*nregion,mm.shape[-1])
            mm_r = mm[:,None,:,:]
            mm_r = mm_r.expand(bsize,nregion,nregion,mm.shape[-1])
            mm_r = mm_r.contiguous()
            mm_r = mm_r.view(bsize*nregion*nregion,mm.shape[-1])
            Rij += self.f_feat_module([mm_l, mm_r])

        Rij = Rij.view(bsize,nregion,nregion,-1)

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
