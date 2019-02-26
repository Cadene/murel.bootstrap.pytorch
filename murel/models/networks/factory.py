import sys
import copy
import torch
import torch.nn as nn
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from block.models.networks.vqa_net import VQANet as AttentionNet
from .murel_net import MuRelNet

def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()['model.network']

    if opt['name'] == 'attention_net':
        net = AttentionNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            attention=opt['attention'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid)

    elif opt['name'] == 'murel_net':
        net = MuRelNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            n_step=opt['n_step'],
            shared=opt['shared'],
            cell=opt['cell'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid)

    else:
        raise ValueError(opt['name'])

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net


if __name__ == '__main__':
    factory()
