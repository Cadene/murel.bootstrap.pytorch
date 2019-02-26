from bootstrap.lib.options import Options
from block.models.criterions.vqa_cross_entropy import VQACrossEntropyLoss

def factory(engine, mode):
    name = Options()['model.criterion.name']
    split = engine.dataset[mode].split
    eval_only = 'train' not in engine.dataset

    if name == 'vqa_cross_entropy':
        if split == 'test':
            return None
        criterion = VQACrossEntropyLoss()

    else:
        raise ValueError(name)

    return criterion
