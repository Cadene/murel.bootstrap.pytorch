from bootstrap.lib.options import Options
from block.models.metrics.vqa_accuracies import VQAAccuracies

def factory(engine, mode):
    name = Options()['model.metric.name']
    metric = None

    if name == 'vqa_accuracies':
        if mode == 'train':
            split = engine.dataset['train'].split
            if split == 'train':
                metric = VQAAccuracies(engine,
                    mode='train',
                    open_ended=('tdiuc' not in Options()['dataset.name']),
                    tdiuc=True,
                    dir_exp=Options()['exp.dir'],
                    dir_vqa=Options()['dataset.dir'])
            elif split == 'trainval':
                metric = None
            else:
                raise ValueError(split)
        elif mode == 'eval':
            metric = VQAAccuracies(engine,
                mode='eval',
                open_ended=('tdiuc' not in Options()['dataset.name']),
                tdiuc=('tdiuc' in Options()['dataset.name'] or Options()['dataset.eval_split'] != 'test'),
                dir_exp=Options()['exp.dir'],
                dir_vqa=Options()['dataset.dir'])
        else:
            metric = None

    else:
        raise ValueError(name)
    return metric
