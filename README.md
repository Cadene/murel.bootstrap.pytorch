# MUREL: Multimodal Relational Reasoning for Visual Question Answering

The **MuRel network** is a Machine Learning model learned end-to-end to answer questions about images. It relies on the object bounding boxes extracted from the image to build a complitely connected graph where each node corresponds to an object or region. The MuRel network contains a MuRel cell over which it iterates to fuse the question representation with local region features, progressively refining visual and question interactions. Finally, after a global aggregation of local representations, it answers the question using a bilinear model. Interestingly, the MuRel network doesn't include an explicit attention mechanism, usually at the core of state-of-the-art models. Its rich vectorial representation of the scene can even be leveraged to visualize the reasoning process at each step.

<p align="center">
    <img src="https://github.com/Cadene/murel.bootstrap.pytorch/blob/master/assets/murel_net.png?raw=true" width="900"/>
</p>

The **MuRel cell** is a novel reasoning module which models interactions between question and image regions. Its pairwise relational component enriches the multimodal representations of each node by taking their context into account in the modeling.

<p align="center">
    <img src="https://github.com/Cadene/murel.bootstrap.pytorch/blob/master/assets/murel_cell.png?raw=true" width="550"/>
</p>

In this repo, we make our datasets and models available via pip install. Also, we provide pretrained models and all the code needed to reproduce the experiments from our [CVPR 2019 paper](https://arxiv.org/abs/1902.09487).

#### Summary

* [Installation](#installation)
    * [Python 3 & Anaconda](#1-python-3--anaconda)
    * [As a standalone project](#2-as-standalone-project)
    * [Download datasets](#3-download-datasets)
    * [As a python library](#2-as-a-python-library)
* [Quick start](#quick-start)
    * [Train a model](#train-a-model)
    * [Evaluate a model](#evaluate-a-model)
* [Reproduce results](#reproduce-results)
    * [VQA2](#vqa2-dataset)
    * [VQACP2](#vqacp2-dataset)
    * [TDIUC](#tdiuc-dataset)
* [Pretrained models](#pretrained-models)
* [Useful commands](#useful-commands)
* [Citation](#citation)
* [Poster](#poster)
* [Authors](#authors)
* [Acknowledgment](#acknowledgment)


## Installation

### 1. Python 3 & Anaconda

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://www.continuum.io/downloads). Then, you can create an environment.

### 2. As standalone project

```
conda create --name murel python=3.7
source activate murel
git clone --recursive https://github.com/Cadene/murel.bootstrap.pytorch.git
cd murel.bootstrap.pytorch
pip install -r requirements.txt
```

### 3. Download datasets

Download annotations, images and features for VQA experiments:
```
bash murel/datasets/scripts/download_vqa2.sh
bash murel/datasets/scripts/download_vgenome.sh
bash murel/datasets/scripts/download_tdiuc.sh
bash murel/datasets/scripts/download_vqacp2.sh
```

**Note:** The features have been extracted from a pretrained Faster-RCNN with caffe. We don't provide the code for pretraining or extracting features for now.

### (2. As a python library)

By importing the `murel` python module, you can access datasets and models in a simple way:
```python
from murel.datasets.vqacp2 import VQACP2
from murel.models.networks.murel_net import MurelNet
from murel.models.networks.murel_cell import MurelCell
from murel.models.networks.pairwise import Pairwise
```

To be able to do so, you can use pip:
```
pip install murel.bootstrap.pytorch
```

Or install from source:
```
git clone https://github.com/Cadene/murel.bootstrap.pytorch.git
python setup.py install
```

**Note:** This repo is built on top of [block.bootstrap.pytorch](https://github.com/Cadene/block.bootstrap.pytorch). We import VQA2, TDIUC, VGenome from the latter.


## Quick start

### Train a model

The [boostrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) file load the options contained in a yaml file, create the corresponding experiment directory and start the training procedure. For instance, you can train our best model on VQA2 by running:
```
python -m bootstrap.run -o murel/options/vqa2/murel.yaml
```
Then, several files are going to be created in `logs/vqa2/murel`:
- [options.yaml](https://github.com/Cadene/block.bootstrap.pytorch/blob/master/assets/logs/vrd/block/options.yaml) (copy of options)
- [logs.txt](https://github.com/Cadene/block.bootstrap.pytorch/blob/master/assets/logs/vrd/block/logs.txt) (history of print)
- [logs.json](https://github.com/Cadene/block.bootstrap.pytorch/blob/master/assets/logs/vrd/block/logs.json) (batchs and epochs statistics)
- [view.html](http://htmlpreview.github.io/?https://raw.githubusercontent.com/Cadene/block.bootstrap.pytorch/master/assets/logs/vrd/block/view.html?token=AEdvLlDSYaSn3Hsr7gO5sDBxeyuKNQhEks5cTF6-wA%3D%3D) (learning curves)
- ckpt_last_engine.pth.tar (checkpoints of last epoch)
- ckpt_last_model.pth.tar
- ckpt_last_optimizer.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_engine.pth.tar (checkpoints of best epoch)
- ckpt_best_eval_epoch.accuracy_top1_model.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_optimizer.pth.tar

Many options are available in the [options directory](https://github.com/Cadene/murel.bootstrap.pytorch/blob/master/murel/options).

### Evaluate a model

At the end of the training procedure, you can evaluate your model on the testing set. In this example, [boostrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) load the options from your experiment directory, resume the best checkpoint on the validation set and start an evaluation on the testing set instead of the validation set while skipping the training set (train_split is empty). Thanks to `--misc.logs_name`, the logs will be written in the new `logs_test.txt` and `logs_test.json` files, instead of being appended to the `logs.txt` and `logs.json` files.
```
python -m bootstrap.run \
-o logs/vqa2/murel/options.yaml \
--exp.resume best_accuracy_top1 \
--dataset.train_split \
--dataset.eval_split test \
--misc.logs_name test
```

## Reproduce results

### VQA2 dataset

#### Training and evaluation (train/val)

We use this simple setup to tune our hyperparameters on the valset.

```
python -m bootstrap.run \
-o murel/options/vqa2/murel.yaml \
--exp.dir logs/vqa2/murel
```

#### Training and evaluation (train+val/val/test)

This heavier setup allows us to train a model on 95% of the concatenation of train and val sets, and to evaluate it on the 5% rest. Then we extract the predictions of our best checkpoint on the testset. Finally, we submit a json file on the EvalAI web site.

```
python -m bootstrap.run \
-o murel/options/vqa2/murel.yaml \
--exp.dir logs/vqa2/murel_trainval \
--dataset.proc_split trainval

python -m bootstrap.run \
-o logs/vqa2/murel_trainval/options.yaml \
--exp.resume best_eval_epoch.accuracy_top1 \
--dataset.train_split \
--dataset.eval_split test \
--misc.logs_name test
```

#### Training and evaluation (train+val+vg/val/test)

Same, but we add pairs from the VisualGenome dataset.

```
python -m bootstrap.run \
-o murel/options/vqa2/murel.yaml \
--exp.dir logs/vqa2/murel_trainval_vg \
--dataset.proc_split trainval \
--dataset.vg True

python -m bootstrap.run \
-o logs/vqa2/murel_trainval_vg/options.yaml \
--exp.resume best_eval_epoch.accuracy_top1 \
--dataset.train_split \
--dataset.eval_split test \
--misc.logs_name test
```

#### Compare experiments on valset

You can compare experiments by displaying their best metrics on the valset.

```
python -m murel.compare_vqa_val -d logs/vqa2/murel logs/vqa2/attention
```

#### Submit predictions on EvalAI

It is not possible to automaticaly compute the accuracies on the testset. You need to submit a json file on the [EvalAI platform](http://evalai.cloudcv.org/web/challenges/challenge-page/80/my-submission). The evaluation step on the testset creates the json file that contains the prediction of your model on the full testset. For instance: `logs/vqa2/murel_trainval_vg/results/test/epoch,19/OpenEnded_mscoco_test2015_model_results.json`. To get the accuracies on testdev or test sets, you must submit this file.


### VQACP2 dataset

#### Training and evaluation (train/val)

```
python -m bootstrap.run \
-o murel/options/vqacp2/murel.yaml \
--exp.dir logs/vqacp2/murel
```

#### Compare experiments on valset

```
python -m murel.compare_vqa_val -d logs/vqacp2/murel logs/vqacp2/attention
```

### TDIUC dataset

#### Training and evaluation (train/val/test)

The full training set is split into a trainset and a valset. At the end of the training, we evaluate our best checkpoint on the testset. The TDIUC metrics are computed and displayed at the end of each epoch. They are also stored in `logs.json` and `logs_test.json`.

```
python -m bootstrap.run \
-o murel/options/tdiuc/murel.yaml \
--exp.dir logs/tdiuc/murel

python -m bootstrap.run \
-o logs/tdiuc/murel/options.yaml \
--exp.resume best_eval_epoch.accuracy_top1 \
--dataset.train_split \
--dataset.eval_split test \
--misc.logs_name test
```

#### Compare experiments

You can compare experiments by displaying their best metrics on the valset or testset.

```
python -m murel.compare_tdiuc_val -d logs/tdiuc/murel logs/tdiuc/attention
python -m murel.compare_tdiuc_test -d logs/tdiuc/murel logs/tdiuc/attention
```

## Pretrained models

```
TODO
```


## Useful commands

### Use tensorboard instead of plotly

Instead of creating a `view.html` file, a tensorboard file will be created:
```
python -m bootstrap.run -o murel/options/vqa2/murel.yaml \
--view.name tensorboard
```

```
tensorboard --logdir=logs/vqa2
```

You can use plotly and tensorboard at the same time by updating the yaml file like [this one](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/options/mnist_plotly_tensorboard.yaml#L38).


### Use a specific GPU

For a specific experiment:
```
CUDA_VISIBLE_DEVICES=0 python -m boostrap.run -o murel/options/vqa2/murel.yaml
```

For the current terminal session:
```
export CUDA_VISIBLE_DEVICES=0
```

### Overwrite an option

The boostrap.pytorch framework makes it easy to overwrite a hyperparameter. In this example, we run an experiment with a non-default learning rate. Thus, I also overwrite the experiment directory path:
```
python -m bootstrap.run -o murel/options/vqa2/murel.yaml \
--optimizer.lr 0.0003 \
--exp.dir logs/vqa2/murel_lr,0.0003
```

### Resume training

If a problem occurs, it is easy to resume the last epoch by specifying the options file from the experiment directory while overwritting the `exp.resume` option (default is None):
```
python -m bootstrap.run -o logs/vqa2/murel/options.yaml \
--exp.resume last
```

### Web API

```
TODO
```

### Extract your own image features

```
TODO
```


## Citation

```
@InProceedings{Cadene_2019_CVPR,
    author = {Cadene, Remi and Ben-Younes, Hedi and Thome, Nicolas and Cord, Matthieu},
    title = {MUREL: {M}ultimodal {R}elational {R}easoning for {V}isual {Q}uestion {A}nswering},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition {CVPR}},
    year = {2019},
    url = {http://remicadene.com/pdfs/paper_cvpr2019.pdf}
}
```

## Poster

```
TODO
```

## Authors

This code was made available by [Hedi Ben-Younes](https://twitter.com/labegne) (Sorbonne-Heuritech), [Remi Cadene](http://remicadene.com) (Sorbonne), [Matthieu Cord](http://webia.lip6.fr/~cord) (Sorbonne) and [Nicolas Thome](http://cedric.cnam.fr/~thomen/) (CNAM).

## Acknowledgment

Special thanks to the authors of [VQA2](TODO), [TDIUC](TODO), [VisualGenome](TODO) and [VQACP2](TODO), the datasets used in this research project.
