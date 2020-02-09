
# # Install maskrcnn-benchmark to extract detectron features
# %cd /content
# !git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
# %cd /content/vqa-maskrcnn-benchmark
# # Compile custom layers and build mask-rcnn backbone
# !python setup.py build
# !python setup.py develop
# sys.path.append('/content/vqa-maskrcnn-benchmark')


#!wget -O /content/model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth 
#!wget -O /content/model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
#!wget -O /content/model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd

import os
from os import path as osp
import json
import yacs

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
#from ipywidgets import widgets, Layout
from io import BytesIO


import sys
# TODO: PYTHON_PATH
#sys.path.append('/private/home/rcadene/doc/vqa-maskrcnn-benchmark')

from copy import deepcopy

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

from bootstrap.lib.logger import Logger

import argparse
from glob import glob
from tqdm import tqdm
from torchvision.datasets.folder import DatasetFolder
from torch.utils.data import DataLoader

def build_model(
        path_yaml='/content/model_data/detectron_model.yaml',
        path_ckpt='/content/model_data/detectron_model.pth'):

    cfg.merge_from_file(path_yaml)
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(path_ckpt, 
                            map_location=torch.device("cpu"))

    load_state_dict(model, checkpoint.pop("model"))

    model.to("cuda")
    model.eval()
    return model

def get_detectron_features(detectron_model, image_path):
    im, im_info, _= image_transform(image_path)
    img_tensor, im_infos = [im], [im_info]
    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to('cuda')
    with torch.no_grad():
        output = detectron_model(current_img_list)
    feat_list = process_feature_extraction(output, im_infos, 'fc6')#, 0.2)
    return feat_list

def image_transform(image_path):
    path = get_actual_image(image_path)
    img = Image.open(path)
    img = img.convert('RGB')
    im = np.array(img)
    try:
        im_norm = im[:, :, ::-1].astype(np.float32) - np.array([102.9801, 115.9465, 122.7717])
    except:
        import pdb;pdb.set_trace()
    im_shape = im.shape
    im_height = im_shape[0]
    im_width = im_shape[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im_resize = cv2.resize(
        im_norm,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    im_tensor = torch.from_numpy(im_resize).permute(2, 0, 1).float()
    im_info = {
        "width": im_width,
        "height": im_height,
        'scale': im_scale
    }
    return im_tensor, im_info, im

def get_actual_image(image_path):
    if image_path.startswith('http'):
        path = requests.get(image_path, stream=True).raw
    else:
        path = image_path
    return path

def process_feature_extraction(
        output,
        im_infos,
        feat_name='fc6',
        background=False,
        num_features=100,
        conf_thresh=0.0):
    batch_size = len(output[0]["proposals"])
    #assert(batch_size==1)
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    output = [output[0], output[1], {}]
    output[2]['proposals_resized'] = []
    output[2]['proposals'] = []
    output[2]['proposals_normalized'] = []
    #output[2]['proposals_normalized'] = []
    output[2][feat_name] = []
    output[2]['scores'] = []
    output[2]['num_boxes'] = []
    #output[2]['objects'] = []
    output[2]['image_scale'] = []
    output[2]['image_width'] = []
    output[2]['image_height'] = []
    output[2]['best_class_ids'] = []
    output[2]['best_scores'] = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_infos[i]['scale']
        scores = score_list[i]
        max_conf = torch.zeros((scores.shape[0])).to(cur_device)
        conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)

        if background:
            start_index = 0
        else:
            start_index = 1

        for cls_ind in range(start_index, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(
                (cls_scores[keep] > max_conf[keep]) &
                (cls_scores[keep] > conf_thresh_tensor[keep]),
                cls_scores[keep],
                max_conf[keep])

        sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
        num_boxes = (sorted_scores[:num_features] != 0).sum()
        keep_boxes = sorted_indices[:num_features]
        #objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)+1

        #keep_boxes = torch.argsort(max_conf, descending=True)[:100]
        proposals = output[0]['proposals'][i][keep_boxes].bbox
        output[2]['proposals_resized'].append(proposals)

        proposals = deepcopy(proposals)
        proposals /= im_infos[i]['scale']
        output[2]['proposals'].append(proposals)

        proposals = deepcopy(proposals)
        proposals[:,0] /= im_infos[i]['width']
        proposals[:,1] /= im_infos[i]['height']
        proposals[:,2] /= im_infos[i]['width']
        proposals[:,3] /= im_infos[i]['height']
        output[2]['proposals_normalized'].append(proposals)
        #output[0]['proposals'][i][keep_boxes].bbox / im_infos[i]['scale']

        #output[2]['proposals_normalized'].append()
        output[2][feat_name].append(feats[i][keep_boxes])
        output[2]['scores'].append(output[0]['scores'][keep_boxes])
        output[2]['num_boxes'].append(num_boxes)
        #output[2]['objects'].append(objects)
        output[2]['image_scale'].append(im_infos[i]['scale'])
        output[2]['image_width'].append(im_infos[i]['width'])
        output[2]['image_height'].append(im_infos[i]['height'])

        scores = torch.softmax(output[2]['scores'][i], dim=0)[:, 1:]
        best_scores = []
        best_class_ids = []
        for j in range(scores.shape[0]):
            sorted_ids = [idx.item() for idx in scores[j].argsort(descending=True)]
            best_scores.append(scores[j][sorted_ids[0]].item())
            best_class_ids.append(sorted_ids[0]+1)
        output[2]['best_scores'].append(best_scores)
        output[2]['best_class_ids'].append(best_class_ids)

    output[2][feat_name] = torch.cat(output[2][feat_name], dim=0)
    output[2]['scores'] = torch.cat(output[2]['scores'], dim=0)
    return output


# def build_vocab(path_vocab):
#     with open(path_vocab, 'r') as f:
#         vocab = ['__background__']
#         vocab += [line.strip() for line in f.readlines()]

#     cname_to_cid = {cname:i for i, cname in enumerate(vocab)}
#     cid_to_cname = {i:cname for i, cname in enumerate(vocab)}
#     return cname_to_cid, cid_to_cname

def build_vocab(path_vocab):
    with open(path_vocab, 'r') as f:
        data = json.load(f)

    vocab = ['__background__']
    vocab += [item['name'] for item in data['categories']]

    cname_to_cid = {cname:i for i, cname in enumerate(vocab)}
    cid_to_cname = {i:cname for i, cname in enumerate(vocab)}
    return cname_to_cid, cid_to_cname


class ImageDataset():

    def __init__(self, dir_images, name_images):
        self.dir_images = dir_images
        self.name_images = name_images

    def __getitem__(self, idx):
        name = self.name_images[idx]
        path = osp.join(self.dir_images, name)
        try:
            im_tensor, im_info, _ = image_transform(path)
        except:
            Logger()(path, log_level=Logger.WARNING)
            #os.system(f'rm {path}')
            im_tensor = None
            im_info = None
        return path, im_tensor, im_info

    def __len__(self):
        return len(self.name_images)


def slice_list(inlist, nslice):
    nelem_per_slice = int(len(inlist)/nslice)
    is_divisible = (len(inlist) % nslice == 0)
    n = nelem_per_slice
    listoflist = [inlist[i:i+n] for i in range(0, len(inlist), n)]
    if is_divisible:
        return listoflist
    left_items = listoflist.pop()
    for i, item in enumerate(left_items):
        listoflist[i].append(item)
    return listoflist


def test_slice_list():
    inlist = [1,2,3,4,5,6,7,8,9]
    nslice = 3
    outlist = slice_list(inlist, 3)
    print(outlist)
    assert outlist == [[1,2,3],[4,5,6],[7,8,9]]
    outlist = slice_list(inlist, 4)
    print(outlist)
    assert outlist == [[1,2,9],[3,4],[5,6],[7,8]]


def main(opt):
    thread_id = opt['thread_id']
    path_logs = f'{opt["dir_logs"]}/logs_split,{opt["split"]}_thread,{thread_id}'
    os.system('mkdir -p '+osp.dirname(path_logs))
    Logger(osp.dirname(path_logs), name=osp.basename(path_logs))

    #dir_data = opt['dir_data']#'data'
    #dir_dpythia = osp.join(dir_data, 'pythia')
    #dir_feats = osp.join(dir_data, 'vqa/coco/extract_rcnn/pythia_fixed_100')
    dir_dpythia = opt['dir_dpythia']
    dir_feats = opt['dir_feats']

    os.system(f'mkdir -p {dir_feats}')

    path_ckpt = osp.join(dir_dpythia, 'detectron_model.pth')
    path_yaml = osp.join(dir_dpythia, 'detectron_model.yaml')
    #path_vocab = osp.join(dir_dpythia, 'objects_vocab.txt')
    path_vocab = osp.join(dir_dpythia, 'visual_genome_categories.json')

    if not osp.isfile(path_ckpt) or not osp.isfile(path_ckpt):
        os.system(f'mkdir -p {dir_dpythia}')
        os.system(f'wget -O {path_ckpt} https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth')
        os.system(f'wget -O {path_yaml} https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml')

    cname_to_cid, cid_to_cname = build_vocab(path_vocab)
    detectron_model = build_model(path_yaml, path_ckpt)

    if 'vgenome' in opt['dir_images']:
        dir_images = opt['dir_images']
    else:
        if opt['split'] in ['train','val']:
            split = opt['split'] + '2014'
        else:
            split = opt['split'] + '2015'
        dir_images = osp.join(opt['dir_images'], split)

    Logger()('Load image names')
    name_images = [osp.basename(path) for path in glob(osp.join(dir_images,'*.jpg'))]
    len_found_images = len(name_images)
    Logger()(f'{len_found_images} images found')

    name_images = [name for name in name_images if not osp.isfile(osp.join(dir_feats, name+'.pth'))]
    len_not_extract = len(name_images)
    Logger()(f'{len_not_extract}/{len_found_images} images not already extracted')

    Logger()('Sort image names')
    name_images = sorted(name_images)

    nb_threads = opt['nb_threads']
    name_images_sliced = slice_list(name_images, nb_threads)
    name_images = name_images_sliced[thread_id]
    Logger()(f'Thread #{thread_id} ({thread_id+1}/{nb_threads}) run on {len(name_images)} images')

    dataset = ImageDataset(dir_images, name_images)

    def collate_fn(x):
        paths = [x[i][0] for i in range(len(x))]
        im_tensor = [x[i][1].detach_() for i in range(len(x))]
        im_infos = [x[i][2] for i in range(len(x))]
        im_tensor = to_image_list(im_tensor, size_divisible=32)
        return paths, im_tensor, im_infos

    #collate_fn=lambda x: (x, x, x)

    batch_size = 4
    batch_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn, num_workers=1)

    Logger()('Extraction BEGIN')
    nb_items = 0
    for paths, im_tensor, im_infos in batch_loader:
        im_tensor = im_tensor.to('cuda')
        with torch.no_grad():
            output = detectron_model(im_tensor)
            detectron_features = process_feature_extraction(output, im_infos, 'fc6')

        bsize = len(detectron_features[2]['proposals'])
        n_boxes_per_image = detectron_features[2]['proposals'][0].shape[0]
        feats = detectron_features[2]['fc6'].split(n_boxes_per_image)
        scores = detectron_features[2]['scores'].split(n_boxes_per_image)

        for i in range(bsize):
            item = {}
            item['pooled_feat'] = feats[i].cpu()
            item['rois'] = detectron_features[2]['proposals'][i].cpu()
            item['norm_rois'] = detectron_features[2]['proposals_normalized'][i].cpu()
            item['all_scores'] = scores[i].cpu()
            item['cls_scores'] = torch.FloatTensor(detectron_features[2]['best_scores'][i])
            item['cls'] = torch.FloatTensor(detectron_features[2]['best_class_ids'][i])

            img_name = paths[i].split('/')[-1]
            path_item = osp.join(dir_feats, img_name+'.pth')
            torch.save(item, path_item)
            nb_items += 1
            Logger()(f'nb_images: {nb_items}/{len(name_images)}')
        
    Logger()('Extraction END')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_logs', default='logs/extract_detectron')
    parser.add_argument('--dir_dpythia', default='data/pythia')
    parser.add_argument('--dir_feats', default='data/vqa/coco/extract_rcnn/pythia_fixed_100')
    parser.add_argument('--dir_images', default='/private/home/rcadene/data/rubi.bootstrap.pytorch/vqa/coco/raw')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--nb_threads', default=16, type=int)
    parser.add_argument('--thread_id', default=0, type=int)
    opt = vars(parser.parse_args())
    #main(opt)

    from tqdm import tqdm
    name_images = [osp.basename(path) for path in glob(osp.join(opt['dir_feats'],'*.pth'))]
    for name in tqdm(name_images):
        try:
            torch.load(osp.join(opt['dir_feats'], name))
        except:
            print(name)
