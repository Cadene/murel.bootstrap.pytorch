import os
import csv
import copy
import json
import torch
import numpy as np
from tqdm import tqdm
from os import path as osp
from bootstrap.lib.logger import Logger
from block.datasets.vqa_utils import AbstractVQA

class VQACP2(AbstractVQA):

    def __init__(self,
            dir_data='data/vqa/vqacp2',
            split='train',
            batch_size=80,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            dir_rcnn='data/coco/extract_rcnn'):
        super(VQACP2, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split=proc_split,
            samplingans=samplingans,
            has_valset=True,
            has_testset=False,
            has_testdevset=False,
            has_testset_anno=False,
            has_answers_occurence=True,
            do_tokenize_answers=False)
        self.dir_rcnn = dir_rcnn

    def add_rcnn_to_item(self, item):
        path_rcnn = os.path.join(self.dir_rcnn, '{}.pth'.format(item['image_name']))
        item_rcnn = torch.load(path_rcnn)
        item['visual'] = item_rcnn['pooled_feat']
        item['coord'] = item_rcnn['rois']
        item['norm_coord'] = item_rcnn['norm_rois']
        item['nb_regions'] = item['visual'].size(0)
        return item

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset['questions'][index]
        #item['original_question'] = question
        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        item['image_name'] = question['image_name']

        # Process Object, Attribut and Relational features
        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            #item['original_annotation'] = annotation
            if 'train' in self.split and self.samplingans:
                proba = annotation['answers_count']
                proba = proba / np.sum(proba)
                item['answer_id'] = int(np.random.choice(annotation['answers_id'], p=proba))
            else:
                item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']

        return item

    def download(self):
        dir_ann = osp.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json -P' + dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json -P' + dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json -P' + dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json -P' + dir_ann)
        train_q = {"questions":json.load(open(osp.join(dir_ann, "vqacp_v2_train_questions.json")))}
        val_q = {"questions":json.load(open(osp.join(dir_ann, "vqacp_v2_test_questions.json")))}
        train_ann = {"annotations":json.load(open(osp.join(dir_ann, "vqacp_v2_train_annotations.json")))}
        val_ann = {"annotations":json.load(open(osp.join(dir_ann, "vqacp_v2_test_annotations.json")))}
        train_q['info'] = {}
        train_q['data_type'] = 'mscoco'
        train_q['data_subtype'] = "train2014cp"
        train_q['task_type'] = "Open-Ended"
        train_q['license'] = {}
        val_q['info'] = {}
        val_q['data_type'] = 'mscoco'
        val_q['data_subtype'] = "val2014cp"
        val_q['task_type'] = "Open-Ended"
        val_q['license'] = {}
        for k in ["info", 'data_type','data_subtype', 'license']:
            train_ann[k] = train_q[k]
            val_ann[k] = val_q[k]
        with open(osp.join(dir_ann, "OpenEnded_mscoco_train2014_questions.json"), 'w') as F:
            F.write(json.dumps(train_q))
        with open(osp.join(dir_ann, "OpenEnded_mscoco_val2014_questions.json"), 'w') as F:
            F.write(json.dumps(val_q))
        with open(osp.join(dir_ann, "mscoco_train2014_annotations.json"), 'w') as F:
            F.write(json.dumps(train_ann))
        with open(osp.join(dir_ann, "mscoco_val2014_annotations.json"), 'w') as F:
            F.write(json.dumps(val_ann))

    def add_image_names(self, dataset):
        for q in dataset['questions']:
            q['image_name'] = 'COCO_%s_%012d.jpg'%(q['coco_split'],q['image_id'])
        return dataset
