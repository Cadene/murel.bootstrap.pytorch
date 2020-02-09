import os
import re
import sys
import json
import copy
import torch
import torch.utils.data as data
import numpy as np
from os import path as osp
from tqdm import tqdm
from collections import Counter
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.datasets.dataset import Dataset
from bootstrap.datasets import transforms as bootstrap_tf
from bootstrap.datasets.dataset import ListDatasets


from .pythia_extract_vocabulary import extract_vocabulary, tokenize as pythia_tokenize
from .pythia_process_answers import word_tokenize as pythia_word_tokenize


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def tokenize_mcb(s):
    t_str = s.lower()
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        t_str = re.sub( i, '', t_str)
    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)
    q_list = re.sub(r'\?','',t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def tokenize_pythia(sentence):
    sentence = sentence.lower()
    sentence = (
        sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s'))
    tokens = SENTENCE_SPLIT_REGEX.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


class AbstractVQA(Dataset):

    def __init__(self,
            dir_data='/local/cadene/data/vqa',
            split='train', 
            batch_size=80,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            ques_max_len=None,
            proc_split='train',
            samplingans=False,
            has_valset=True,
            has_testset=True,
            has_testset_anno=False,
            has_testdevset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False):
        super(AbstractVQA, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle)
        self.nans = nans
        self.minwcount = minwcount
        self.nlp = nlp
        self.ques_max_len = ques_max_len
        self.proc_split = proc_split
        self.samplingans = samplingans
        # preprocessing
        self.has_valset = has_valset
        self.has_testset = has_testset
        self.has_testset_anno = has_testset_anno
        self.has_testdevset = has_testdevset
        self.has_answers_occurence = has_answers_occurence
        self.do_tokenize_answers = do_tokenize_answers

        self.pythia_preprocess = Options()['dataset'].get('pythia_preprocess', False)
        self.full_trainval = Options()['dataset'].get('full_trainval', False)

        # sanity checks
        if self.split in ['test', 'val'] and self.samplingans:
            raise ValueError()

        self.dir_raw = os.path.join(self.dir_data, 'raw')
        if not os.path.exists(self.dir_raw):
            self.download()

        self.dir_processed = os.path.join(self.dir_data, 'processed')
        self.subdir_processed = self.get_subdir_processed()
        self.path_wid_to_word = osp.join(self.subdir_processed, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subdir_processed, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subdir_processed, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subdir_processed, 'ans_to_aid.pth')
        self.path_trainset = osp.join(self.subdir_processed, 'trainset.pth')
        self.path_valset = osp.join(self.subdir_processed, 'valset.pth')
        self.path_is_qid_testdev = osp.join(self.subdir_processed, 'is_qid_testdev.pth')
        self.path_testset = osp.join(self.subdir_processed, 'testset.pth')
        
        if not os.path.exists(self.subdir_processed):
            if self.pythia_preprocess:
                self.process_pythia()
            else:
                self.process()

        self.wid_to_word = torch.load(self.path_wid_to_word)
        self.word_to_wid = torch.load(self.path_word_to_wid)
        self.aid_to_ans = torch.load(self.path_aid_to_ans)
        self.ans_to_aid = torch.load(self.path_ans_to_aid)

        if 'train' in self.split:
            self.dataset = torch.load(self.path_trainset)
        elif self.split == 'val':
            if self.proc_split == 'train':
                self.dataset = torch.load(self.path_valset)
            elif self.proc_split == 'trainval':
                self.dataset = torch.load(self.path_trainset)
        elif self.split == 'test':
            self.dataset = torch.load(self.path_testset)
            if self.has_testdevset:
                self.is_qid_testdev = torch.load(self.path_is_qid_testdev)

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.PadTensors(use_keys=[
                'question', 'pooled_feat', 'cls_scores', 'rois', 'cls', 'cls_oh', 'norm_rois'
            ]),
            #bootstrap_tf.SortByKey(key='lengths'), # no need for the current implementation
            bootstrap_tf.StackTensors()
        ])

        if self.proc_split == 'trainval' and self.split in ['train','val']:
            if not self.full_trainval:
                self.bootstrapping()

    def add_word_tokens(self, word_to_wid):
        for word, wid in word_to_wid.items():
            if word not in self.word_to_wid:
                self.word_to_wid[word] = len(self.word_to_wid)+1 # begin at 1, last is UNK
        self.wid_to_word = {wid:word for word, wid in self.word_to_wid.items()}

    def bootstrapping(self):
        rnd = np.random.RandomState(seed=Options()['misc']['seed'])
        indices = rnd.choice(len(self),
                             size=int(len(self)*0.95),
                             replace=False)
        if self.split == 'val':
            indices = np.array(list(set(np.arange(len(self))) - set(indices)))
        self.dataset['questions'] = [self.dataset['questions'][i] for i in indices]
        self.dataset['annotations'] = [self.dataset['annotations'][i] for i in indices]

    def __len__(self):
        return len(self.dataset['questions'])

    def get_image_name(self, image_id='1', format='COCO_%s_%012d.jpg'):
        return format%(self.get_subtype(), image_id)

    def name_subdir_processed(self):
        subdir = 'nans,{}_minwcount,{}_nlp,{}_proc_split,{}'.format(
            self.nans, self.minwcount, self.nlp, self.proc_split)
        if self.pythia_preprocess:
            subdir += "_pythia"
        return subdir

    def get_subdir_processed(self):
        name = self.name_subdir_processed()
        subdir = os.path.join(self.dir_processed, name)
        return subdir

    def get_subtype(self, testdev=False):
        if testdev:
            return 'test-dev2015'
        if self.split in ['train', 'val']:
            return self.split+'2014'
        elif self.split == 'test':
            return self.split+'2015'
        elif self.split == 'testdev':
            return 'test-dev2015'

    ################################################################################
    # Preprocessing

    def download(self):
        raise NotImplementedError()

    def process_pythia(self):
        dir_ann = osp.join(self.dir_raw, 'annotations')
        path_train_ann = osp.join(dir_ann, 'mscoco_train2014_annotations.json')
        path_train_ques = osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json')
        path_val_ann = osp.join(dir_ann, 'mscoco_val2014_annotations.json')
        path_val_ques = osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json')
        path_test_ques = osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json')
        path_test_ann = osp.join(dir_ann, 'mscoco_test2015_annotations.json')
        path_testdev_ques = osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json')
        

        from .pythia_extract_vocabulary import extract_vocabulary
        from .pythia_process_answers import process_answers

        wid_to_word, word_to_wid, wcounts = extract_vocabulary(
            [path_train_ques,  path_val_ques, path_test_ques]   
        )
        
        aid_to_ans, ans_to_aid = process_answers(path_train_ann, path_val_ann, min_freq=9)

        train_ann = json.load(open(path_train_ann))
        train_ques = json.load(open(path_train_ques))
        trainset = self.merge_annotations_with_questions(train_ann, train_ques)
        trainset = self.add_image_names(trainset)

        if self.has_valset:
            val_ann = json.load(open(path_val_ann))
            val_ques = json.load(open(path_val_ques))
            valset = self.merge_annotations_with_questions(val_ann, val_ques)
            valset = self.add_image_names(valset)

        if self.has_testset:
            test_ques = json.load(open(path_test_ques))
        
            if self.has_testset_anno:
                test_ann = json.load(open(path_test_ann))
                testset = self.merge_annotations_with_questions(test_ann, test_ques)
            else:
                testset = test_ques
            testset = self.add_image_names(testset)

        if self.has_testdevset:
            testdev = json.load(open(path_testdev_ques))
            testdev = self.add_image_names(testdev)

        #import pdb;pdb.set_trace()
        #if self.has_valset or self.has_testset:
        #    assert(len(all_questions) != len(trainset['questions']))

        if self.proc_split == 'trainval' and self.has_valset:
            trainset['questions'] += valset['questions']
            trainset['annotations'] += valset['annotations']

        trainset['annotations'] = self.pythia_add_answer(trainset['annotations'])
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.pythia_add_answer(valset['annotations'])
        if self.has_testset_anno:
            testset['annotations'] = self.pythia_add_answer(testset['annotations'])

        trainset['annotations'] = self.tokenize_answers(trainset['annotations'])
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.tokenize_answers(valset['annotations'])

        # top_answers = self.top_answers(trainset['annotations'], self.nans)
        # aid_to_ans = [a for i,a in enumerate(top_answers)]
        # ans_to_aid = {a:i for i,a in enumerate(top_answers)}

        trainset['questions'] = self.tokenize_questions(trainset['questions'], "pythia")
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.tokenize_questions(valset['questions'], "pythia")
        if self.has_testset:
            testset['questions'] = self.tokenize_questions(testset['questions'], "pythia")

        all_questions = copy.deepcopy(trainset['questions'])
        if self.proc_split == 'train' and self.has_valset:
            all_questions += valset['questions']
        if self.has_testset:
            all_questions += testset['questions']

        # Creation of word dictionary made before removing questions
        # Bigger dictionary == Better chance to model words from testset == Better generalization
        # top_words, wcounts = self.top_words(all_questions, self.minwcount)
        # wid_to_word = {i+1:w for i,w in enumerate(top_words)}
        # word_to_wid = {w:i+1 for i,w in enumerate(top_words)}

        # import pdb;pdb.set_trace()
        # assert(len(word_to_wid)==17871)

        # Remove the questions not in top answers
        # breakpoint()
        # trainset['annotations'], trainset['questions'] = self.annotations_in_top_answers(
        #     trainset['annotations'], trainset['questions'], aid_to_ans)

        # Logger()('DSODOUSDJSD', log_)
        # valset['annotations'], valset['questions'] = self.annotations_in_top_answers(
        #     valset['annotations'], valset['questions'], aid_to_ans)

        trainset['questions'] = self.insert_UNK_token(trainset['questions'], wcounts, 0)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.insert_UNK_token(valset['questions'], wcounts, 0)
        if self.has_testset:
            testset['questions'] = self.insert_UNK_token(testset['questions'], wcounts, 0)
        
        trainset['questions'] = self.encode_questions(trainset['questions'], word_to_wid)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.encode_questions(valset['questions'], word_to_wid)
        if self.has_testset:
            testset['questions'] = self.encode_questions(testset['questions'], word_to_wid)

        trainset['annotations'] = self.encode_answers(trainset['annotations'], ans_to_aid)
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.encode_answers(valset['annotations'], ans_to_aid)
        if self.has_testset_anno:
            testset['annotations'] = self.encode_answers(testset['annotations'], ans_to_aid)

        if self.has_answers_occurence:
            trainset['annotations'] = self.add_answers_occurence(trainset['annotations'], ans_to_aid)
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.add_answers_occurence(valset['annotations'], ans_to_aid)

        Logger()('Save processed datasets to {}'.format(self.subdir_processed))
        os.system('mkdir -p '+self.subdir_processed)
        if self.has_testdevset:
            is_qid_testdev = {item['question_id']:True for item in testdev['questions']}
            torch.save(is_qid_testdev, self.path_is_qid_testdev)
        
        
        torch.save(wid_to_word, self.path_wid_to_word)
        torch.save(word_to_wid, self.path_word_to_wid)
        torch.save(aid_to_ans, self.path_aid_to_ans)
        torch.save(ans_to_aid, self.path_ans_to_aid)
        torch.save(trainset, self.path_trainset)
        if self.proc_split == 'train' and self.has_valset:
            torch.save(valset, self.path_valset)
        if self.has_testset:
            torch.save(testset, self.path_testset)


    def process(self):
        dir_ann = osp.join(self.dir_raw, 'annotations')
        path_train_ann = osp.join(dir_ann, 'mscoco_train2014_annotations.json')
        path_train_ques = osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json')
        path_val_ann = osp.join(dir_ann, 'mscoco_val2014_annotations.json')
        path_val_ques = osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json')
        path_test_ques = osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json')
        path_test_ann = osp.join(dir_ann, 'mscoco_test2015_annotations.json')
        path_testdev_ques = osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json')
        
        train_ann = json.load(open(path_train_ann))
        train_ques = json.load(open(path_train_ques))
        trainset = self.merge_annotations_with_questions(train_ann, train_ques)
        trainset = self.add_image_names(trainset)

        if self.has_valset:
            val_ann = json.load(open(path_val_ann))
            val_ques = json.load(open(path_val_ques))
            valset = self.merge_annotations_with_questions(val_ann, val_ques)
            valset = self.add_image_names(valset)

        if self.has_testset:
            test_ques = json.load(open(path_test_ques))
        
            if self.has_testset_anno:
                test_ann = json.load(open(path_test_ann))
                testset = self.merge_annotations_with_questions(test_ann, test_ques)
            else:
                testset = test_ques
            testset = self.add_image_names(testset)

        if self.has_testdevset:
            testdev = json.load(open(path_testdev_ques))
            testdev = self.add_image_names(testdev)

        #import pdb;pdb.set_trace()
        #if self.has_valset or self.has_testset:
        #    assert(len(all_questions) != len(trainset['questions']))

        if self.proc_split == 'trainval' and self.has_valset:
            trainset['questions'] += valset['questions']
            trainset['annotations'] += valset['annotations']

        trainset['annotations'] = self.add_answer(trainset['annotations'])
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.add_answer(valset['annotations'])
        if self.has_testset_anno:
            testset['annotations'] = self.add_answer(testset['annotations'])

        if self.do_tokenize_answers:
            trainset['annotations'] = self.tokenize_answers(trainset['annotations'])
            if self.proc_split == 'train' and self.has_valset:
                valset['annotations'] = self.tokenize_answers(valset['annotations'])

        top_answers = self.top_answers(trainset['annotations'], self.nans)
        aid_to_ans = [a for i,a in enumerate(top_answers)]
        ans_to_aid = {a:i for i,a in enumerate(top_answers)}

        trainset['questions'] = self.tokenize_questions(trainset['questions'], self.nlp)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.tokenize_questions(valset['questions'], self.nlp)
        if self.has_testset:
            testset['questions'] = self.tokenize_questions(testset['questions'], self.nlp)

        all_questions = copy.deepcopy(trainset['questions'])
        if self.proc_split == 'train' and self.has_valset:
            all_questions += valset['questions']
        if self.has_testset:
            all_questions += testset['questions']

        # Creation of word dictionary made before removing questions
        # Bigger dictionary == Better chance to model words from testset == Better generalization
        top_words, wcounts = self.top_words(all_questions, self.minwcount)
        wid_to_word = {i+1:w for i,w in enumerate(top_words)}
        word_to_wid = {w:i+1 for i,w in enumerate(top_words)}

        # import pdb;pdb.set_trace()
        # assert(len(word_to_wid)==17871)

        # Remove the questions not in top answers
        trainset['annotations'], trainset['questions'] = self.annotations_in_top_answers(
            trainset['annotations'], trainset['questions'], aid_to_ans)

        # Logger()('DSODOUSDJSD', log_)
        # valset['annotations'], valset['questions'] = self.annotations_in_top_answers(
        #     valset['annotations'], valset['questions'], aid_to_ans)

        trainset['questions'] = self.insert_UNK_token(trainset['questions'], wcounts, self.minwcount)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.insert_UNK_token(valset['questions'], wcounts, self.minwcount)
        if self.has_testset:
            testset['questions'] = self.insert_UNK_token(testset['questions'], wcounts, self.minwcount)
        
        trainset['questions'] = self.encode_questions(trainset['questions'], word_to_wid)
        if self.proc_split == 'train' and self.has_valset:
            valset['questions'] = self.encode_questions(valset['questions'], word_to_wid)
        if self.has_testset:
            testset['questions'] = self.encode_questions(testset['questions'], word_to_wid)

        trainset['annotations'] = self.encode_answers(trainset['annotations'], ans_to_aid)
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.encode_answers(valset['annotations'], ans_to_aid)
        if self.has_testset_anno:
            testset['annotations'] = self.encode_answers(testset['annotations'], ans_to_aid)

        if self.has_answers_occurence:
            trainset['annotations'] = self.add_answers_occurence(trainset['annotations'], ans_to_aid)
        if self.proc_split == 'train' and self.has_valset:
            valset['annotations'] = self.add_answers_occurence(valset['annotations'], ans_to_aid)

        Logger()('Save processed datasets to {}'.format(self.subdir_processed))
        os.system('mkdir -p '+self.subdir_processed)
        if self.has_testdevset:
            is_qid_testdev = {item['question_id']:True for item in testdev['questions']}
            torch.save(is_qid_testdev, self.path_is_qid_testdev)
        torch.save(wid_to_word, self.path_wid_to_word)
        torch.save(word_to_wid, self.path_word_to_wid)
        torch.save(aid_to_ans, self.path_aid_to_ans)
        torch.save(ans_to_aid, self.path_ans_to_aid)
        torch.save(trainset, self.path_trainset)
        if self.proc_split == 'train' and self.has_valset:
            torch.save(valset, self.path_valset)
        if self.has_testset:
            torch.save(testset, self.path_testset)

    def tokenize_answers(self, annotations):
        Logger()('Example of modified answers after preprocessing:')
        for i, ex in enumerate(annotations):
            s = ex['answer']
            if self.nlp == 'nltk':
                ex['answer'] = " ".join(word_tokenize(str(s).lower()))
            elif self.nlp == 'mcb':
                ex['answer'] = " ".join(tokenize_mcb(s))
            elif self.nlp == "pythia":
                ex['answer'] = pythia_word_tokenize(s)
            else:
                ex['answer'] = " ".join(tokenize(s))
            if i < 10: Logger()('{} became -> {} <-'.format(s,ex['answer']))
            if i>0 and i % 1000 == 0:
                sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(annotations), i*100.0/len(annotations)) )
                sys.stdout.flush() 
        return annotations

    def add_image_names(self, dataset):
        for q in dataset['questions']:
            q['image_name'] = 'COCO_%s_%012d.jpg'%(dataset['data_subtype'],q['image_id'])
        return dataset

    def add_answer(self, annotations):
        for item in annotations:
            item['answer'] = item['multiple_choice_answer']
        return annotations

    def pythia_add_answer(self, annotations):
        for item in annotations:
            item['answer'] = pythia_word_tokenize(item['multiple_choice_answer'])
        return annotations


    def top_answers(self, annotations, nans):
        counts = {}
        for item in tqdm(annotations):
            ans = item['answer'] 
            counts[ans] = counts.get(ans, 0) + 1

        cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
        Logger()('Top answer and their counts:')
        for i in range(20):
            Logger()(cw[i])

        vocab = []
        for i in range(nans):
            vocab.append(cw[i][1])
        Logger()('Number of answers left: {} / {}'.format(len(vocab), len(cw)))
        return vocab[:nans]

    def annotations_in_top_answers(self, annotations, questions, top_answers):
        new_anno = []
        new_ques = []
        if len(annotations) != len(questions): # sanity check
            raise ValueError()
        for i in tqdm(range(len(annotations))):
            if annotations[i]['answer'] in top_answers:
                new_anno.append(annotations[i])
                new_ques.append(questions[i])
        Logger()('Number of examples reduced from {} to {}'.format(
            len(annotations), len(new_anno)))
        return new_anno, new_ques

    def tokenize_questions(self, questions, nlp):
        Logger()('Tokenize questions')
        if nlp == 'nltk':
            from nltk.tokenize import word_tokenize
        for item in tqdm(questions):
            ques = item['question']
            if nlp == 'nltk':
                item['question_tokens'] = word_tokenize(str(ques).lower())
            elif nlp == 'mcb':
                item['question_tokens'] = tokenize_mcb(ques)
            elif nlp == 'pythia':
                item['question_tokens'] = tokenize_pythia(ques)
            else:
                item['question_tokens'] = tokenize(ques)
        return questions

    def top_words(self, questions, minwcount):
        wcounts = {}
        for item in questions:
            for w in item['question_tokens']:
                wcounts[w] = wcounts.get(w, 0) + 1
        cw = sorted([(count,w) for w, count in wcounts.items()], reverse=True)
        Logger()('Top words and their wcounts:')
        for i in range(20):
            Logger()(cw[i])

        total_words = sum(wcounts.values())
        Logger()('Total words: {}'.format(total_words))
        bad_words = [w for w in sorted(wcounts) if wcounts[w] <= minwcount]
        vocab = [w for w in sorted(wcounts) if wcounts[w] > minwcount]
        bad_count = sum([wcounts[w] for w in bad_words])
        Logger()('Number of bad words: {}/{} = {:.2f}'.format(len(bad_words), len(wcounts), len(bad_words)*100.0/len(wcounts)))
        Logger()('Number of words in vocab would be {}'.format(len(vocab)))
        Logger()('Number of UNKs: {}/{} = {:.2f}'.format(bad_count, total_words, bad_count*100.0/total_words))
        vocab.append('UNK')
        return vocab, wcounts

    def merge_annotations_with_questions(self, ann, ques):
        for key in ann:
            if key not in ques:
                ques[key] = ann[key]
        return ques

    def insert_UNK_token(self, questions, wcounts, minwcount):
        unk_token = "UNK"
        if self.pythia_preprocess:
            unk_token = "<unk>"
        for item in questions:
            item['question_tokens_UNK'] = [w if wcounts.get(w,0) > minwcount else unk_token for w in item['question_tokens']]
        return questions

    def encode_questions(self, questions, word_to_wid):
        for item in questions:
            item['question_wids'] = [word_to_wid[w] for w in item['question_tokens_UNK']]
        return questions

    def encode_answers(self, annotations, ans_to_aid):
        for item in annotations:
            # associate to examples when answer is not in classes dictionnary
            # the class_id of the less occuring answer
            # Warning: the accuracy may be a bit wrong (to use only during validation)
            item['answer_id'] = ans_to_aid.get(item['answer'], len(ans_to_aid)-1)
        return annotations

    def add_answers_occurence(self, annotations, ans_to_aid):
        # for samplingans during training
        for item in annotations:
            item['answers_word'] = []
            item['answers_id'] = []
            item['answers_count'] = []
            answers = [a['answer'] for a in item['answers']]
            for ans, count in dict(Counter(answers)).items():
                if ans in ans_to_aid:
                    item['answers_word'].append(ans)
                    item['answers_id'].append(ans_to_aid[ans])
                    item['answers_count'].append(count)
        return annotations

    #############################################################################""
    # Preprocessing on a list dataset (vqa2+vgenome setup)

    def sync_from(self, dataset):
        Logger()('Sync {}.{} from {}.{}'.format(
            self.__class__.__name__,
            self.split,
            dataset.__class__.__name__,
            dataset.split))

        if 'annotations' in self.dataset:
            Logger()('Removing triplets with answer not in dict answer')
            list_anno = []
            list_ques = []
            for i in tqdm(range(len(self))):
                ques = self.dataset['questions'][i]
                answer = self.dataset['annotations'][i]['answer']
                if answer in dataset.ans_to_aid:
                    # sync answer_id
                    anno = self.dataset['annotations'][i]
                    anno['answer_id'] = dataset.ans_to_aid[answer]

                    list_anno.append(anno)
                    list_ques.append(ques)
            self.dataset['annotations'] = list_anno
            self.dataset['questions'] = list_ques
            Logger()('{} / {} remaining questions after filter_from'.format(len(list_ques), len(self.dataset['questions'])))

        # sync question_wids
        Logger()('Sync question_wids')
        for i in tqdm(range(len(self))):
            ques = self.dataset['questions'][i]
            for j, token in enumerate(ques['question_tokens']):
                if token in dataset.word_to_wid:
                    ques['question_tokens_UNK'][j] = token
                    ques['question_wids'][j] = dataset.word_to_wid[token]
                else:
                    ques['question_tokens_UNK'][j] = 'UNK'
                    ques['question_wids'][j] = dataset.word_to_wid['UNK']

        # sync dict word, dict ans
        self.word_to_wid = dataset.word_to_wid
        self.wid_to_word = dataset.wid_to_word
        self.ans_to_aid = dataset.ans_to_aid
        self.aid_to_ans = dataset.aid_to_ans


class ListVQADatasets(ListDatasets):

    def __init__(self,
             datasets,
             split='train',
             batch_size=4,
             shuffle=False,
             pin_memory=False,
             nb_threads=4,
             seed=1337):
        super(ListVQADatasets, self).__init__(
            datasets=datasets,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            bootstrapping=False,
            seed=seed)

        self.subdir_processed = self.make_subdir_processed()
        Logger()('Subdir proccessed: {}'.format(self.subdir_processed))
        self.path_wid_to_word = osp.join(self.subdir_processed, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subdir_processed, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subdir_processed, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subdir_processed, 'ans_to_aid.pth')

        self.process()
        
        # if not os.path.isdir(self.subdir_processed):
        #     self.process()
        # else:
        #     Logger()('Loading list_datasets_vqa proccessed state')
        #     self.wid_to_word = torch.load(self.path_wid_to_word)
        #     self.word_to_wid = torch.load(self.path_word_to_wid)
        #     self.aid_to_ans = torch.load(self.path_aid_to_ans)
        #     self.ans_to_aid = torch.load(self.path_ans_to_aid)

        #     for i in range(len(self.datasets)):
        #         subdir_processed = os.path.join(self.subdir_processed, '{}.{}'.format(
        #             self.datasets[i].__class__.__name__,
        #             self.datasets[i].split))
        #         path_dataset = os.path.join(subdir_processed, 'dataset.pth')
        #         self.datasets[i].dataset = torch.load(path_dataset)
        #     Logger()('Done !')

        Logger()('Final number of tokens {}'.format(len(self.word_to_wid)))

        self.make_lengths_and_ids()
        

    def make_subdir_processed(self):
        processed = ''
        for i in range(0, len(self.datasets)):
            processed += '{}.{}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split,
                self.datasets[i].name_subdir_processed())
            if i < len(self.datasets)-1:
                processed += '+'
        self.subdir_processed = osp.join(self.datasets[0].dir_processed, processed)
        return self.subdir_processed

    def process(self):
        os.system('mkdir -p '+self.subdir_processed)

        for i in range(1,len(self.datasets)):
            Logger()('Add word tokens of {}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split))
            self.datasets[0].add_word_tokens(self.datasets[i].word_to_wid)

        for i in range(len(self.datasets)):
            Logger()('Sync {}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split))
            self.datasets[i].sync_from(self.datasets[0])

        self.word_to_wid = self.datasets[0].word_to_wid
        self.wid_to_word = self.datasets[0].wid_to_word
        self.ans_to_aid = self.datasets[0].ans_to_aid
        self.aid_to_ans = self.datasets[0].aid_to_ans

        Logger()('Saving list_datasets_vqa proccessed state')
        torch.save(self.wid_to_word, self.path_wid_to_word)
        torch.save(self.word_to_wid, self.path_word_to_wid)
        torch.save(self.aid_to_ans, self.path_aid_to_ans)
        torch.save(self.ans_to_aid, self.path_ans_to_aid)

        for i in range(len(self.datasets)):
            subdir_processed = os.path.join(self.subdir_processed, '{}.{}'.format(
                self.datasets[i].__class__.__name__,
                self.datasets[i].split))
            os.system('mkdir -p '+subdir_processed)

            path_dataset = os.path.join(subdir_processed, 'dataset.pth')
            torch.save(self.datasets[i].dataset, path_dataset)
        Logger()('Done !')

    def get_subtype(self):
        return self.datasets[0].get_subtype()

