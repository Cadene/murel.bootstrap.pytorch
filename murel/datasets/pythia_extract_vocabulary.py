# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import json
import os
from collections import Counter
import re
from tqdm import tqdm
from bootstrap.lib.logger import Logger


SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")

def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=["'s"], remove=[",", "?"]):
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def get_text(input_files):
    """
    Override this in your child class to extract custom text
    Default for VQA. Make sure to return a list of all possible text
    """
    text = []

    for input_file in input_files:
        with open(input_file, "r") as f:
            text += json.load(f)["questions"]

    return text

def save_vocabulary(out_dir, vocabulary, vocab_file_name):
    vocab_file = os.path.join(out_dir, vocab_file_name)
    with open(vocab_file, "w") as f:
        f.writelines([w + "\n" for w in vocabulary])

def extract_vocabulary(input_files, min_freq=0):
    # os.makedirs(out_dir, exist_ok=True)
    Logger()('Extracting vocabulary BEGIN')
    word_count = Counter()

    questions = get_text(input_files)
    question_length = [None] * len(questions)

    for inx, question in enumerate(tqdm(questions)):
        words = tokenize(question["question"])
        question_length[inx] = len(words)
        word_count.update(words)

    vocabulary = [w[0] for w in word_count.items() if w[1] >= min_freq]
    vocabulary.sort()
    vocabulary = ["<unk>"] + vocabulary

    wid_to_words = vocabulary
    words_to_wid = {}
    for i, word in enumerate(wid_to_words):
        words_to_wid[word] = i
    Logger()('Extracting vocabulary END')
    return wid_to_words, words_to_wid, word_count
    # print("min text len=", min(text_lengths))
    # print("max text len=", max(text_lengths))

    # save_vocabulary(out_dir, vocabulary, vocab_file_name)

