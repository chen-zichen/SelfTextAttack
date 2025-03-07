# Quiet TensorFlow.
import os

import numpy as np
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import textattack
from textattack import Attacker
from textattack.attack_recipes import MyBAEGarg2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper, huggingface_model_wrapper
from textattack.models.wrappers import HuggingFaceModelWrapper

def load_dataset_sst(path = 'repos/text_pgd_attack/ag/'):
    def process_file(file):    
        # sentence_list = []
        # label_list = []
        data_list = []
        with open(path + file,'r',encoding = 'utf-8') as f:
            for line in f:
                sen, label = line.split("\t",1)
                data_item = [sen, int(label)]
                data_list.append(data_item)
        return data_list
    test_dataset = process_file("test.tsv")
    return test_dataset

directory = 'repos/std_text_pgd_attack/checkpoints/albert-xxlarge-v2-ag'
model = AlbertForSequenceClassification.from_pretrained(directory)
tokenizer = AlbertTokenizer.from_pretrained(directory)
wrapper_model = huggingface_model_wrapper.HuggingFaceModelWrapper(model, tokenizer)
recipe = MyBAEGarg2019.build(wrapper_model)

# dataset = HuggingFaceDataset("allocine", split="test")
dataset = load_dataset_sst()
dataset = textattack.datasets.Dataset(dataset)

attack_args = textattack.AttackArgs(num_examples = -1, log_to_txt = './log/bae_ag_albertxxlargev2.txt', query_budget = 500)
attacker = Attacker(recipe, dataset, attack_args)
results = attacker.attack_dataset()

