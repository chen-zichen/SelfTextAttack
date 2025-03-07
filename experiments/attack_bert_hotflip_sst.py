# Quiet TensorFlow.
import os

import numpy as np
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import textattack
from textattack import Attacker
from textattack.attack_recipes import MyHotFlipEbrahimi2017
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper, huggingface_model_wrapper
from textattack.models.wrappers import HuggingFaceModelWrapper

def load_dataset_sst(path = 'repos/text_pgd_attack/sst-2/'):
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
    train_dataset = process_file("train.tsv")
    valid_dataset = process_file("valid.tsv")
    test_dataset = process_file("test.tsv")
    return test_dataset

directory = 'repos/text_pgd_attack/checkpoints/bert-base-uncased-sst'
model = BertForSequenceClassification.from_pretrained(directory)
tokenizer = BertTokenizer.from_pretrained('repos/text_pgd_attack/checkpoints/bert-base-uncased-sst')
wrapper_model = huggingface_model_wrapper.HuggingFaceModelWrapper(model, tokenizer)
recipe = MyHotFlipEbrahimi2017.build(wrapper_model)

# dataset = HuggingFaceDataset("allocine", split="test")
dataset = load_dataset_sst()
dataset = textattack.datasets.Dataset(dataset)

attack_args = textattack.AttackArgs(num_examples = 100, log_to_txt = '../log/bertattack_sst_bertbase.txt', query_budget = 500)
attacker = Attacker(recipe, dataset, attack_args)
results = attacker.attack_dataset()

