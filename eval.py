import numpy as np
import torch
import torch.nn.functional as F

from textattack.models.bert_models import BertSModel, Attacker

from transformers import BertTokenizer, BertForSequenceClassification, AdamW

import os

from tqdm import tqdm


class Config():
    model_type = 'bert-base-uncased'
    output_dir = 'repos/TextAttack/checkpoints/bert-sst-at/'
    output_dir_rb = 'repos/TextAttack/checkpoints/bert-sst-at/rb'
    dataset_dir = 'repos/text_grad/sst-2/'
    cache_dir = 'model_cache/bert_model/bert-base-uncased/'
    finetune_dir = 'repos/text_grad/checkpoints/bert-base-uncased-sst/'
    num_labels = 2
    log_dir = 'ATLog/test'
    saved_model = 'repos/TextAttack/checkpoints/bert-sst-at'

    # at_type = 'augmentation'  ## augmentation/epoch_aug/batch_aug
    at_type = 'epoch_aug'  ## augmentation/epoch_aug/batch_aug
    # at_type = 'batch_aug'  ## augmentation/epoch_aug/batch_aug

    num_epochs = 5
    batch_size = 32

    epoch = 0


def write_adv(file_addr, orig_list, adv_list, label_list):
    with open(file_addr, 'w', encoding = 'utf-8') as f:
        for i in range(len(orig_list)):
            orig = orig_list[i]
            adv = adv_list[i]
            label = label_list[i]
            f.write(orig + '\n')
            f.write(adv + '\n')
            f.write(str(label) + '\n\n')

# test robust acc
config = Config()
device = torch.device("cuda")
dataset_dir = config.dataset_dir + "/" + str(config.epoch) + "/"
cls_model = BertSModel(model_type = config.model_type, output_dir = config.output_dir, cache_dir = config.cache_dir,
                    dataset_dir = config.dataset_dir, num_labels = config.num_labels, device = device)

finetune_dir = config.saved_model + "/" + str(config.epoch) + "/"
if config.at_type in ['augmentation', 'epoch_aug']:
    surrogate_model = BertSModel(fine_tune_dir = finetune_dir, num_labels = config.num_labels, device = device)
    attacker = Attacker(victim_model = surrogate_model.model, tokenizer = surrogate_model.tokenizer)
log_dir = config.log_dir
batch_size = config.batch_size


# data
_,_,_,_,test_corpus, test_label = cls_model.load_dataset()

test_set = [(test_corpus[i], test_label[i]) for i in range(len(test_corpus))]

if config.at_type in ['augmentation', 'epoch_aug']:
    surrogate_model.model.eval()
    surrogate_model.eval_on_test()
    print("generate adversarial examples for epoch 0...")
    perturbed_examples = attacker.perturb(test_set, visualize = True)
    file_addr = log_dir + 'aug_epoch0.txt'
    write_adv(file_addr, test_corpus, perturbed_examples, test_label)

    perturbed_label = test_label[:]
    concat_test_corpus = test_corpus + perturbed_examples
    concat_test_label = test_label + perturbed_label
    concat_test_xs, concat_train_masks = cls_model.tokenize_corpus(concat_test_corpus)
    concat_test_ys = np.array(concat_test_label)

else:
    test_xs, test_masks = cls_model.tokenize_corpus(test_corpus)
    test_ys = np.array(test_label)

# prepare data
test_xs, test_masks = cls_model.tokenize_corpus(test_corpus)
test_ys = np.array(test_label)

for epoch in range(config.num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    cls_model.model.eval()
    perturbed_examples = attacker.perturb(test_set,)
    perturbed_label = test_label[:]
    concat_test_corpus = test_corpus + perturbed_examples
    concat_test_label = test_label + perturbed_label

    concat_test_xs, concat_train_masks = cls_model.tokenize_corpus(concat_test_corpus)
    concat_test_ys = np.array(concat_test_label)

    num_examples = concat_test_xs.shape[0]
    selection = np.random.choice(num_examples,size = num_examples,replace = False)
    batches_per_epoch = num_examples // batch_size

    for idx in tqdm(range(batches_per_epoch)):
        batch_idx = selection[idx * batch_size:(idx + 1) * batch_size]
        batch_corpus = [test_corpus[x] for x in batch_idx]
        batch_labels = [test_label[x] for x in batch_idx]

        batch_instances = [test_set[x] for x in batch_idx]

        adv_corpus = attacker.perturb(batch_instances)
        adv_labels = batch_labels[:]
        concat_batch_corpus = batch_corpus + adv_corpus
        concat_batch_labels = batch_labels + adv_labels

        concat_xs, concat_masks = cls_model.tokenize_corpus(concat_batch_corpus)
        batch_xs = torch.LongTensor(concat_xs).to(cls_model.device)
        batch_masks = torch.LongTensor(concat_masks).to(cls_model.device)
        batch_ys = torch.LongTensor(concat_batch_labels).to(cls_model.device)
        # X and y
        batch_order = torch.randperm(batch_xs.size(0))
        batch_xs = batch_xs[batch_order]
        batch_masks = batch_masks[batch_order]
        batch_ys = batch_ys[batch_order]

        result = cls_model.model(input_ids = batch_xs,labels = batch_ys,attention_mask = batch_masks)
        loss = result.loss
        logits = result.logits

        epoch_loss += loss.item()
        epoch_accuracy += torch.argmax(logits,dim = 1).eq(batch_ys).sum().item()/batch_size

    epoch_loss /= batches_per_epoch
    epoch_accuracy /= batches_per_epoch      
    print(epoch,' ',epoch_loss, ' ',epoch_accuracy)  
