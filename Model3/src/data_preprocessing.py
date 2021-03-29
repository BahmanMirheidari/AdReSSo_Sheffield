import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import re
import torch
from src.conf import *
from sklearn.preprocessing import StandardScaler
from src.Models import  TransformerBlock, TokenAndPositionEmbedding
from keras.utils.np_utils import to_categorical
import pdb
from tensorflow.keras import layers
import random
import tensorflow as tf
from sklearn.preprocessing import normalize
def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\+","",string)
    string = re.sub(r"\?","",string)
    string = re.sub(r"\//","",string)
    return string.strip().lower()

def test_data_load(data_list, data_folder):
    texts = []
    for item in data_list:
        name = item[:-1]
        trans_path = os.path.join(data_folder, name)
        transcript = open(trans_path).readlines()[0]
        transcript = clean_str(transcript)
        texts.append(transcript)
    return texts

def data_load(data_list, data_folder, label_dict):
    texts = []
    labels = []
    for item in data_list:
        name = item[:-1]
        label = label_dict[name]
        trans_path = os.path.join(data_folder, name)
        transcript = open(trans_path).readlines()[0]
        transcript = clean_str(transcript)
        texts.append(transcript)
        labels.append(label)
    return labels, texts


def sentence_tokenize(sentences, tokenizer, add_special_tokens):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    # For every sentence...
    for sent in sentences:
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = add_special_tokens, # Add '[CLS]' and '[SEP]'
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    input_ids = pad_sequences(input_ids, maxlen=MAX_TRAN_LEN, dtype="long",
                              value=0, truncating="post", padding="post")
    #print('\Done.')
    
    return input_ids
def sentence_mask(input_ids):
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
        
    return attention_masks

def wav2vec_load(data_list, feats_folder):
    feats = []
    for item in data_list:
        name = item[:-1]
        feats_path = os.path.join(feats_folder, name+'.npy')
        feat = np.load(feats_path)
        avg_feat = np.mean(feat, axis = 0)
        feats.append(avg_feat)
    feats = np.asarray(feats)
    feats = normalize(feats)
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    return feats


def fusion_data_load(data_list, data_folder, wav2vec_feats_folder, label_dict, tokenizer):
    labels, texts = data_load(data_list, data_folder,label_dict)
    wav2vec_feats = wav2vec_load(data_list, wav2vec_feats_folder)

    inputs_id = sentence_tokenize(texts, tokenizer, add_special_tokens=True)
    attention_masks=sentence_mask(inputs_id)

    inputs_id = torch.Tensor(inputs_id).to(torch.int64)
    attention_masks = torch.Tensor(attention_masks).to(torch.int64)
    labels = torch.Tensor(labels).to(torch.int64)
    wav2vec_feats = torch.Tensor(wav2vec_feats).to(torch.float32)

    data = TensorDataset(inputs_id, wav2vec_feats, attention_masks, labels)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size = batch_size)

    return dataloader

def test_fusion_data_load(data_list, data_folder, wav2vec_feats_folder, tokenizer):
    texts = test_data_load(data_list, data_folder)
    wav2vec_feats = wav2vec_load(data_list, wav2vec_feats_folder)

    inputs_id = sentence_tokenize(texts, tokenizer, add_special_tokens=True)
    attention_masks=sentence_mask(inputs_id)

    inputs_id = torch.Tensor(inputs_id).to(torch.int64)
    attention_masks = torch.Tensor(attention_masks).to(torch.int64)
    wav2vec_feats = torch.Tensor(wav2vec_feats).to(torch.float32)
    data = TensorDataset(inputs_id, wav2vec_feats, attention_masks)
    dataloader = DataLoader(data, batch_size = batch_size)
    return dataloader