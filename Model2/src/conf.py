from transformers import BertTokenizer
import os
import argparse
import configparser as ConfigParser
from optparse import OptionParser
import pdb

test_dataset = 'wav2vec_version/wav2vec_transcripts'
test_list = 'ADReSSo_list'
classification_task = 'BERT_tunning'
model_type = 'BERT'
batch_size = 4 # should be global variable
drop_rate = 0.3
epochs = 8
l2 = 0.0001
MAX_TRAN_LEN = 256
class_num = 2
SEED = 12345678
data_folder = "/data/ac1yp/data/cookie_theft/all_transcripts"
train_data_folder = "/fastdata/ac1yp/ADReSSo/wav2vec_version/wav2vec_transcripts"
test_data_folder = os.path.join("/fastdata/ac1yp/ADReSSo/", test_dataset)
label_dict_path = "/data/ac1yp/data/cookie_theft/lists/ADReSSo_test/ADReSSo/list/label_dict.npy"
wav2vec_feats_folder = '/fastdata/ac1yp/ADReSSo/wav2vec_version/wav2vec_feats/'
def make_path(options):
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(options.pre_trained_model, do_lower_case=True)
    root_folder = "/fastdata/ac1yp/bert"
    model_save_folder = os.path.join(root_folder, options.train_task, test_dataset, "fine_tune_models")
    results_save_folder = os.path.join(root_folder, options.train_task, test_dataset, "results")
    
    if os.path.exists(model_save_folder) == False:
        os.makedirs(model_save_folder)
    if os.path.exists(results_save_folder) == False:
        os.makedirs(results_save_folder)

    data_list_folder = os.path.join("/data/ac1yp/data/cookie_theft/lists/", options.train_task, "list")

    return tokenizer, model_save_folder, results_save_folder, data_list_folder

