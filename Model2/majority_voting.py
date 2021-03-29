import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import pdb
#f = sys.argv[1]
pred_label_folder = os.path.join('/mnt/fastdata/ac1yp/bert/ADReSSo_test/ADReSSo/wav2vec_version/wav2vec_transcripts/results/pred_labels/')
label_dict_path = '/data/ac1yp/data/cookie_theft/lists/ADReSSo_test/ADReSSo/list/label_dict.npy'
test_list_path = '/data/ac1yp/data/cookie_theft/lists/ADReSSo_test/ADReSSo/list/ADReSSo_list'
output_dict_path = '/mnt/fastdata/ac1yp/bert/ADReSSo_test/ADReSSo/wav2vec_version/wav2vec_transcripts/results/pred_labels/test_pred_label_dict.npy'

def result_save(test_list, pred_label_list):
    pred_label_dict = {}
    for test_id, pred_label in zip(test_list,pred_label_list):
        pred_label_dict[test_id[:-1]] = pred_label
    np.save(output_dict_path,pred_label_dict)

label_dict = np.load(label_dict_path, allow_pickle=True).item()
test_list = open(test_list_path).readlines()
test_label_list = []
for test_id in test_list:
    test_label = label_dict[test_id[:-1]]
    test_label_list.append(test_label)

Fold = 10
pred_label_matrix = []
f_score_list = []
for idx in range(Fold):
    pred_label_path = os.path.join(pred_label_folder, str(idx), 'bert-base-uncased_pred_labels.npy')
    pred_label = np.load(pred_label_path)
    accuracy = accuracy_score(test_label_list, pred_label)
    pre, rec, f_1, U = precision_recall_fscore_support(test_label_list, pred_label)
    f_score_list.append(accuracy)
    pred_label_matrix.append(pred_label)

test_label_list = np.asarray(test_label_list)
pred_label_matrix = np.asarray(pred_label_matrix)

pred_label_list = np.mean(pred_label_matrix, axis = 0)
pred_label_list = np.asarray([round(x) for x in pred_label_list])
result_save(test_list, pred_label_list)
