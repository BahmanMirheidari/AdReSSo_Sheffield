import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
def data_load(data_list, feats_dict_folder, label_dict):
    feat_list = []
    label_list = []
    for name in data_list:
        feat_path = os.path.join(feats_dict_folder, name[:-1]+'.npy')
        feat = np.load(feat_path, allow_pickle=True)
        avg_feat = np.mean(feat, axis = 0)
        feat_list.append(avg_feat)
        label = label_dict[name[:-1]]
        label_list.append(label)
    return np.asarray(feat_list), np.asarray(label_list)

def test_data_load(data_list, feats_dict_folder):
    feat_list = []
    for name in data_list:
        feat_path = os.path.join(feats_dict_folder, name[:-1]+'.npy')
        feat = np.load(feat_path, allow_pickle=True)
        avg_feat = np.mean(feat, axis = 0)
        feat_list.append(avg_feat)

    return np.asarray(feat_list)


acoustic_feat_idx = sys.argv[1]
feats_dict_folder = os.path.join('/mnt/fastdata/ac1yp/ADReSSo/version_3/wav2vec_feats/', str(acoustic_feat_idx))
list_path = '/data/ac1yp/data/cookie_theft/lists/ADReSSo_test/ADReSSo/list'
label_dict_path = '/data/ac1yp/data/cookie_theft/label_dict_full.npy'
results_save_folder = os.path.join('/mnt/fastdata/ac1yp/ADReSSo/version_3/wav2vec_feats/', str(acoustic_feat_idx))
if os.path.exists(results_save_folder) == False:
    os.makedirs(results_save_folder)
label_dict = np.load(label_dict_path, allow_pickle=True).item()

Fold = 10
accuracy_list = []
recall_list = []
f_score_list = []
precision_list = []


test_label_matrix = []
for idx in range(Fold):
    train_list = open(os.path.join(list_path, str(idx), 'train_list')).readlines()
    dev_list = open(os.path.join(list_path, str(idx), 'dev_list')).readlines()
    test_list = open(os.path.join(list_path, 'mapped_DB_list')).readlines()


    train_feats, train_labels = data_load(train_list, feats_dict_folder, label_dict)
    dev_feats, dev_labels = data_load(dev_list, feats_dict_folder, label_dict)
    test_feats = test_data_load(test_list, feats_dict_folder)

    # decision tree
    DT =  make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
    SVM = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    TB = make_pipeline(StandardScaler(), BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0))
    KNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

    lr = TB

    lr.fit(train_feats, train_labels)
    pred_labels = lr.predict(dev_feats)
    precision, recall, f_score, _ = precision_recall_fscore_support(dev_labels, pred_labels)
    accuracy = accuracy_score(dev_labels, pred_labels)

    accuracy_list.append(accuracy)
    recall_list.append(recall)
    f_score_list.append(f_score)
    precision_list.append(precision)


    test_pred_labels = lr.predict(test_feats)


    # save predicted label
    label_save_path = os.path.join(results_save_folder, 'pred_labels', str(idx), 'pred_labels')
    if os.path.exists(os.path.join(results_save_folder, 'pred_labels', str(idx))) == False:
        os.makedirs(os.path.join(results_save_folder, 'pred_labels', str(idx)))
    np.save(label_save_path, np.asarray(test_pred_labels))


accuracy_list = np.asarray(accuracy_list)
recall_list = np.asarray(recall_list)
f_score_list = np.asarray(f_score_list)
precision_list = np.asarray(precision_list)
print(np.mean(precision_list, axis = 0), np.mean(recall_list, axis = 0),np.mean(f_score_list, axis = 0), np.mean(accuracy_list))





