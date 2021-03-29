import numpy as np
from src.conf import *
import csv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
def result_write(pre_list, rc_list, f_score_list, acc_list, results_save_path):
    avg_pre = np.mean(np.asarray(pre_list))
    avg_rc = np.mean(np.asarray(rc_list))
    avg_f1 = np.mean(np.asarray(f_score_list))
    avg_acc = np.mean(np.asarray(acc_list))
    print("Average precision: {0:.2f}".format(avg_pre))
    print("Average recall: {:}".format(avg_rc))
    print("Average F score: {0:.2f}".format(avg_f1))    
    print("Average accuracy : {0:.2f}".format(avg_acc))    
    # dataframe = pd.DataFrame({'precision':avg_pre,'recall':avg_rc,'f1':avg_f1,'accuracy':avg_acc})
    # dataframe.to_csv(results_save_path,index=True,sep=',')
    with open(results_save_path,"w") as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(["precision","recall","f_score","accuracy"])
        for pre,rc,f1,acc in zip(pre_list, rc_list, f_score_list, acc_list):
            writer.writerow([pre,rc,f1,acc])
            
        writer.writerow([avg_pre, avg_rc, avg_f1, avg_acc])
def result_cal(model, test_matrix, test_labels):

    pred_score = model.predict(test_matrix)
    pred_label = np.argmax(pred_score, axis=1)
    if len(test_labels[0]) == 2:
        test_labels = np.argmax(test_labels, axis=1)
    val_precision, val_recall, val_f_score, _ = precision_recall_fscore_support(test_labels, pred_label)
    val_accuracy = accuracy_score(test_labels, pred_label)
    print(val_precision, val_recall, val_f_score, val_accuracy)
    return val_precision, val_recall, val_f_score, val_accuracy