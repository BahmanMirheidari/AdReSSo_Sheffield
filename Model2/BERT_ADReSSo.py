import os
from src.data_preprocessing import *
from transformers import BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from src.BERT_train import fine_tune
from src.GPU_setup import device_setup
from src.BERT_predict import model_predict_test, model_predict_dev
from src.criteria_cal import *
from src.conf import *
from src.result_saving import result_write
import pdb
import csv
# Reading cfg file

def main():
    ## set up
    parser=OptionParser()
    parser.add_option("--Fold") 
    parser.add_option("--train_task")
    parser.add_option("--pre_trained_model")
    options, args = parser.parse_args()
    tokenizer, model_save_folder, results_save_folder, data_list_folder = make_path(options)
    device = device_setup()
    reset_random_seeds(SEED)

    ## data load
    label_dict = np.load(label_dict_path, allow_pickle=True).item()
    test_label_dict = np.load(label_dict_path, allow_pickle=True).item()
    
    print(options.pre_trained_model)
    train_lst_path = os.path.join(data_list_folder, str(options.Fold), 'train_list')
    train_lists = open(train_lst_path).readlines()
    dev_lst_path = os.path.join(data_list_folder, str(options.Fold), 'dev_list')
    dev_lists = open(dev_lst_path).readlines()
    test_lst_path = os.path.join(data_list_folder, test_list)
    test_lists = open(test_lst_path).readlines()

    train_dataloader = data_processing(train_lists, train_data_folder, label_dict, tokenizer, random=True)
    dev_dataloader = data_processing(dev_lists, train_data_folder, label_dict, tokenizer, random=True)
    test_dataloader = test_data_processing(test_lists, test_data_folder, tokenizer, random=False)

    model_saved_path = os.path.join(model_save_folder, str(options.Fold), 'best_model.pt')
    if os.path.exists(os.path.join(model_save_folder, str(options.Fold))) == False:
        os.mkdir(os.path.join(model_save_folder, str(options.Fold)))
    ## load pre-trained model
    model = BertForSequenceClassification.from_pretrained(
    options.pre_trained_model, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = output_hidden_states, # Whether the model returns all hidden-states.
    )
    model.cuda()
    ## fine tune the pre_trained model
    if classification_task == 'BERT_tunning':
        fine_tune(model, epochs, train_dataloader, dev_dataloader, device, model_saved_path)
    
    ## get the predict test result
    predictions, test_hidden_state_list = model_predict_test(model, test_dataloader, device, model_saved_path)
    predictions = type_check(predictions)
    pred_labels = np.argmax(predictions, axis=1).flatten()

    ## save predicted label
    label_save_path = os.path.join(results_save_folder, 'pred_labels', str(options.Fold), options.pre_trained_model+'_pred_labels')
    if os.path.exists(os.path.join(results_save_folder, 'pred_labels', str(options.Fold))) == False:
        os.makedirs(os.path.join(results_save_folder, 'pred_labels', str(options.Fold)))
    np.save(label_save_path, np.asarray(pred_labels))

    ## get the predict dev result
    true_labels, predictions, val_hidden_state_list = model_predict_dev(model, dev_dataloader, device, model_saved_path)
    val_accuracy = flat_accuracy(true_labels, predictions)
    true_labels = type_check(true_labels)
    predictions = type_check(predictions)
    pred_labels = np.argmax(predictions, axis=1).flatten()
    val_precision, val_recall, val_f_score, _ = precision_recall_fscore_support(true_labels, pred_labels)
    print("dev set", val_precision, val_recall, val_f_score, val_accuracy)

    ## save dev result    
    results_save_path = os.path.join(results_save_folder, options.pre_trained_model+'_val.csv')
    if str(options.Fold) == '0':
        with open(results_save_path,"a+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['test_data:', test_data_folder, 'train_data:', train_data_folder])
            writer.writerow(["precision_cn","precision_ad","recall_cn","recall_ad","f_score_cn","f_score_ad","accuracy"])    
    with open(results_save_path,"a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([val_precision[0], val_precision[1],val_recall[0],val_recall[1], val_f_score[0], val_f_score[1], val_accuracy])

if __name__ == "__main__":
    main()
