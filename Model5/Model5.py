
# coding: utf-8

# # AdRESSO21

# ## Preparing the train and test sets

# Since we are interested in the task 1 of the challenge, we use only the diagnostic data.
# The structure of train_files are as provided by the challenge authors.

# First import the main python packages needed as well the root directory paths.

# In[ ]:

import os
import glob
import tensorflow as tf
import numpy as np
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig
from tensorflow.keras import layers
from tensorflow import keras
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

SEED = 12345678  
#SEED = 352389
#SEED = 756897
#SEED = 4646786
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(SEED)
   tf.random.set_seed(SEED)
   np.random.seed(SEED) 
    
# set random seeds for reproducibility
reset_random_seeds()

# root directory that contains all of the sub directories for the challenge
#root_dir = "/Users/bahmanmirheidari/Documents/AdRESSO/AdRESSO21"
root_dir = "/fastdata/ac1bm/AdRESSO/AdRESSO/AdRESSO21"

# path to the challenge 2021 train set
#train_files_path = root_dir + "/train_files"
train_files_path = root_dir + "/../train_files" 

# path to the challenge 2021 test set
#test_files_path = root_dir + "/test_files"
test_files_path = root_dir + "/../test_files"  

# path to the train audio files
train_audio_files_path = train_files_path + "/diagnosis/train/audio"  

# path to the test audio files
test_audio_files_path = test_files_path + "/audio"  

# path to the ASR outputs for the train wave files
train_asr_output_path = root_dir +"/ASR_Results/adresso_train"

# path to the ASR outputs for the test wave files
test_asr_output_path = root_dir +"/ASR_Results/adresso_test"

# maximum text length
MAX_TEXT_LENGTH = 200 

# use 20% of train set for evaluation set
validation_split = 0.2

# max iteration
max_iteration = 1

# epochs
epochs = 1
batch_size = 32
hidden_dims = 384
hidden_states_size=6
bert_model='bert-large-uncased'

# model name
model_name = 'model_l2'

# path to save dnn logs
dnn_log_path = root_dir + "/dnn_log_" + model_name 


# From the files of the train set we can produce the labels and from the segmentation we can get the PAT only segments. For the test set we don't have the labels, but we can get the codes and the segmentations.

# In[ ]:

# first concentrate on the train set
# list of the file names in ad group
ad_files_list = glob.glob(train_audio_files_path + "/ad/*.wav")

# list of the file names in cn group
cn_files_list = glob.glob(train_audio_files_path + "/cn/*.wav")

# codes are taken from the wav file name
train_codes = np.array([f.split("/")[-1].replace(".wav","") for f in ad_files_list + cn_files_list])

# label 1 for the ad group and 0 for the cn group
train_labels = np.array([1] * len(ad_files_list) + [0] * len(cn_files_list))

# print 3 sample codes from the ad and 3 from the cn groups
print("3 sample codes fom ad/cn groups: %s / %s"%(";".join(train_codes[:3]),";".join(train_codes[-3:])))

# print the coressponding labels
print("3 sample labels from ad/cn groups: ", train_labels[:3], train_labels[-3:])

# print the total number of samples
print("the total number of the train samples: %d" %len(train_codes))

# now go to the test set
# list of the file names in test set
test_files_list = glob.glob(test_audio_files_path + "/*.wav")

# test codes are also taken from the wav file name
test_codes = np.array([f.split("/")[-1].replace(".wav","") for f in test_files_list])

# print the total number of test samples
print()
print("the total number of the test samples: %d" %len(test_codes))       


# ## Get ASR outputs

# We have passed the audio files to our ASR and the ASR has produced some lattices. From the laticces we have generated a number of CTM files with confidence using different language model weights and word insertion penalty values. The results are saved in a number of text files.

# In[ ]:

# read through the ctm file words and if the segment is allowed (is in PAR only segments) then get the text and confidence score
def read_ctm_file(ctm_file_path):
    output = {}
    
    for line in open(ctm_file_path):
        strs = line.strip().split()
        if len(strs) > 5:
            code, start, length, word, conf_score = strs[0], float(strs[2]), float(strs[3]), strs[4], int(100* float(strs[5]))
             
            if not code in output:
                output[code] = {"text":[], "conf":[]}
            output[code]["text"].append(word)
            output[code]["conf"].append(conf_score)
    print("ctm:%s codes:%d" %(ctm_file_path, len(output)))
                
    return output 
            
def get_asr_text_confs(asr_output_path):
    results = {} 
    
    # now read all the ctm files and get the text and conf scores
    for ctm_name in glob.glob(asr_output_path + "/firstbest.*.ctm"):
        # read text and confs from the ctm file 
        ctm_output = read_ctm_file(ctm_name)  
        results[ctm_name] = ctm_output
    
    count = 0 
    for ctm_name in results:
        count += 1
        if count > 2:
            break
            
        sample_code = sorted(results[ctm_name])[0]
        print("path:%s ctm-name:%s sample code:%s text:%s conf:%s" % (asr_output_path, ctm_name, sample_code," ".join(results[ctm_name][sample_code]['text']),"-".join([str(s) for s in results[ctm_name][sample_code]['conf']])))
    print()   
    
    return results
        
# extract all the text/confs from the ctm files for the train set    
train_asr_data = get_asr_text_confs(train_asr_output_path)

# extract all the text/confs from the ctm files for the test set 
test_asr_data = get_asr_text_confs(test_asr_output_path)


# ## Prepare data for DNNs

# We will make two data sets out of the ASR outputs: data from the best conf data from all hypothesis.

# In[ ]:

# get rid of the ASR special words 
def trim(text, conf):
    text_out, conf_out = [], []
    for i, word in enumerate(text):
        if not word in ['<NOISE>', '<SPOKEN_NOISE>']:
            text_out.append(word)
            conf_out .append(conf[i])
            
    return text_out, conf_out

def init_data():
    return {'train':{'confs':[],'sentences':[],'codes':[],'input_ids':[],'attention_masks':[],'labels':[]},
            'test':{'confs':[],'sentences':[],'codes':[],'input_ids':[],'attention_masks':[],'labels':[]}}

# make ready two types of data
def prepare_data(train_asr_data, train_codes, train_labels, test_asr_data, test_codes, bert_model="bert-large-uncased", max_text_length=MAX_TEXT_LENGTH):
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    data = init_data()   
    train_source, test_source = [train_asr_data[a] for a in train_asr_data], [test_asr_data[a] for a in test_asr_data] 
         
    for source in test_source:
        for idx, code in enumerate(source): 
            text, conf = trim(source[code]['text'], source[code]['conf'])
            sentence = " ".join(text)
            confs = np.array([-1] * max_text_length)
            
            confs[:len(conf)] = np.array(conf[:max_text_length])
            bert_inp = tokenizer.encode_plus(sentence, add_special_tokens=True,max_length=max_text_length,pad_to_max_length=True,return_attention_mask=True)
            input_ids = bert_inp['input_ids']
            attention_masks = bert_inp['attention_mask'] 
            
            data['test']['sentences'].append(sentence)
            data['test']['confs'].append(confs)
            data['test']['codes'].append(code)
            data['test']['input_ids'].append(np.array(input_ids))
            data['test']['attention_masks'].append(np.array(attention_masks))
    for source in train_source:
        for idx, code in enumerate(source): 
            tr_idx = list(train_codes).index(code)
            text, conf = trim(source[code]['text'], source[code]['conf'])
            sentence = " ".join(text)
            confs = np.array([-1] * max_text_length)
            
            confs[:len(conf)] = np.array(conf[:max_text_length])
            bert_inp = tokenizer.encode_plus(sentence, add_special_tokens=True,max_length=max_text_length,pad_to_max_length=True,return_attention_mask=True)
            input_ids = bert_inp['input_ids']
            attention_masks = bert_inp['attention_mask'] 
            
            data['train']['sentences'].append(sentence)
            data['train']['confs'].append(confs)
            data['train']['codes'].append(code)
            data['train']['input_ids'].append(np.array(input_ids))
            data['train']['attention_masks'].append(np.array(attention_masks))
            data['train']['labels'].append(train_labels[tr_idx])
        
    for d in data:
        data[d]['labels'] = np.array(data[d]['labels'])
        data[d]['confs'] = np.array(data[d]['confs'])
        data[d]['input_ids'] = np.array(data[d]['input_ids'])
        data[d]['attention_masks'] = np.array(data[d]['attention_masks'])
        print() 
        print("len %s: %d" % (d, len(data[d]['confs'])))
        print("sentences:",data[d]['sentences'][:1])
        print("confs:",data[d]['confs'][:1])  
        
    return data 

# get dnn data 
data_dictionary = prepare_data(train_asr_data, train_codes, train_labels, test_asr_data, test_codes)


# ## Now train DNN

# In[ ]:

# using mode get vote
def get_mode(a):
    if isinstance(a, list):
        return list(mode(a)[0])[0], list(mode(a)[1])[0]
    else:
        return list(mode(list(a))[0])[0], list(mode(list(a))[1])[0]
    
def vote(y, codes):
    result = {'codes':{},'visited':{},'modes':[],'out_codes':[]}
    for i, code in enumerate(codes):
        if code not in result['codes']:
            result['codes'][code] = []
        result['codes'][code].append(y[i])

    for code in codes:
        if not code in result['visited']:
            result['visited'][code] = True
            md,_ = get_mode(result['codes'][code])
            result['modes'].append(md)
            result['out_codes'].append(code)
    return result['modes'], result['out_codes']

# maks the structure of the model ready
def build_model(hidden_dims=768,num_classes=2,hidden_states_size=4, max_text_length=MAX_TEXT_LENGTH, bert_model='bert-large-uncased'):
    
    init_model = TFBertForSequenceClassification.from_pretrained(bert_model, output_hidden_states=True)

    input_ids = keras.Input(shape=(max_text_length, ),dtype='int64')
    attention_mask = keras.Input(shape=(max_text_length, ), dtype='int64')
    input_confs = keras.Input(shape=(max_text_length, ),dtype='int64') 
    
    confs = layers.Dense(hidden_dims, activation="relu")(input_confs) 
    transformer = init_model([input_ids, attention_mask])    
    hidden_states = transformer[1]  
     
    hiddes_states_ind = list(range(-hidden_states_size, 0, 1)) 
    selected_hiddes_states = layers.concatenate(tuple([hidden_states[i] for i in hiddes_states_ind])) 
    transformer_output = layers.Flatten()(selected_hiddes_states) 
    transformer_output = layers.Dense(hidden_dims, activation='relu')(transformer_output) 
    
    merged = layers.concatenate([transformer_output, confs])  
    outputs = layers.Dense(num_classes, activation="softmax")(merged)
     
    model = keras.models.Model(inputs = [input_ids, attention_mask,input_confs] , outputs = outputs)

    model.summary()
    
    return model

# trains the model
def train(model, model_name, input_ids, attention_masks, confs, labels, validation_split=validation_split,epochs=10, batch_size=32,dnn_log_path=dnn_log_path): 
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)
    best_model_path = dnn_log_path + '/best_' + model_name +'.h5'
    
    model.compile(loss=loss,optimizer=optimizer,metrics=[metric])  
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=best_model_path,
                                                 save_weights_only=True,monitor='val_accuracy',
                                                 mode='max',save_best_only=True, verbose=1),
                                                 keras.callbacks.TensorBoard(log_dir=dnn_log_path)]
    # load best weights if exists
    model = load_best_model(model, best_model_path)
    
    # fit the model
    model.fit([input_ids,attention_masks, confs],labels, 
                      batch_size=batch_size,epochs=epochs,
                      validation_split=validation_split,
                      callbacks=callbacks,
                      shuffle=True)
    # load best weights
    model = load_best_model(model, best_model_path)

    model.compile(loss=loss,optimizer=optimizer,metrics=[metric])
    return model

def load_best_model(model, model_path):
    if os.path.exists(model_path):
        # load weights
        model.load_weights(model_path)
    return model 

# predicts
def predict(model, input_ids, attention_masks, confs): 
    preds = model.predict([input_ids, attention_masks, confs])
    return [np.argmax(pr) for pr in preds]

# prints the results and saves as a csv file
def print_results(codes, predictions, csv_file=None): 
    heading = "ID,Prediction"
    print(heading)
    if csv_file != None:
        out_csv = open(csv_file, "w")
        out_csv.write("%s\n"%heading) 
    for i in range(len(codes)):
        code, pred = codes[i], predictions[i]
        pred = 'ad' if pred == 1 else 'cn'
        out_str = "%s,%s" %(code, pred)
        print(out_str)
        if csv_file != None:
            out_csv.write("%s\n"%out_str)
    if csv_file != None:
        out_csv.close()

def calcualate_f1(predictions, labels, average='weighted'):  
    accuracy = accuracy_score(np.array(labels), np.array(predictions))
    print("Accuracy:%.4f" % accuracy)
    
    precision, recall, fscore, _ = precision_recall_fscore_support(np.array(labels), np.array(predictions), average=None)
    print(precision, recall, fscore)
    
    precision, recall, fscore, _ = precision_recall_fscore_support(np.array(labels), np.array(predictions), average=average)
    print(precision, recall, fscore) 


# ## Model 1
# 
# The Bert large is used for building the model and to train and test, all the hypothesis and the confidence scores are used.

# In[ ]:

# build the model1
model1 = build_model(hidden_dims=hidden_dims,hidden_states_size=hidden_states_size,bert_model=bert_model)

prediction_list, code_list = [], []
for _ in range(max_iteration):
    # train the DNN 
    model1 = train(model1, model_name, data_dictionary['train']['input_ids'], 
                   data_dictionary['train']['attention_masks'],
                   data_dictionary['train']['confs'],
                   data_dictionary['train']['labels'], 
                   epochs=epochs, 
                   batch_size=batch_size,
                   dnn_log_path=dnn_log_path + "/" + model_name) 

    # predict labels of the test set
    predicted_model1 = predict(model1, 
                               data_dictionary['test']['input_ids'],
                               data_dictionary['test']['attention_masks'],
                               data_dictionary['test']['confs'])
    prediction_list += predicted_model1
    code_list += data_dictionary['test']['codes']

predicted_model1_voted, t_codes = vote(prediction_list, code_list)

# print the results and save them in a csv file
print_results(t_codes, predicted_model1_voted, csv_file=root_dir + "/" + model_name +"_test_results_task1.csv")


# ## Extra check
# 
# Extra check with an oracle test estimation.

# In[ ]:

# extra check REMOVE ME AFTERWARD
best_preds = { 'adrsdt1': 1, 'adrsdt2': 1, 'adrsdt3': 0, 'adrsdt4': 1, 'adrsdt5': 0, 
             'adrsdt6': 0,'adrsdt7': 1, 'adrsdt8': 1, 'adrsdt9': 1, 'adrsdt10': 0, 
             'adrsdt11': 0, 'adrsdt12': 0, 'adrsdt13': 1, 'adrsdt14': 1, 'adrsdt15': 0, 
             'adrsdt16': 1, 'adrsdt17': 0, 'adrsdt18': 1, 'adrsdt19': 1, 'adrsdt20': 1, 
             'adrsdt21': 0, 'adrsdt22': 0, 'adrsdt23': 1, 'adrsdt24': 1, 'adrsdt25': 1, 
             'adrsdt26': 0, 'adrsdt27': 0, 'adrsdt28': 1, 'adrsdt29': 1, 'adrsdt30': 0, 
             'adrsdt31': 0, 'adrsdt32': 1, 'adrsdt33': 1, 'adrsdt34': 1, 'adrsdt35': 1, 
             'adrsdt36': 0, 'adrsdt37': 1, 'adrsdt38': 0, 'adrsdt39': 1, 'adrsdt40': 0, 
             'adrsdt41': 1, 'adrsdt42': 0, 'adrsdt43': 0, 'adrsdt44': 1, 'adrsdt45': 0, 
             'adrsdt46': 1, 'adrsdt47': 0, 'adrsdt48': 0, 'adrsdt49': 0, 'adrsdt50': 0,
             'adrsdt51': 0, 'adrsdt52': 0, 'adrsdt53': 0, 'adrsdt54': 0, 'adrsdt55': 0, 
             'adrsdt56': 0, 'adrsdt57': 1, 'adrsdt58': 0, 'adrsdt59': 1, 'adrsdt60': 1,
             'adrsdt61': 1, 'adrsdt62': 1, 'adrsdt63': 1, 'adrsdt64': 1, 'adrsdt65': 0, 
             'adrsdt66': 0, 'adrsdt67': 0, 'adrsdt68': 0, 'adrsdt69': 0, 'adrsdt70': 1, 'adrsdt71': 1
    }

def f1(csv_file):
    test_codes, predictions = [], []
    for line in open(csv_file):
        strs = line.strip().split(",")
        test_codes.append(strs[0])
        pred = 1 if strs[1] == 'ad' else 0
        predictions.append(pred)
    
    test, prds, test2, prds2, unk = [], [], [], [], ['adrsdt15','adrsdt26','adrsdt40','adrsdt49','adrsdt58', 'adrsdt67']
    for code in sorted(best_preds):
        if code in test_codes:
            idx = test_codes.index(code)
            p = predictions[idx] 
            test.append(best_preds[code])
            prds.append(p) 
            if not code in unk:
                test2.append(best_preds[code])
                prds2.append(p)
                
    print("All labels:")       
    calcualate_f1(prds, test)
    print()
    print("Known labels:")
    calcualate_f1(prds, test)


# In[ ]:

# model 1
f1(csv_file=root_dir + "/" + model_name +"_test_results_task1.csv") 


# In[ ]:



