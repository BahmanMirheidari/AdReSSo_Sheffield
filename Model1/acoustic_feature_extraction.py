from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model, Wav2Vec2ForCTC
import soundfile as sf
import pdb
import torch
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
import librosa
layer_num = 25
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")
train_wav_folder = '/data/ac1yp/data/cookie_theft/ADReSSo/ADReSSo_train/audio'
test_wav_folder = '/data/ac1yp/data/cookie_theft/ADReSSo/ADReSSo_test/audio'
transcript_dict_folder = '/fastdata/ac1yp/ADReSSo/final/wav2vec_transcripts'
feats_dict_folder = '/mnt/fastdata/ac1yp/ADReSSo/final/wav2vec_feats'
if os.path.exists(feats_dict_folder) == False:
    os.makedirs(feats_dict_folder)
if os.path.exists(transcript_dict_folder) == False:
    os.makedirs(transcript_dict_folder)


def hidden_states_format(hidden_states):
    feat_list = []
    for idx in range(layer_num):
        hidden_state = hidden_states[idx].detach().numpy()
        hidden_state = np.mean(np.squeeze(hidden_state), axis = 0)
        feat_list.append(hidden_state)
    return np.asarray(feat_list)

def wav_to_feat(wav_seg):
    input_values = tokenizer(wav_seg, return_tensors="pt").input_values  # Batch size 1
    outputs = model(input_values, output_hidden_states=True)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim = -1)
    transcription = tokenizer.batch_decode(prediction)[0]
    hidden_states = outputs.hidden_states
    hidden_states = hidden_states_format(hidden_states)
    return hidden_states, transcription


def feats_extract(full_audio, fs):
    # considering the waveform is too long
    wav_seg_num = int(len(full_audio)/(fs*5))
    hidden_states_list = []
    transcript_list = []
    for idx in range(wav_seg_num):
        wav_seg = full_audio[idx*(fs*5):(idx+1)*(fs*5)]
        hidden_states, transcript = wav_to_feat(wav_seg)
        hidden_states_list.append(hidden_states)
        transcript_list.extend(transcript)
    hidden_states_list = np.asarray(hidden_states_list)
    hidden_states_list= hidden_states_list.transpose(1,0,2)
    transcript_list = ''.join(transcript_list)
    return hidden_states_list, transcript_list


def feats_save(feat_list, feats_dict_folder, wav_name):
    for layer_idx in range(layer_num):
        feat_path = os.path.join(feats_dict_folder, str(layer_idx), wav_name.split('.')[0]+'.npy')
        feat_array = feat_list[layer_idx]
        if os.path.exists(os.path.join(feats_dict_folder, str(layer_idx))) == False:
            os.mkdir(os.path.join(feats_dict_folder, str(layer_idx)))
        np.save(feat_path, feat_array)


def avg_feat_gene(wav_path, feats_dict_folder, wav_name):
    audio_input, fs = librosa.load(wav_path, sr = 16000)
    if len(audio_input.shape) > 1:
        audio_input = audio_input[:,0] + audio_input[:,1]
    feat_list, transcript = feats_extract(audio_input, fs)
    feats_save(feat_list, feats_dict_folder, wav_name)
    output_path = os.path.join(transcript_dict_folder, wav_name.split('.')[0])
    output_f = open(output_path, 'w')
    output_f.write(transcript.lower())
    output_f.close()



def feat_prepare(wav_folder, feats_dict_folder, layer_num):
    for wav_name in os.listdir(wav_folder):
        print(wav_name)
        if os.path.exists(os.path.join(feats_dict_folder, str(layer_num-1), wav_name.split('.')[0]+'.npy')) == True:
            continue
        wav_path = os.path.join(wav_folder, wav_name)
        avg_feat_gene(wav_path, feats_dict_folder, wav_name)

feat_prepare(train_wav_folder, feats_dict_folder, layer_num)
feat_prepare(test_wav_folder, feats_dict_folder, layer_num)

