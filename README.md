# AdReSSo_Sheffield

<p align="center"><img width="50%" src="image.gif" /></p>  
AdRESSO21 challenge Interspeech 2021 - Team Sheffield

Authors 
----------
Yilin Pan, Bahman Mirheidari, and Heidi Christensen 

# Requirements
----------   
## Kaldi
Kaldi is a well-known automatic speech recognition toolkit (https://kaldi-asr.org/).
## wav2vec2.0
wav2vec2.0 is a self-supervised end-to-end ASR system (https://github.com/huggingface/transformers)

### How to install Kaldi on Linux/MacOS 
1. Get Kaldi from the GitHub: ``git clone https://github.com/kaldi-asr/kaldi.git kaldi-trunk --origin golden``.
2. Check the dependencies and install the required tools: ``kaldi-trunk/tools/extras/check_dependencies.sh``.
3. Example of required tools on a MacOs: ``brew cmake install automake gfortran subversion``, you may need other tools on a Linux machine.
4. Install irstlm: ``kaldi-trunk/toolsextras/install_irstlm.sh`` and add the line  ``[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh`` to path.sh file of the example you may want to run.
5. Make codes: ``cd kaldi-trunk/tools; make``. 
6. Make source codes: ``cd kaldi-trunk/src; ./configure; make``. 
7. Check it is running: go to ~/.bash_profile (.bash_rc on linux) and add the varible KALDI_ROOT and set it to the kaldi's main director e.g. ``export KALDI_ROOT=/Users/bahmanmirheidari/Documents/AdRESSO/kaldi-trunk``, then go to ``cd kaldi-trunk/egs/wsj/s5;`` and change the content as ``
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=KALDI_ROOT/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C``. Add this line to the end of bash_profile ``. $KALDI_ROOT/egs/wsj/s5/path.sh``. Now run ``copy-feats --help``.
    
  
  
### ASR training
#### Kaldi script based pipeline ASR system
We used Transfer learning from Librispeech on our conversational datasets as well as AMI dataset.
The ASR_results containe the results from the ASR. 30 hypothesis calculated using different language model weights and word insertion penalty.


#### wav2vec2.0 based end-to-end ASR system 
Wav2vec2.0 as an end-to-end ASR paradigm was used for both the audio transcription and the acoustic feature extraction. 
For extracting the acoustic features from the wav2vec2.0 structure, the transformer layers' outputs were extracted as the acoustic representations of the input waveform segments. 
The pre-trained model (https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) is adapted on the IVA dataset.
 
### Models
We have trained five models, codes and outputs are inside the folders Model1, Model2, Model3, Model4, and Model5


Model1: The acoustic only system. The outputs of the Wav2vec2.0 hidden layers (transformer layers) are used for representing the acoustic inforamtion in the wave recordings. For classification, multiple classifiers: KNN, SVM, DT and TB are used. The wav2vec2.0 is also fine-tuned with our self-collected data. 

Model2: The text only model is the BERT-based sequential classifiers with fine-tuned wav2vec2.0 transcripts as the input. 

Model3: The fusion between wav2vec2.0 extracted acoustic features and output transcripts by BERT-base model.

Model4: The BERT-based model that using both the hypotheses with the confidence scores output by the kaldi pipeline ASR system (the maximum word length of the input sequence equals to 105).

Model5: The BERT-based model that using both the hypotheses with the confidence scores output by the kaldi pipeline ASR system (the maximum word length of the input sequence equals to 100).

### Citing our work

Please use the following, to cite our research:
```
@inproceedings{pan2021,
  title={Using the Outputs of Different Automatic Speech Recognition Paradigms for Acoustic-and BERT-Based Alzheimer’s Dementia Detection Through Spontaneous Speech},
  author={Pan, Yilin and Mirheidari, Bahman and Harris, Jennifer M and Thompson, Jennifer C and Jones, Matthew and Snowden, Julie S and Blackburn, Daniel and Christensen, Heidi},
  year={2021}
}
```



 
