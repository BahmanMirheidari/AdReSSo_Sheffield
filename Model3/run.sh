#! /bin/bash
#$ -l rmem=6G,h_rt=16:00:00,gpu=1
#$ -P tapas
#$ -q tapas.q
#$ -m eba
#$ -M yilin.pan@sheffield.ac.uk
#$ -o outputs-ADReSSo
#$ -e error-ADReSSo
#$ -N BERT

source ~/.bashrc

module load libs/cudnn/7.6.5.32/binary-cuda-10.1.243
source activate tensorflow-gpu


for layer_idx in {0..24}; do
for pre_trained_model in bert-base-uncased;do
  for idx in {0..9};do
    echo $idx
    python BERT_fusion.py --Fold=$idx --pre_trained_model=$pre_trained_model --train_task=ADReSSo_test/ADReSSo --layer_idx=$layer_idx
  done
done
done
