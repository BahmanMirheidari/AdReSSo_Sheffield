import torch
from src.conf import *
import pdb
import numpy as np
def test_model_predict(model, test_dataloader, device, best_model_path):
    # Put model in evaluation mode
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    predictions  = []
    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_wav2vec, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, wav2vec_feats = b_wav2vec,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # Store predictions and true labels
        predictions.append(logits)
    print('DONE.')
    return  predictions

def dev_model_predict(model, test_dataloader, device, best_model_path):
    # Put model in evaluation mode
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    # Tracking variables
    predictions , true_labels = [], []
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_wav2vec, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, wav2vec_feats = b_wav2vec,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('DONE.')
    return  true_labels, predictions