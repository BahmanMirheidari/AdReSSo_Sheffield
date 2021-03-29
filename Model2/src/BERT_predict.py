import torch
from src.conf import *
import pdb
import numpy as np
def model_predict_test(model, test_dataloader, device, best_model_path):
    # Put model in evaluation mode
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    # Tracking variables
    predictions , true_labels = [], []
    # Predict
    hidden_state_list = []
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
                            
            
        logits = outputs[0]
        if output_hidden_states == True:
            hidden_state_matrix = []
            for idx in range(1,5): 
                hidden_state_matrix.append(outputs[1][-idx].cpu().numpy())
            hidden_state = np.concatenate(np.asarray(hidden_state_matrix), axis=-1)
            hidden_state_list.append(hidden_state)
            
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # Store predictions and true labels
        predictions.append(logits)
    print('DONE.')
    return  predictions, hidden_state_list

def model_predict_dev(model, test_dataloader, device, best_model_path):
    # Put model in evaluation mode
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    # Tracking variables
    predictions , true_labels = [], []
    # Predict
    hidden_state_list = []
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)


        logits = outputs[0]
        if output_hidden_states == True:
            hidden_state_matrix = []
            for idx in range(1,5):
                hidden_state_matrix.append(outputs[1][-idx].cpu().numpy())
            hidden_state = np.concatenate(np.asarray(hidden_state_matrix), axis=-1)
            hidden_state_list.append(hidden_state)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()


        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('DONE.')
    return  true_labels, predictions, hidden_state_list