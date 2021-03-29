import random
import numpy as np
import torch
import time
from src.timer import format_time
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, BertConfig
from src.criteria_cal import flat_accuracy, calculat_f1
from src.data_preprocessing import reset_random_seeds
import pdb
from sklearn.metrics import f1_score
from src.conf import *

def fine_tune(model, epochs, train_dataloader, dev_dataloader, device, model_saved_path):
    reset_random_seeds(SEED)
    loss_values = []

    
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    # For each epoch...
    dev_best_f1 = 0.0
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        t0 = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        t0 = time.time()

        model.eval()
        # Tracking variables
        eval_loss, eval_accuracy, eval_f_score = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Evaluate data for one epoch
        # ========================================
        #               Evaluation
        # ========================================
        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(label_ids, logits)
            eval_accuracy += tmp_eval_accuracy
            pred_label = np.argmax(logits, axis=1) 
            tmp_eval_f_score = f1_score(label_ids, pred_label, average='weighted')
            eval_f_score += tmp_eval_f_score
            nb_eval_steps += 1
        if dev_best_f1 < eval_f_score/nb_eval_steps:
            torch.save(model.state_dict(), model_saved_path)
            dev_best_f1 = eval_f_score/nb_eval_steps
            print("best model updated, F score: {0:.2f}".format(eval_f_score/nb_eval_steps))
    print("")
    print("Training complete!")


