import numpy as np
from sklearn.metrics import matthews_corrcoef
import pdb
from keras import backend as K
# Function to calculate the accuracy of our predictions vs labels

def pr(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def rc(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):

    # If there are no true positives, fix the F score at 0 like sklearn.
    #if K.sum(K.clip(y_true, 0, 1)) == 0:
    #    return 0
    p = pr(y_true, y_pred)
    r = rc(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def f1(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    
    return fbeta_score(y_true, y_pred, beta=1)
    
def type_check(params):
    if (type(params).__name__=='list') == True:
        params = np.concatenate(params, axis = 0)
    return params

def flat_accuracy(true_labels, predictions):
    true_labels = type_check(true_labels)
    predictions = type_check(predictions)

    pred_labels = np.argmax(predictions, axis=1).flatten()
    true_labels = true_labels.flatten()
    return np.sum(pred_labels == true_labels) / len(true_labels)


def matthews_cal(true_labels, predictions):
    matthews_set = []
    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')
    # For each input batch...
    for i in range(len(true_labels)):

        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    print(matthews_set)
    return matthews_set

def comb_pred(true_labels, predictions):
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    print('MCC: %.3f' % mcc)
    return mcc


def calculat_f1(true_labels, predictions):

    true_labels = type_check(true_labels)
    predictions = type_check(predictions)
    pred_labels = np.argmax(predictions, axis=1).flatten()
    true_labels = true_labels.flatten()
    
    precision = pr(true_labels, pred_labels)
    recall = rc(true_labels, pred_labels)
    f_score = f1(true_labels, pred_labels)
    #print("  precision: {0:.2f}".format(precision))
    #print("  recall: {:}".format(recall))
    #print("  F score: {0:.2f}".format(f_score))

    return precision, recall, f_score