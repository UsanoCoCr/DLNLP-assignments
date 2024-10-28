import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy
    }