import numpy as np
import evaluate

def compute_metrics(p):
    # Extract the highest probability predictions
    predictions = np.argmax(p.predictions, axis=1)
    # Load the metric for F1 and accuracy evaluation
    f1_metric = evaluate.load('f1')
    acc_metric = evaluate.load('accuracy')
    
    # Compute micro F1, macro F1, and accuracy
    micro_f1 = f1_metric.compute(predictions=predictions, references=p.label_ids, average='micro')
    macro_f1 = f1_metric.compute(predictions=predictions, references=p.label_ids, average='macro')
    accuracy = acc_metric.compute(predictions=predictions, references=p.label_ids)
    
    # Return the computed metrics
    return {
        "micro_f1": micro_f1['f1'],
        "macro_f1": macro_f1['f1'],
        "accuracy": accuracy['accuracy']
    }