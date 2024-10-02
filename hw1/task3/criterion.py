import torch
import torch.nn.functional as F
from torch import nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import math

class TranslationMetrics:
    def __init__(self):
        # Smoothing function for BLEU score calculation
        self.smoothing_function = SmoothingFunction().method1

    def calculate_bleu(self, candidate, references):
        """
        Calculate the BLEU score for a single candidate sentence against references.
        
        :param candidate: a list of tokens (strings) representing the candidate translation.
        :param references: a list of lists of tokens (strings) representing reference translations.
        :return: BLEU score
        """
        return sentence_bleu(references, candidate, smoothing_function=self.smoothing_function)

    def calculate_perplexity(self, model, data_loader, loss_function):
        """
        Calculate the perplexity of a language model given a data loader and loss function.
        
        :param model: the language model being evaluated
        :param data_loader: DataLoader providing the dataset for evaluation
        :param loss_function: the loss function used to evaluate the model (typically cross-entropy)
        :return: perplexity score
        """
        model.eval()
        total_loss = 0
        total_words = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_words += inputs.size(0)

        avg_loss = total_loss / total_words
        perplexity = math.exp(avg_loss)
        return perplexity
