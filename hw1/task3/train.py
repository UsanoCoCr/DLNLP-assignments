import torch
from torch.utils.data import DataLoader, Dataset
from model.LSTM import seq2seq, encoder_LSTM, decoder_LSTM
from torch.nn.utils.rnn import pad_sequence
import MeCab
import nltk
from nltk.tokenize import word_tokenize
import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

mecab = MeCab.Tagger('-Owakati')
nltk.download('punkt_tab')

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sentence, trg_sentence = self.data[idx]
        src_indices = torch.tensor([self.src_vocab[word] for word in src_sentence], dtype=torch.long)
        trg_indices = torch.tensor([self.trg_vocab[word] for word in trg_sentence], dtype=torch.long)
        return src_indices, trg_indices

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch_padded = pad_sequence(src_batch, padding_value=0, batch_first=False)
    trg_batch_padded = pad_sequence(trg_batch, padding_value=0, batch_first=False)
    return src_batch_padded, trg_batch_padded

def train_LSTM(dataloader, jpn_vocab_size, eng_vocab_size, embedding_dim, hidden_dim, num_layers_encoder, num_layers_decoder, bidirectional, num_epochs, learning_rate, device):
    encoder = encoder_LSTM(jpn_vocab_size, embedding_dim, hidden_dim, num_layers_encoder, bidirectional).to(device)
    decoder = decoder_LSTM(eng_vocab_size, embedding_dim, hidden_dim, hidden_dim, num_layers_decoder).to(device)
    model = seq2seq(encoder, decoder, device).to(device)
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = 0
    for epoch in range(num_epochs):
        for i, (context, target) in enumerate(dataloader):
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(context, target)
            loss = calculate_loss(output, target, loss_function, device)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
        torch.save(model.state_dict(), "lstm_model.ckpt".format(epoch+1))
    
    return model

def calculate_loss(prediction, target, loss_function, device):
    # predictions = [seq_len, batch_size, vocab_size]
    # targets = [seq_len, batch_size]
    prediction = prediction.permute(1, 0, 2) # [batch_size, seq_len, vocab_size]
    target = target.permute(1, 0) # [batch_size, seq_len]
    prediction = prediction.reshape(-1, prediction.size(-1))  # [batch_size * seq_len, vocab_size]
    target = target.reshape(-1)  # [batch_size * seq_len]
    loss = loss_function(prediction, target)
    return loss

def evaluate_LSTM(model, dataloader, device, eng_vocab_size, eng_idx_to_word):
    model.eval()

    with torch.no_grad():
        bleu_score = 0
        perplexity = 0
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            output = model(context, target, teacher_forcing_ratio=0)
            eval = criterion(output, target, eng_vocab_size, eng_idx_to_word)
            bleu_score += eval[0]
            perplexity += eval[1]
        
        bleu_score /= len(dataloader)
        perplexity /= len(dataloader)
        print("BLEU Score: {:.4f}, Perplexity: {:.4f}".format(bleu_score, perplexity))

import torch.nn.functional as F
def criterion(output, target, vocab_size, index_to_word):
    # output = [seq_len, bsz, vocab_size]
    # target = [seq_len, bsz]
    output_p = output.view(-1, vocab_size)
    target_p = target.view(-1)
    loss = F.cross_entropy(output_p, target_p)
    perplexity = torch.exp(loss)

    _, predicted_indices = torch.max(output, dim=2)
    predicted_indices = predicted_indices.permute(1, 0) # [bsz, seq_len]
    target = target.permute(1, 0) # [bsz, seq_len]
    bleu_score = 0
    for i in range(predicted_indices.size(0)):
        predicted_sentence = [index_to_word[idx.item()] for idx in predicted_indices[i]]
        target_sentence = [index_to_word[idx.item()] for idx in target[i]]
        print("Predicted: ", predicted_sentence)
        print("Target: ", target_sentence)
        bleu_score += sentence_bleu([target_sentence], predicted_sentence, smoothing_function=SmoothingFunction().method4)
    bleu_score /= predicted_indices.size(0)
    return bleu_score, perplexity


def load_data(src_file, trg_file):
    with open(src_file, 'r', encoding='utf-8') as file:
        src = [mecab.parse(line).split() for line in file]
    with open(trg_file, 'r', encoding='utf-8') as file:
        trg = [word_tokenize(line) for line in file]
    return list(zip(src, trg))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    train_data = load_data("./dataset/train_jpn.txt", "./dataset/train_eng.txt")
    val_data = load_data("./dataset/val_jpn.txt", "./dataset/val_eng.txt")
    test_data = load_data("./dataset/test_jpn.txt", "./dataset/test_eng.txt")

    with open('cbow_jpn_vocab.pkl', 'rb') as jpn_vocab_file:
        jpn_vocab = pickle.load(jpn_vocab_file)

    with open('cbow_eng_vocab.pkl', 'rb') as eng_vocab_file:
        eng_vocab = pickle.load(eng_vocab_file)

    jpn_vocab_size = len(jpn_vocab)
    eng_vocab_size = len(eng_vocab)
    print("Japanese vocab size: ", jpn_vocab_size)
    print("English vocab size: ", eng_vocab_size)

    jpn_word_to_idx = {word: i for i, word in enumerate(jpn_vocab)}
    jpn_idx_to_word = {i: word for i, word in enumerate(jpn_vocab)}
    eng_word_to_idx = {word: i for i, word in enumerate(eng_vocab)}
    eng_idx_to_word = {i: word for i, word in enumerate(eng_vocab)}

    train_dataset = TranslationDataset(train_data, jpn_word_to_idx, eng_word_to_idx)
    test_dataset = TranslationDataset(test_data, jpn_word_to_idx, eng_word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = train_LSTM(dataloader=train_loader, 
                       jpn_vocab_size=jpn_vocab_size,
                       eng_vocab_size=eng_vocab_size,
                       embedding_dim=256, 
                       hidden_dim=512, 
                       num_layers_encoder=1,
                       num_layers_decoder=1, 
                       bidirectional=False,
                       num_epochs=10, 
                       learning_rate=0.001, 
                       device=device)
    
    """ model = seq2seq(encoder_LSTM(jpn_vocab_size, 256, 512, 1, False), 
                    decoder_LSTM(eng_vocab_size, 256, 512, 512, 1), 
                    device)
    model.to(device)
    model.load_state_dict(torch.load('lstm_model.ckpt')) """
    evaluate_LSTM(model, test_loader, device, eng_vocab_size, eng_idx_to_word)
