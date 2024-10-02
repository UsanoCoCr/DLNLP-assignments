import torch
from torch.utils.data import DataLoader, Dataset
from model.LSTM import seq2seq, encoder_LSTM, decoder_LSTM
from torch.nn.utils.rnn import pad_sequence
import MeCab
import nltk
from nltk.tokenize import word_tokenize
import pickle

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

def evaluate_LSTM(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            output = model(context)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print("Accuracy: {}%".format(100 * correct / total))

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
    evaluate_LSTM(model, test_loader, device=device)
