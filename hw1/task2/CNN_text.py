import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import jieba



def get_vocab(filename):
    vocab = []
    file = open(filename, "r", encoding="utf-8")
    lines = file.readlines()
    text_input = []
    idx_input = []
    for line in lines:
        raw_list = jieba.lcut(line[0:len(line)-3], use_paddle=True)
        text_input.append(raw_list)
        idx_input.append(line[len(line)-2])
        for word in raw_list:
            if word not in vocab:
                vocab.append(word)
    file.close()
    return text_input, idx_input, vocab

class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.overtime_pooling = nn.ModuleList([nn.MaxPool1d(kernel_size=embedding_dim - fs + 1) for fs in filter_sizes])
        self.mlp = nn.Sequential(
            nn.Linear(n_filters * len(filter_sizes), 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        # text = [bsz, len]
        embedded = self.embedding(text) # [bsz, len, emb_dim]
        embedded = embedded.unsqueeze(1) # [bsz, 1, len, emb_dim]
        # print(embedded.shape)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # [bsz, n_filters, len - fs + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1) # [bsz, n_filters]
        output = self.mlp(cat)
        return self.softmax(output)

def accuracy(preds, y):
    preds = preds.argmax(dim=1, keepdim=True)
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float() / y.shape[0]
    return acc 

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        input, label = batch
        input = input.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        predictions = model(input).squeeze(1)
        loss = criterion(predictions, label)
        acc = accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    # add early stopping
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            input, label = batch
            input = input.to(device)
            label = label.to(device)
            predictions = model(input).squeeze(1)
            loss = criterion(predictions, label)
            acc = accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
        
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_text, train_idx,vocab = get_vocab("train.txt")
    val_text, val_idx, _ = get_vocab("dev.txt")
    test_text, test_idx, _ = get_vocab("test.txt")
    # train_text += val_text
    # train_idx += val_idx

    vocab.append("<unk>")

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    train_text = [[word2idx[word] if word in word2idx else word2idx["<unk>"] for word in sentence] for sentence in train_text]
    val_text = [[word2idx[word] if word in word2idx else word2idx["<unk>"] for word in sentence] for sentence in val_text]
    test_text = [[word2idx[word] if word in word2idx else word2idx["<unk>"] for word in sentence] for sentence in test_text]

    train_text_tensors = [torch.tensor(sentence) for sentence in train_text]
    train_text_tensor = pad_sequence(train_text_tensors, batch_first=True, padding_value=0)
    val_text_tensors = [torch.tensor(sentence) for sentence in val_text]
    val_text_tensor = pad_sequence(val_text_tensors, batch_first=True, padding_value=0)
    test_text_tensors = [torch.tensor(sentence) for sentence in test_text]
    test_text_tensor = pad_sequence(test_text_tensors, batch_first=True, padding_value=0)

    train_idx = [int(idx) for idx in train_idx]
    val_idx = [int(idx) for idx in val_idx]
    test_idx = [int(idx) for idx in test_idx]

    train_text_tensor, train_idx = torch.tensor(train_text_tensor, dtype=torch.long), torch.tensor(train_idx, dtype=torch.int64)
    val_text_tensor, val_idx = torch.tensor(val_text_tensor, dtype=torch.long), torch.tensor(val_idx, dtype=torch.int64)
    test_text_tensor, test_idx = torch.tensor(test_text_tensor, dtype=torch.long), torch.tensor(test_idx, dtype=torch.int64)

    BATCH_SIZE = 16
    train_data = torch.utils.data.TensorDataset(train_text_tensor, train_idx)
    val_data = torch.utils.data.TensorDataset(val_text_tensor, val_idx)
    test_data = torch.utils.data.TensorDataset(test_text_tensor, test_idx)
    train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_iterator = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)
    test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 256
    N_FILTERS = 20
    FILTER_SIZES = [1,2,3,4,5,6]
    OUTPUT_DIM = 4
    DROPOUT = 0.1
    model = CNN_Text(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    N_EPOCHS = 10
    history_val_acc = []
    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch+1:02}')
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

        # add early stopping
        val_loss, val_acc = evaluate(model, val_iterator, criterion, device)
        print(f'\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')
        if len(history_val_acc) > 0 and val_acc < history_val_acc[-1]:
            print("Early stopping at epoch", epoch)
            break
        history_val_acc.append(val_acc)

        torch.save(model.state_dict(), "CNN_text.pth")

    model.load_state_dict(torch.load("CNN_text.pth"))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
