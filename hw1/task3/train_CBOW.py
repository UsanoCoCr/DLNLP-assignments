import torch
from model.CBOW import CBOW
from torch.utils.data import DataLoader
import pickle

def train_CBOW(train_data, vocab_size, embedding_dim, num_epochs, batch_size, learning_rate, device):
    model = CBOW(vocab_size, embedding_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for i, (context, target) in enumerate(dataloader):
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
        torch.save(model.state_dict(), "cbow_" + language + "_model.ckpt".format(epoch+1))
    
    return model

def evaluate_CBOW(model, test_data, batch_size, device):
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
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

vocab = []
def get_vocab(data):
    for context, target in data:
        for word in context:
            if word not in vocab:
                vocab.append(word)
        if target not in vocab:
            vocab.append(target)
    return vocab

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    language = 'eng'

    train_cbow_file = open("./dataset/cbow_" + language + "_dataset/train_cbow.pkl", "rb")
    train_data = pickle.load(train_cbow_file)
    train_cbow_file.close()

    val_cbow_file = open("./dataset/cbow_" + language + "_dataset/val_cbow.pkl", "rb")
    val_data = pickle.load(val_cbow_file)
    val_cbow_file.close()

    test_cbow_file = open("./dataset/cbow_" + language + "_dataset/test_cbow.pkl", "rb")
    test_data = pickle.load(test_cbow_file)
    test_cbow_file.close()

    get_vocab(train_data)
    get_vocab(val_data)
    get_vocab(test_data)
    vocab_size = len(vocab)
    print("Vocab size: ", vocab_size)

    # save vocab
    vocab_file = open("cbow_" + language + "_vocab.pkl", "wb")
    pickle.dump(vocab, vocab_file)
    vocab_file.close()
    print("Saving vocab to cbow_" + language + "_vocab.pkl")

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    train_data = [(torch.tensor([word_to_idx[word] for word in context], dtype=torch.long), torch.tensor(word_to_idx[target], dtype=torch.long)) for context, target in train_data]
    val_data = [(torch.tensor([word_to_idx[word] for word in context], dtype=torch.long), torch.tensor(word_to_idx[target], dtype=torch.long)) for context, target in val_data]
    test_data = [(torch.tensor([word_to_idx[word] for word in context], dtype=torch.long), torch.tensor(word_to_idx[target], dtype=torch.long)) for context, target in test_data]

    embedding_dim = 256
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # model = train_CBOW(train_data, vocab_size, embedding_dim, num_epochs, batch_size, learning_rate, device)
    model = CBOW(vocab_size, embedding_dim).to(device)
    model.load_state_dict(torch.load("cbow_" + language + "_model.ckpt"))
    evaluate_CBOW(model, test_data, batch_size, device)