import torch

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, context):
        x = self.embeddings(context)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.linear(x)
        return x