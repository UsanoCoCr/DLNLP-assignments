import torch
import random

class encoder_LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional):
        super(encoder_LSTM, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.load_embedding()
        print("Encoder LSTM model loaded")
        
    def forward(self, context):
        # context = [len, bsz]
        x = self.embeddings(context) # x = [len, bsz, dim]
        x, (hidden, cell) = self.lstm(x) # hidden/cell = [num_layers, bsz, dim]
        hidden, cell = hidden[-1], cell[-1] # hidden/cell = [bsz, dim]
        return x, hidden, cell # x = [len, bsz, dim]
    
    def load_embedding(self):
        checkpoint = torch.load('cbow_jpn_model.ckpt')
        pretrained_embeddings = checkpoint['embeddings.weight']
        self.embeddings.weight.data.copy_(pretrained_embeddings)
        self.embeddings.weight.requires_grad = True
    
class attention(torch.nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(attention, self).__init__()
        self.linear = torch.nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = torch.nn.Linear(decoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden = [1, bsz, dim_dec]
        # encoder_outputs = [len, bsz, dim_enc]
        len = encoder_outputs.shape[0]
        decoder_hidden = decoder_hidden.repeat(len, 1, 1) # [len, bsz, dim_dec]

        energy = torch.tanh(self.linear(torch.cat((decoder_hidden, encoder_outputs), dim=2))) # [len, bsz, dim_dec]
        attention_score = self.v(energy).squeeze(2) # [len, bsz]
        attention_score = torch.nn.functional.softmax(attention_score, dim=0) # [len, bsz]
        return attention_score # [len, bsz]
    
class decoder_LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, hidden_dim, num_layers):
        super(decoder_LSTM, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim+encoder_hidden_dim, hidden_dim, num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)
        self.attention = attention(encoder_hidden_dim, hidden_dim)
        self.load_embedding()
        print("Decoder LSTM model loaded")

    def forward(self, context, encoder_outputs, hidden, cell):
        # context = [bsz]
        # encoder_outputs = [len, bsz, dim_enc]
        # hidden/cell = [bsz, dim]
        x = self.embeddings(context) # x = [bsz, dim]
        attention_score = self.attention(hidden, encoder_outputs) # [len, bsz]
        attention_score = attention_score.permute(1, 0).unsqueeze(1) # [bsz, 1, len]
        context = attention_score.bmm(encoder_outputs.permute(1, 0, 2)) # [bsz, 1, dim_enc]
        context = context.squeeze(1) # [bsz, dim_enc]
        x = torch.cat((x, context), dim=1) # x = [bsz, dim+dim_enc]
        x = x.unsqueeze(0) # x = [1, bsz, dim+dim_enc]
        x, (hidden, cell) = self.lstm(x, (hidden, cell))
        x = self.linear(x)
        return x, hidden, cell # x = [1, bsz, vocab_size]
    
    def load_embedding(self):
        checkpoint = torch.load('cbow_eng_model.ckpt')
        pretrained_embeddings = checkpoint['embeddings.weight']
        self.embeddings.weight.data.copy_(pretrained_embeddings)
        self.embeddings.weight.requires_grad = True
    
class seq2seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, context, target, teacher_forcing_ratio=0.5):
        # context = [len_cnt, bsz]
        # target = [len_trg, bsz]
        encoder_output, hidden, cell = self.encoder(context)
        trg_len, bsz = target.shape
        trg_vocab_size = self.decoder.linear.out_features

        outputs = torch.zeros(trg_len, bsz, trg_vocab_size).to(self.device)
        input = target[0, :] # input = [bsz]
        hidden = hidden.unsqueeze(0) # hidden = [1, bsz, dim]
        cell = cell.unsqueeze(0) # cell = [1, bsz, dim]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, encoder_output, hidden, cell) # output = [1, bsz, vocab_size]
            outputs[t] = output.squeeze(0)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2) # top1 = [1, bsz]
            input = target[t] if teacher_force else top1.squeeze(0) # input = [bsz]
        
        return outputs