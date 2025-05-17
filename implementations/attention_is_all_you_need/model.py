import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
random.seed(101)
torch.manual_seed(101)

with open('config.json', 'r') as f:
    config = json.load(f)

seq_len = config["seq_len"]
n_data = config["n_data"]
n_embd = config["n_embd"]
vocab_size = config["vocab_size"]
batch_size = config["batch_size"]
n_heads = config["n_heads"]
dropout = config["dropout"]
expansion_factor = config["expansion_factor"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
    
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super(TokenEmbeddings, self).__init__()
        self.embedder = nn.Embedding(vocab_size, n_embd).to(device) # maps vocab_size to n_embd
    def forward(self, x):
        return self.embedder(x.to(device))  
    
def get_positional_encodings(seq_len, n_embd): 
    x = torch.zeros(seq_len, n_embd) # seq_len * n_embd
    pos = torch.arange(0, seq_len).unsqueeze(1) # seq_len * 1
    i = torch.arange(0, n_embd, 2) # (n_embd/2)
    denominator = torch.pow(10000, -i / n_embd) # (n_embd/2)
    x[:, 0::2] = torch.sin(pos * denominator)
    x[:, 1::2] = torch.cos(pos * denominator) 
    return x.to(device)

class Attention(nn.Module):
    def __init__(self, n_embd):
        super(Attention, self).__init__()
        self.n_embd = n_embd
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
    def forward(self, q_encodings, k_encodings, v_encodings, mask=False):
        Q = self.q(q_encodings)
        K = self.k(k_encodings)
        V = self.v(v_encodings)
        sims = torch.matmul(Q, K.transpose(dim0=-2, dim1=-1))
        scaled_sims = sims / torch.sqrt(torch.tensor(self.n_embd, dtype=torch.float32, device=device))
        if mask:
            _, y, z = q_encodings.shape 
            mask_matrix = torch.tril(torch.ones(y, z, device=device))
            mask_matrix = torch.where(mask_matrix == 0, float('-inf'), 0.0)
            scaled_sims = scaled_sims + mask_matrix
        attention_percentages = F.softmax(scaled_sims, dim=-1)
        attention_scores = torch.matmul(attention_percentages, V)
        return attention_scores
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_heads == 0, "n_embd i.e, d_model should be divisible by no of Attention heads."
        self.n_embd=n_embd
        self.n_heads=n_heads
        self.head_dim=n_embd//n_heads
        self.heads=nn.ModuleList(
            [Attention(self.head_dim).to(device) for _ in range(n_heads)]
        )
        self.proj=nn.Linear(n_embd, n_embd).to(device)
    def forward(self, q_encodings, k_encodings, v_encodings, mask=False):
        cat=torch.cat([
            head(
                q_encodings[:,:,i * self.head_dim : (i + 1) * self.head_dim], 
                k_encodings[:,:,i * self.head_dim : (i + 1) * self.head_dim], 
                v_encodings[:,:,i * self.head_dim : (i + 1) * self.head_dim],
                mask
            ) for i, head in enumerate(self.heads)], dim=-1)
        mha=self.proj(cat)
        return mha
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.mha=MultiHeadAttention(n_embd, n_heads)
        self.norm1=nn.LayerNorm(n_embd)
        self.norm2=nn.LayerNorm(n_embd)
        self.dropout=nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, n_embd * expansion_factor, bias=False),
            nn.GELU(),
            nn.Linear(n_embd * expansion_factor, n_embd, bias=False),
        )
    def forward(self, q, k, v):
        att=self.mha(q, k, v)
        out1=self.dropout(self.norm1(q+att))
        ff=self.feedforward(out1)
        out2=self.dropout(self.norm2(out1+ff))
        return out2
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, seq_len, dropout, expansion_factor):
        super(Encoder, self).__init__()
        self.te=TokenEmbeddings(vocab_size, n_embd)
        self.pe=get_positional_encodings(seq_len, n_embd).to(device)
        self.TransformerStack=nn.ModuleList([
            TransformerBlock(n_embd, n_heads, dropout, expansion_factor) for _ in range(n_layers)
        ])

    def forward(self, batch_data):
        token_embeddings=self.te(batch_data).to(device)
        ip=token_embeddings+self.pe
        for block in self.TransformerStack:
            ip=block(ip, ip, ip)
        return ip
    
class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads, expansion_factor, dropout):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.masked_attention = MultiHeadAttention(n_embd=n_embd, n_heads=n_heads)
        self.Transformer = TransformerBlock(
                                n_embd=n_embd,
                                n_heads=n_heads,
                                expansion_factor=expansion_factor,
                                dropout=dropout
                            )
        
    def forward(self, q_enc, k_enc, v_enc, q_dec, k_dec, v_dec, mask=True):
        out1 = self.dropout(self.norm(q_dec + self.masked_attention(q_dec, k_dec, v_dec, mask)))
        out2 = self.dropout(self.norm(out1 + self.Transformer(out1, k_enc, v_enc)))
        return out2



class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layers, n_embd, n_heads, expansion_factor, dropout):
        super(Decoder, self).__init__()
        self.te=TokenEmbeddings(vocab_size, n_embd)
        self.pe=get_positional_encodings(seq_len, n_embd)
        self.DecoderStack = nn.ModuleList([
            DecoderBlock(
                n_embd=n_embd,
                n_heads=n_heads,
                expansion_factor=expansion_factor,
                dropout=dropout
            ) for _ in range(n_layers)
        ]) 
        
    def forward(self, decoder_input, encoder_output):
        token_embeddings = self.te(decoder_input)
        ip = token_embeddings + self.pe.to(device)
        for block in self.DecoderStack:
            ip = block(encoder_output, encoder_output, encoder_output, ip, ip, ip)
        return ip
  
class MachineTranslator(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embd, n_heads, n_layers, expansion_factor, dropout):
        super(MachineTranslator, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            seq_len=seq_len,
            n_embd=n_embd,
            n_heads=n_heads,
            n_layers=n_layers,
            expansion_factor=expansion_factor,
            dropout=dropout
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            seq_len=seq_len,
            n_layers=n_layers,
            n_embd=n_embd,
            n_heads=n_heads,
            expansion_factor=expansion_factor,
            dropout=dropout
        )
        self.final_linear = nn.Linear(n_embd, vocab_size)

    def forward(self, encoder_input, decoder_input):
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)
        output_logits = self.final_linear(decoder_output)
        return output_logits
