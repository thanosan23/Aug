import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        x = x.view(batch_size * sequence_length, d_model)

        Q = self.query(x).view(batch_size, sequence_length, d_model)
        K = self.key(x).view(batch_size, sequence_length, d_model)
        V = self.value(x).view(batch_size, sequence_length, d_model)

        scores = Q.bmm(K.transpose(1, 2)) / math.sqrt(d_model)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.bmm(V).view(batch_size * sequence_length, d_model)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            NewGELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        attended = self.attention(x, mask).view(batch_size, sequence_length, d_model)
        x = self.norm1(attended + x)
        fedforward = self.ffn(x)
        return self.norm2(fedforward + self.dropout(x))
    
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(256, d_model) 
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.text) - self.sequence_length

    def __getitem__(self, idx):
        return self.text[idx:idx+self.sequence_length], self.text[idx+1:idx+self.sequence_length+1]

def create_dataset(input_file, sequence_length):
    with open(input_file, 'r') as f:
        text = f.read()
    text = torch.tensor([ord(c) for c in text], dtype=torch.long)  # Convert characters to integers
    return TextDataset(text, sequence_length)

def train_model(model, dataset, batch_size, num_steps):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters())
    for step, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss {loss.item()}")
        if step >= num_steps:
            break

def generate_name(model, max_length=10):
    model.eval() 
    with torch.no_grad():
        name = '' 
        x = torch.zeros(1, 1, dtype=torch.long) 
        for _ in range(max_length):
            output = model(x)
            probabilities = F.softmax(output[0, -1, :], dim=-1)
            next_character = torch.multinomial(probabilities, 1).item()
            next_character %= model.embedding.num_embeddings
            name += chr(next_character)
            x = torch.cat([x, torch.tensor([[next_character]], dtype=torch.long)], dim=1)
        return name

# dataset = create_dataset('input.txt', sequence_length=100)
# model = Transformer(d_model=512, nhead=8, num_layers=6)
# train_model(model, dataset, batch_size=64, num_steps=20)
# print(generate_name(model))