import torch
import torch.nn as nn
from torch.nn import functional as F

# parameters
batch_size = 32  # number of independent sequences to run in parallel
block_size = 8  # max context lenght of the sequences
max_iters = 5000
eval_interval = 500  # iterations between model evaluations
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # n of batches to evaluate performance (more batches -> more precise)
n_embd = 32 # embedding dimension, can also be refered as C

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()

# tokenization, vocab at character level
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# train-test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loader
def get_batch(split):
    # samples a random batch of x's and y's
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data)-block_size, (batch_size,))  # random index
    x = torch.stack([data[i:block_size+i] for i in ix])
    y = torch.stack([data[i+1:block_size+i+1] for i in ix])
    return x.to(device), y.to(device)

# evaluation
@torch.no_grad()  # no need to keep track of gradients to eval
def estimate_loss():
    out = {}
    model.eval()  # choose inference mode for layers like dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # reactivate dropout
    return out

class Head(nn.Module):
    '''' one head of self-attention '''
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) # non-trainable parameters

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B, T, head_size
        q = self.key(x) # B, T, head_size
        # compute affinities
        wei = q @ k.transpose(-2,-1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))  # tril[:T,:T] is used for compatibility with generate method
        wei = F.softmax(wei, dim=-1) # B, T, T
        # agregate values
        v = self.value(x) # B, T, head_size
        out = wei @ v # (B, T, T) @ (B, T, head_size) ->B, T, head_size
        return out

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token reads logits for the next one from this lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B, T, n_emdb)
        x = self.sa_head(x) # (B, T, head_size=n_embd)
        logits = self.lm_head(x) # (B,T,vocab_size)

        # cross_entropy takes input B, C, T for multiple outputs. In this case we need a different output per sequence (T)
        # or we can explode the batch dimension by the number of sequences and generate a "bigger" batch. At the end we just need to take a bunch of examples, each one with its corresponding class probabilities
        if targets is None: # compatibility with generate method (no target provided)
            loss = None 
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx B, T (it is the current_context)
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -block_size:]
            # predict
            logits, loss = self(idx_cond)
            # logits is B, T, C-> we only want the last timestep logits
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx

model = BigramLanguageModel()
m = model.to(device)  # move model parameters to gpu

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # evaluate every eval_interval iterations
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')  # sample a batch
    logits, loss = m(xb, yb)  # calculate loss
    optimizer.zero_grad(set_to_none=True)  # reset gradients
    loss.backward()  # calculate gradients wrt loss
    optimizer.step()  # update weights

# generate using model
# init context with first token (\n)
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, 500)[0].tolist()))
