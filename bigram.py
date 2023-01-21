import torch
import torch.nn as nn
from torch.nn import functional as F

# parameters
batch_size = 4  # number of independent sequences to run in parallel
block_size = 8  # max context lenght of the sequences
max_iters = 30000
eval_interval = 300  # iterations between model evaluations
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n of batches to evaluate performance (more batches -> more precise)
eval_iters = 200

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

# simple bigram model


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token reads logits for the next one from this lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        # cross_entropy takes input B, C, T for multiple outputs. In this case we need a different output per sequence (T)
        # or we can explode the batch dimension by the number of sequences and generate a "bigger" batch. At the end we just need to take a bunch of examples, each one with its corresponding class probabilities
        if targets is None:
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
            # logits is B, T, C-> we only want the last timestep logits
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx


model = BigramLanguageModel(vocab_size)
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
