import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embd_dim = embd_dim

        # Ensure the embedding dimension is divisible by the number of heads
        assert self.embd_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.depth = self.embd_dim // self.num_heads

        self.wq = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.wk = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.wv = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.dense = nn.Linear(self.embd_dim, self.embd_dim)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scale = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        scaled_attention_logits = matmul_qk / scale

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.embd_dim)

        # Final linear layer
        output = self.dense(output)
        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, embd_dim, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(embd_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embd_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embd_dim, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(embd_dim, num_heads)
        self.feed_forward = PositionwiseFeedforward(embd_dim, d_ff)
        self.norm1 = nn.LayerNorm(embd_dim)
        self.norm2 = nn.LayerNorm(embd_dim)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src = src + self.self_attn(src2, src2, src2, src_mask)
        src2 = self.norm2(src)
        src = src + self.feed_forward(src2)
        return src


# Example of using this custom TransformerEncoderLayer in your model
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size

        self.embed = nn.Embedding(self.vocab_size, self.n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(self.n_embd, self.n_head, 4 * self.n_embd) for _ in range(self.n_layer)
        ])

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, emb_dim)

        for layer in self.layers:
            x = layer(x)

        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, emb_dim)
        out = x.mean(dim=1)  # Mean over sequence dimension for classification
        return out


class FeedforwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.log_softmax(x)