import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.dropout = nn.Dropout(dropout_rate)

        assert self.embd_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.depth = self.embd_dim // self.num_heads

        self.wq = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.wk = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.wv = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.dense = nn.Linear(self.embd_dim, self.embd_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, alibi, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scale = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        scaled_attention_logits = matmul_qk / scale
        scaled_attention_logits += alibi[:, :, :seq_len, :seq_len]

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)  # Apply dropout to the attention weights

        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.embd_dim)
        output = self.dense(output)
        attention_weights = attention_weights[:, 0, :, :]
        return output, attention_weights


class PositionwiseFeedforward(nn.Module):
    def __init__(self, embd_dim, d_ff, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embd_dim, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(d_ff, embd_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first linear layer and ReLU activation
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embd_dim, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embd_dim, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedforward(embd_dim, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(embd_dim)
        self.norm2 = nn.LayerNorm(embd_dim)

    def forward(self, src, alibi, src_mask=None):
        src2 = self.norm1(src)
        src2 = src2.permute(1, 0, 2)
        attn_output, attn_map = self.self_attn(src2, src2, src2, alibi, src_mask)
        attn_output = attn_output.permute(1, 0, 2)
        src = src + attn_output
        src2 = self.norm2(src)
        src = src + self.feed_forward(src2)
        return src, attn_map


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
            # In the feed forward network, the hidden layer size is 4 times the embedding dimension
            TransformerEncoderLayer(self.n_embd, self.n_head, 4 * self.n_embd) for _ in range(self.n_layer)
        ])

    def create_alibi(self, batch_size, num_heads, seq_len):
        # Create AliBi mask with the bias increasing linearly for each head
        alibi = torch.arange(seq_len).unsqueeze(0).unsqueeze(0)
        alibi = alibi.repeat(batch_size, num_heads, 1)
        alibi = alibi.unsqueeze(-1) - alibi.unsqueeze(-2)
        alibi = torch.abs(alibi).float()
        alibi = alibi.to(next(self.parameters()).device)  # Ensure same device
        return alibi

    def forward(self, x):
        # the size of x is (batch_size, seq_len, emb_dim)
        batch_size = x.size(0)
        seq_len = x.size(1)
        alibi = self.create_alibi(batch_size, self.n_head, seq_len)

        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, emb_dim)

        attention_maps = []  # List to store attention maps from each layer
        for layer in self.layers:
            x, attn_map = layer(x, alibi)
            attention_maps.append(attn_map)

        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, emb_dim)
        out = x.mean(dim=1)  # Mean over sequence dimension for classification

        return out, attention_maps


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


class UnifiedClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super(UnifiedClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        embeddings, _ = self.encoder(x)
        output = self.classifier(embeddings)
        return output
