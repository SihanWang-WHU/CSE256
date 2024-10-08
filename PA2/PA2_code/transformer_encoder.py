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
        # x is initially in shape (batch_size, seq_len, emb_dim)
        # Split the emb_dim into (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # Permute to get dimensions (batch_size, num_heads, seq_len, depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)  # Make sure this is indeed sequence length

        # the size of q, k,v is (batch_size, num_heads, seq_len, depth)
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Ensure calculations respect the seq_len and batch_size distinction
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # Heads, Batch, Seq_len
        scale = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        scaled_attention_logits = matmul_qk / scale
        # print("Scaled Attention Logits Shape:", scaled_attention_logits.shape)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # the size of attention_weights is (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        # print("Attention Weights Sum:", attention_weights.sum(dim=-1))

        output = torch.matmul(attention_weights, v)

        # Correcting the concatenate and reshape logic
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.embd_dim)  # Correcting this line to use seq_len

        # Final linear layer
        output = self.dense(output)
        attention_weights = attention_weights[:, 0, :, :]
        return output, attention_weights


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
        # the size of the src is (seq_len, batch_size, emb_dim)
        src2 = self.norm1(src)
        src2 = src2.permute(1, 0, 2)
        # the size of the attn_map is (batch_size, num_heads, seq_len, seq_len)
        attn_output, attn_map = self.self_attn(src2, src2, src2, src_mask)  # capture attention map
        attn_output = attn_output.permute(1, 0, 2)
        src = src + attn_output
        src2 = self.norm2(src)
        src = src + self.feed_forward(src2)
        return src, attn_map


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
            # In the feed forward network, the hidden layer size is 4 times the embedding dimension
            TransformerEncoderLayer(self.n_embd, self.n_head, 4 * self.n_embd) for _ in range(self.n_layer)
        ])

    def forward(self, x):
        # the size of x is (batch_size, seq_len, emb_dim)
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, emb_dim)

        attention_maps = []  # List to store attention maps from each layer
        for layer in self.layers:
            x, attn_map = layer(x)
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
