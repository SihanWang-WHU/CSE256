import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ Single element of self-attention mechanism """

    def __init__(self, dimension_size, embed_dim, sequence_length, drop_prob=0.2):
        super().__init__()
        self.encrypt = nn.Linear(embed_dim, dimension_size, bias=False)
        self.decrypt = nn.Linear(embed_dim, dimension_size, bias=False)
        self.combine = nn.Linear(embed_dim, dimension_size, bias=False)
        self.register_buffer('mask_lower_triangular', torch.tril(torch.ones(sequence_length, sequence_length)))

    def forward(self, sequence):
        batch_size, seq_len, channels = sequence.shape
        key = self.encrypt(sequence)
        query = self.decrypt(sequence)
        attention_scores = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        attention_scores = attention_scores.masked_fill(self.mask_lower_triangular[:seq_len, :seq_len] == 0,
                                                        float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        value = self.combine(sequence)
        output = attention_scores @ value
        return output, attention_scores


class MultiHeadAttention(nn.Module):
    """ Group of self-attention elements operating in parallel """

    def __init__(self, count_heads, size_head, embed_dim, sequence_length, drop_prob=0.2):
        super().__init__()
        self.cluster = nn.ModuleList([Head(size_head, embed_dim, sequence_length) for _ in range(count_heads)])
        self.final_projection = nn.Linear(size_head * count_heads, embed_dim)

    def forward(self, sequence):
        outputs, attentions = zip(*[element(sequence) for element in self.cluster])
        output = torch.cat(outputs, dim=-1)
        attentions = torch.stack(attentions, dim=1)  # Stack along a new head dimension
        attentions = attentions[:, 0, :, :]
        return self.final_projection(output), attentions


class FeedForward(nn.Module):
    """ Linear transformation followed by an activation function """

    def __init__(self, embed_dim, drop_prob=0.2):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, sequence):
        return self.pipeline(sequence)


class Block(nn.Module):
    """ Transformer module integrating self-attention and feedforward network """

    def __init__(self, embed_dim, num_heads, sequence_length):
        super().__init__()
        dimension_size = embed_dim // num_heads
        self.attention = MultiHeadAttention(num_heads, dimension_size, embed_dim, sequence_length)
        self.transformation = FeedForward(embed_dim)
        self.normalize1 = nn.LayerNorm(embed_dim)
        self.normalize2 = nn.LayerNorm(embed_dim)

    def forward(self, sequence):
        sequence_normed = self.normalize1(sequence)
        attended_sequence, attention_maps = self.attention(sequence_normed)
        sequence = sequence + attended_sequence
        sequence = sequence + self.transformation(self.normalize2(sequence))
        return sequence, attention_maps


class GPTLanguageModel(nn.Module):
    """ GPT model that returns logits, loss, and attention maps """

    def __init__(self, size_vocabulary, embed_dim, num_heads, num_layers, sequence_length):
        super().__init__()
        self.token_embeddings = nn.Embedding(size_vocabulary, embed_dim)
        self.position_embeddings = nn.Embedding(sequence_length, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, sequence_length) for _ in range(num_layers)])
        self.final_normalization = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, size_vocabulary)

    def forward(self, index, targets=None):
        batch_size, time_steps = index.shape
        token_embedding = self.token_embeddings(index)
        position_embedding = self.position_embeddings(torch.arange(time_steps, device=index.device))
        combined_embedding = token_embedding + position_embedding
        attentions = []
        for block in self.blocks:
            combined_embedding, attention_maps = block(combined_embedding)
            attentions.append(attention_maps)
        final_output = self.final_normalization(combined_embedding)
        logits = self.output_head(final_output)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss, torch.stack(attentions)

    def generate(self, index, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            index_conditioned = index[:, -block_size:]
            logits, _ = self(index_conditioned)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            index = torch.cat((index, next_index), dim=1)
        return index
