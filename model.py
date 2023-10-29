import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import DEVICE, EPS
from torch import Tensor
import numpy as np

class PromeLayerNorm(nn.Module):
    """
    This class implements a Prome layer normalization layer.

    This layer normalizes the input activations based on the mean and standard deviation of the activations along a given dimension.

    Args:
        input_size (int): The size of the input dimension.
        epsilon (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: A tensor of the same shape as the input tensor.
    """
    def __init__(self, input_size, epsilon):
        super(PromeLayerNorm, self).__init__()
        self.input_size = input_size
        self.gamma = torch.nn.Parameter(torch.ones(input_size))
        self.beta = torch.nn.Parameter(torch.zeros(input_size))
        self.epsilon = epsilon

    def forward(self, x):
        """
        Computes the layer normalization of the input tensor.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: A tensor of the same shape as the input tensor.
        """
        # Compute the mean and standard deviation of the input.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Add a small value epsilon to the standard deviation to avoid division by zero.
        x_norm = (x - mean) / (std + self.epsilon)

        # Return the output of the normalized input, multiplied by the scaling factor and added to the bias.
        return self.gamma * x_norm + self.beta

class PromeEmbedding(nn.Module):
    """
    This class implements a Prome embedding layer.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding.
        padding_idx (int, optional): The padding index. If this is not None, then the padding index will be masked out when calculating the embedding.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim).
    """
    def __init__(self, vocab_size, embedding_dim, padding_idx = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, embedding_dim))
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        """
        Calculates the embedding for the given input IDs.

        Args:
            input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        input_ids = input_ids.long()
        if self.padding_idx is not None:
            input_ids = input_ids.masked_fill(input_ids == self.padding_idx, 0)

        # get symbol vector
        output = self.weight[input_ids]

        return output
    
class AttentionHead(nn.Module):
    """
    One head of the self-attention layer
    """

    def __init__(self, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        # tril is a lower triangular matrix. it is not a parameter
        # of the model, so we assign it to the module using register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # let's also add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Tril matrix (lower triagular matrix) is used to mask 
        # future positions (setting them to -inf) so that the
        # decoder "learns" to predict next words
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)

        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple Heads of self-attention in parallel
    """

    def __init__(self, num_heads, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size=head_size,
                    num_embed=num_embed,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_embed, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # output of the self-attention
        out = torch.concat([h(x) for h in self.heads], dim=-1)
        # apply the linear projection layer
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    A simple linear layer followed by GeLu
    """

    def __init__(self, num_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.GELU(),
            nn.Linear(4 * num_embed, num_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    This calss will group together MultiHead Attention and
    FeedForward NN, so that we can copy it in Transformer
    """

    def __init__(self, num_heads, block_size, num_embed, dropout):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            num_embed=num_embed,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffwd = FeedForward(num_embed=num_embed, dropout=dropout)
        # add the layer normalization
        self.ln1 = PromeLayerNorm(num_embed, EPS)
        self.ln2 = PromeLayerNorm(num_embed, EPS)

    def forward(self, x):
        # "x +" is the skip (or residual) connection
        # it helps with optimization
        # also we apply layer normalization before self-attention
        # and feed-forward (a reshufle from original paper)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class TransformerEncoder(nn.Module):
    """
    This class implements a Transformer encoder.

    Args:
        num_heads (int): The number of attention heads.
        block_size (int): The size of the input sequence.
        num_embed (int): The dimension of the embedding.
        num_layers (int): The number of encoder blocks.
        dropout (float): The dropout rate.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim).
    """
    def __init__(self, num_heads, block_size, num_embed, num_layers, dropout):
        super().__init__()
        # Create the embedding layer.
        self.pemb = PromeEmbedding(block_size, num_embed)

        # Create a sequential block of Transformer blocks.
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    num_heads=num_heads,
                    block_size=block_size,
                    num_embed=num_embed,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Encodes the input sequence.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim).
        """

        # Pass the embedded input sequence through the sequential block of Transformer blocks.
        for block in self.blocks:
            x = block(x)

        return x

class TransformerDecoder(nn.Module):
    """
    This class implements a Transformer decoder.

    Args:
        num_heads (int): The number of attention heads.
        block_size (int): The size of the input sequence.
        num_embed (int): The dimension of the embedding.
        num_layers (int): The number of decoder blocks.
        dropout (float): The dropout rate.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim).
    """
    def __init__(self, num_heads, block_size, num_embed, num_layers, dropout):
        super().__init__()

        # Create the embedding layer.
        self.pemb = PromeEmbedding(block_size, num_embed)

        # Create a sequential block of Transformer blocks.
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    num_heads=num_heads,
                    block_size=block_size,
                    num_embed=num_embed,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Create the encoder.
        self.encoder = TransformerEncoder(num_heads, block_size, num_embed, num_layers, dropout)

        # Create a list of residual connections.
        self.residuals = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # Create a layer normalization layer.
        self.norm = PromeLayerNorm(num_embed, EPS)
        # Create a dropout layer.
        self.dropout = nn.Dropout(dropout)
        # Create a softmax layer.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Decodes the input sequence.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim).
        """

        # Encode the input sequence.
        output_encoder = self.encoder(x)

        # Add positional encodings to the input sequence.
        x = x + self.pemb(torch.arange(x.size(1)))

        # Pass the input sequence through the sequential block of Transformer blocks, with residual connections and layer normalization.
        for block, residual in zip(self.blocks, self.residuals):
            x = block(x)
            x = x + residual(x)
            x = x + output_encoder
            x = self.norm(x)
            x = self.dropout(x)

        # Apply a softmax layer to the output of the last Transformer block.
        x = self.softmax(x)

        return x
    
class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # a simple lookup table that stores embeddings of a fixed dictionary and size
        # each token directly reads off the logits for the next token from a lookup table
        # see more: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.vocab_size = kwargs.get("vocab_size", 100)
        self.num_embed = kwargs.get("num_embed", 32)
        self.block_size = kwargs.get("block_size", 8)
        self.num_heads = kwargs.get("num_heads", 4)
        self.num_layers = kwargs.get("num_layers", 4)
        self.dropout = kwargs.get("dropout", 0.2)
        # each token reads the logits for the next token from a lookup table
        self.token_embedding_table = PromeEmbedding(self.vocab_size, self.num_embed)
        # each position from 0 to block_size-1 will get its embedding
        self.position_embedding_table = PromeEmbedding(self.block_size, self.num_embed)

        self.decoder = TransformerDecoder(self.num_heads, self.block_size, self.num_embed, self.num_layers, self.dropout)

        # we add the layer norm before the Linear layer
        self.ln_f = PromeLayerNorm(self.num_embed, EPS)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are (B,T) tensor of integers
        # the token_emb is (B, T, C), C = NUM_EMBED
        token_emb = self.token_embedding_table(idx)
        # (T, C)
        posit_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))

        x = token_emb + posit_emb

        # apply one head of self-attention
        x = self.decoder(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)

        # Compute the loss
        if targets != None:
            # cross_entropy accepts inputs in a (batch_size, num_classes)
            # so we need to reformat our logits dimensions to
            # (batch_size * time, dim_vocabulary), time = block_size
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T, ))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx