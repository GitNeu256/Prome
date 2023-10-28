import os
import torch
from datetime import datetime

# hyperparameters
BATCH_SIZE = 32  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 64  # what is the maximum context length for predictions?
MAX_ITER = 3500  # number of training iterations
EVAL_INTER = 500
LEARNING_RATE = 3e-5
EPS = 1e-8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_HEAD = 7
SIZE_EMBED = 256
NUM_EMBED = NUM_HEAD * SIZE_EMBED
NUM_LAYER = 7
DROPOUT = 0.3

def encode(text_seq: str, tokenizer: any) -> torch.Tensor:
    """
    Function to encode input text using a pre-trained tokenizer and vectorized lookups
    """
    # tokenize the input text
    tokens = tokenizer.tokenize(text_seq)
    # convert the tokens to their corresponding ids
    token_indices = tokenizer.convert_tokens_to_ids(tokens)
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    return token_indices


def decode(enc_sec: torch.Tensor, tokenizer: any) -> str:
    """
    Function to decode a sequence of token indices back to a string
    """
    # convert the indices to a list
    enc_sec = enc_sec.tolist()
    # decode the indices to a string
    text = tokenizer.decode(enc_sec)
    return text


def get_batch(data: list[str], block_size: int, batch_size: int):
    """
    This is a simple function to create batches of data.
    GPUs allow for parallel processing we can feed multiple chunks at once
    so that's why we would need batches - how many independant sequences
    will we process in parallel.

    Parameters:
    data: list[str]: data to take batch from
    block_size (int): size of the text that is proccessed at once
    batch_size (int): number of sequences to process in parallel

    Returns:
    x, y: a tuple with token sequence and token target
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # we stack batch_size rows of sentences
    # so x and y are the matrices with rows_num=batch_size
    # and col_num=block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    # y is x shifted one position right - because we predict
    # word in y having all the previous words as context
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(
    data: list[str],
    model: torch.nn.Module,
    block_size: int,
    batch_size: int,
    eval_iters: int = 10,
):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data=data, block_size=block_size, batch_size=batch_size)
        logits, loss = model.forward(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out
