import torch
from model import Transformer
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from utils import (
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    NUM_EMBED,
    NUM_HEAD,
    NUM_LAYER,
    encode,
    decode
)


tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = Tokenizer.from_file("BPE-like-tokenizer.json"),
    unk_token = "[UNK]", 
    pad_token = "[PAD]", 
    cls_token = "[CLS]", 
    sep_token = "[SEP]", 
    mask_token = "[MASK]"
)
vocab_size = tokenizer.vocab_size

# model
model = Transformer(
    vocab_size=vocab_size,
    num_embed=NUM_EMBED,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEAD,
    num_layers=NUM_LAYER,
    dropout=DROPOUT
)
# load model to GPU if available
m = model.to(DEVICE)
# print the number of parameters in the model

m = torch.load("base_model.pth", map_location=torch.device(DEVICE))
m.eval()

print(
    "Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6)
)

# generate some output based on the context
#context = torch.tensor(np.array(encode("Hello! My name is ", tokenizer)))
#context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
context_np = np.array(encode("math", tokenizer))
context_np = np.array([context_np])
context = torch.from_numpy(context_np)
print(context)
print(
    decode(
        enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
        tokenizer=tokenizer,
    )
)