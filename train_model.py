import os
import random
import torch
from model import Transformer
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # pip install transformers
from tokenizers import Tokenizer
from utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    NUM_EMBED,
    NUM_HEAD,
    NUM_LAYER,
    MAX_ITER,
    EVAL_INTER,
    encode,
    decode,
    get_batch,
    estimate_loss,
    EPS
)

# load model from checkpoint
# m = load_model_from_checkpoint(Transformer,vocab_size=vocab_size)

# example to decode sequence
# enc_sec = m.generate(idx=torch.zeros((1,1), dtype=torch.long),
# max_new_tokens=20)[0].tolist()
# print(decode(vocab=vocab, enc_sec=enc_sec))

# raw data
data_raw_mass = []
pre_val_data_mass = []
path_do_data = "data/english"
for file in os.listdir(path_do_data):
    data_raw = open(f"{path_do_data}/{file}", encoding="utf-8").read()
    data_raw_mass.append(data_raw[:int(0.8 * len(data_raw))])
    pre_val_data_mass.append(data_raw[int(0.8 * len(data_raw)):])

data_raw = "\n".join(data_raw_mass)
pre_val_data = "\n".join(pre_val_data_mass)
    
# we use pretrained BERT tokenizer for performance improvements
#tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = Tokenizer.from_file("BERT-like-tokenizer.json")
)
vocab_size = tokenizer.vocab_size
# data_raw = data_raw[4000000:] # short dataset
#pre_val_data = open("data/english.txt", encoding = "utf-8").read()

# train/val split
data = encode(text_seq=data_raw, tokenizer=tokenizer)
#n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data #[:n]
val_data = encode(text_seq=pre_val_data, tokenizer=tokenizer) #data[n:]

# train a new model
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
print(
    "Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6)
)
# optimizer takes the model's parameters and the learning rate as input,
# and updates the parameters during the training process in order to
# minimize the loss function.
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE, eps = EPS)

best_loss = 10000

for step in range(MAX_ITER):
    # every EVAL_INTER evaluate the loss on train and val sets
    if step % EVAL_INTER == 0 or step == MAX_ITER - 1:
        loss_train = estimate_loss(
            data=train_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        loss_val = estimate_loss(
            data=val_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))

    # sample a batch of data
    xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    logits, loss = m.forward(xb, yb)
    # zero_grad() method sets the gradients of all parameters in the optimizer to zero
    optimizer.zero_grad(set_to_none=True)
    # backward() method on the loss variable calculates the gradients 
    # of the loss with respect to the model's parameters.
    loss.backward()
    # step() method on the optimizer updates the model's parameters 
    # using the calculated gradients, in order to minimize the loss.
    optimizer.step()

torch.save(model, 'base_model.pth')

# generate some output based on the context
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(
    decode(
        enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
        tokenizer=tokenizer,
    )
)
