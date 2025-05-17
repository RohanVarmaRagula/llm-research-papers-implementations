import json
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from model import MachineTranslator

random.seed(101)
torch.manual_seed(101)

with open('config.json', 'r') as f:
    config = json.load(f)

seq_len = config["seq_len"]
n_data = config["n_data"]
n_embd = config["n_embd"]
vocab_size = config["vocab_size"]
batch_size = config["batch_size"]
n_heads = config["n_heads"]
dropout = config["dropout"]
expansion_factor = config["expansion_factor"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
    

def tokenize_dataset(data, tokenizer):
    german = [x["de"] for x in data] 
    english = [x["en"] for x in data] 

    return tokenizer(
        english,
        text_target=german,
        max_length=seq_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

def get_batch(data, n_batch):
    random_ids = random.sample(range(0, n_data), n_batch)
    return {
        'input_ids': data['input_ids'][random_ids],
        'attention_mask': data['attention_mask'][random_ids],
        'labels': data['labels'][random_ids]
    }

dataset = load_dataset("wmt/wmt14", "de-en")
train = dataset["train"].select(range(n_data))['translation']
test = dataset["test"]['translation']

mbart_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
mbart_tokenizer.src_lang = "en_XX"
mbart_tokenizer.tgt_lang = "de_DE"
print(f"Vocab size: {len(mbart_tokenizer)}")
print(f"Source Language: {mbart_tokenizer.src_lang}")
print(f"Target Language: {mbart_tokenizer.tgt_lang}")

train_tokens = tokenize_dataset(train, mbart_tokenizer)
test_tokens = tokenize_dataset(test, mbart_tokenizer)

model = MachineTranslator(vocab_size, seq_len, n_embd, n_heads, n_layers, expansion_factor, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=mbart_tokenizer.pad_token_id)
optimizer = optim.Adam(
    model.parameters(),
      lr=learning_rate,
      betas=(0.9, 0.98),
      eps=1e-9
)
num_batches = 100
print(f"Running the model for a toal of {epochs} epochs with {num_batches} batches each.")
for i in range(epochs):
    epoch_loss = 0
    for batch_i in range(num_batches):
        batch = get_batch(train_tokens, batch_size)
        
        encoder_input = batch['input_ids'].to(device)
        target_ids = batch['labels'].to(device)
        decoder_input = torch.zeros(target_ids.shape)
        # decoder's input is right shift by one token as given in paper, the first token is set as the start token from our tokenizer
        decoder_input[:, 0] = mbart_tokenizer.lang_code_to_id[mbart_tokenizer.tgt_lang]
        decoder_input[:, 1:] = target_ids[:, :-1]
        decoder_input = decoder_input.long()
        
        optimizer.zero_grad()
        logits = model(encoder_input, decoder_input)
        logits = logits.view(-1, logits.size(-1))  
        target_ids = target_ids.view(-1) 
        loss = criterion(logits, target_ids)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        epoch_loss += loss.item()
        if batch_i % 100 == 0:
            print(f"|----Epoch {i}.{batch_i} : Loss of the current batch {loss.item():.4f}")
    print(f"Epoch {i} : Average Loss per Batch {epoch_loss / num_batches:.4f}")
    