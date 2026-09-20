import os
import glob
import pickle
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# Define the path to your dataset
dataset_dir = 'PythonChatbot_CS5090/training_data/files'
os.makedirs(dataset_dir, exist_ok=True)

# 1. Read all Python files
python_files = glob.glob(os.path.join(dataset_dir, '**', '*.py'), recursive=True)
print(f"Found {len(python_files)} Python files.")

# add the text from the files into datachunks
data_chunks = []
for file_path in python_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_chunks.append(f.read())
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

# Combine all code with newlines separating files
full_text = "\n\n".join(data_chunks)
print(f"Length of dataset in characters: {len(full_text):,}")

# 2. Initialize and Train a Custom BPE Tokenizer
print("Training custom BPE tokenizer (vocab_size=8192)...")
tokenizer = Tokenizer(BPE(unk_token="[UNK]")) #UNK is for a fallback if any data is unrecognizeable.
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False) #convert to bytes to prevent "out of vocabulary" error

# Configure the trainer for 8192 tokens
# special tokens are for:
# unkown data
# remove padding to make sequences all the same length while training
# BOS and EOS are used for telling the model where a file starts and stops
trainer = BpeTrainer(
    vocab_size=8192, 
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    show_progress=True
)

# Train the tokenizer on our Python text
tokenizer.train_from_iterator(data_chunks, trainer=trainer)

# Save the tokenizer configuration so you can decode model outputs later
tokenizer_path = os.path.join(dataset_dir, 'tokenizer.json')
tokenizer.save(tokenizer_path)
print(f"Saved tokenizer config to {tokenizer_path}")

# 3. Encode the dataset using the newly trained tokenizer
print("Encoding dataset...")
encoded = tokenizer.encode(full_text)
tokens = encoded.ids
print(f"Dataset has {len(tokens):,} tokens")

# 4. Create the train and test splits (90/10 split)
n = len(tokens)
train_ids = tokens[:int(n * 0.9)]
val_ids = tokens[int(n * 0.9):]
print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# 5. Export to bin files, uint16 is used to save space
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(dataset_dir, 'train.bin'))
val_ids.tofile(os.path.join(dataset_dir, 'val.bin'))

# 6. Save meta.pkl for nanoGPT's train.py
# NanoGPT's train.py looks for vocab_size in meta.pkl
meta = {
    'vocab_size': 8192,
}
with open(os.path.join(dataset_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
    
print("Preparation complete! train.bin, val.bin, and meta.pkl saved.")