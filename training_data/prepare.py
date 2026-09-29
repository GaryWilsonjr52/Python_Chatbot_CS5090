import os
import glob
import json
import random
import pickle
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

VOCAB_SIZE = 8192
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'files')
os.makedirs(dataset_dir, exist_ok=True)

# 1. Read all .py files AND extract code cells from .ipynb notebooks
py_files = glob.glob(os.path.join(dataset_dir, '**', '*.py'), recursive=True)
ipynb_files = glob.glob(os.path.join(dataset_dir, '**', '*.ipynb'), recursive=True)
print(f"Found {len(py_files)} .py files and {len(ipynb_files)} .ipynb files.")

data_chunks = []

for file_path in py_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if len(text) > 50:
                data_chunks.append(text)
    except Exception as e:
        pass

for nb_path in ipynb_files:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            cells = []
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = "".join(cell.get('source', [])).strip()
                    # Skip magic commands like !pip install or %matplotlib
                    clean_lines = [
                        line for line in source.splitlines()
                        if not line.strip().startswith(('!', '%'))
                    ]
                    if clean_lines:
                        cells.append("\n".join(clean_lines))
            nb_code = "\n\n".join(cells).strip()
            if len(nb_code) > 50:
                data_chunks.append(nb_code)
    except Exception as e:
        pass

# Shuffle chunks so all 9 repos are evenly mixed between train.bin and val.bin
random.seed(42)
random.shuffle(data_chunks)
print(f"Total valid code scripts extracted and shuffled: {len(data_chunks)}")

full_text = "\n\n[EOF]\n\n".join(data_chunks)
print(f"Length of dataset in characters: {len(full_text):,}")

# 2. Train Custom BPE Tokenizer with vocab_size=2048
print(f"Training custom BPE tokenizer (vocab_size={VOCAB_SIZE})...")
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[EOF]"],
    show_progress=True
)
tokenizer.train_from_iterator(data_chunks, trainer=trainer)

tokenizer_path = os.path.join(base_dir, 'tokenizer.json')
tokenizer.save(tokenizer_path)

# 3. Encode dataset
encoded = tokenizer.encode(full_text)
tokens = encoded.ids
print(f"Dataset has {len(tokens):,} tokens")

# 4. 90/10 Train/Val Split
n = len(tokens)
train_ids = np.array(tokens[:int(n * 0.9)], dtype=np.uint16)
val_ids = np.array(tokens[int(n * 0.9):], dtype=np.uint16)

train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

# 5. Save updated vocab_size in meta.pkl
meta = {'vocab_size': VOCAB_SIZE}
with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved train.bin ({len(train_ids):,} tokens), val.bin ({len(val_ids):,} tokens), and meta.pkl (vocab_size={VOCAB_SIZE}).")