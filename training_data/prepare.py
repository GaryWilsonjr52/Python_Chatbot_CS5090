%%writefile /content/Python_Chatbot_CS5090/training_data/prepare.py
import os
import ast
import re
import glob
import json
import random
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, Regex, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel

warnings.filterwarnings("ignore", category=SyntaxWarning)

VOCAB_SIZE = 8192
TARGET_CHARS = 400_000_000  # ~400MB of raw code -> ~100M tokens
EOF_SEP = "\n\n[EOF]\n\n"

base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
dataset_dir = os.path.join(base_dir, 'files')
os.makedirs(dataset_dir, exist_ok=True)

PANDAS_INDICATORS = re.compile(
    r'\b(pd\.|df\.|df\[|DataFrame|Series|read_csv|read_excel|read_parquet|'
    r'to_csv|to_datetime|groupby|iloc|loc|fillna|dropna|merge|concat|pivot_table|'
    r'rolling|ewm|resample|pct_change|shift|cumsum|cumprod|quantile|assign|apply)\b'
)

def extract_pandas_functions(code_str):
    """Parses Python source into AST and extracts standalone functions that use Pandas."""
    if "pandas" not in code_str and "pd." not in code_str and "df" not in code_str:
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code_str)
    except Exception:
        return []

    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                func_src = ast.unparse(node)
                lines = func_src.splitlines()
                if 4 <= len(lines) <= 250 and PANDAS_INDICATORS.search(func_src):
                    funcs.append(func_src)
            except Exception:
                continue
    return funcs

# 1. Read all local .py and .ipynb files
py_files = glob.glob(os.path.join(dataset_dir, '**', '*.py'), recursive=True)
ipynb_files = glob.glob(os.path.join(dataset_dir, '**', '*.ipynb'), recursive=True)
print(f"Found {len(py_files)} .py files and {len(ipynb_files)} .ipynb files.")

data_chunks = []
total_chars = 0

for fpath in tqdm(py_files, desc="Processing local .py files"):
    try:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        for func in extract_pandas_functions(code):
            data_chunks.append(func)
            total_chars += len(func) + len(EOF_SEP)
    except Exception:
        continue

for fpath in tqdm(ipynb_files, desc="Processing local .ipynb files"):
    try:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            nb = json.load(f)
        cell_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        for func in extract_pandas_functions(cell_code):
            data_chunks.append(func)
            total_chars += len(func) + len(EOF_SEP)
    except Exception:
        continue

print(f"Local files yielded {len(data_chunks):,} Pandas functions ({total_chars / 1e6:.2f} MB).")

# 2. Stream CodeParrot from Hugging Face until hitting ~100M Tokens
print("Streaming 'codeparrot/codeparrot-clean' to reach ~100M tokens (~400MB)...")
try:
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    pbar = tqdm(total=TARGET_CHARS, initial=total_chars, unit="char", desc="Streaming CodeParrot")
    
    for row in ds:
        if total_chars >= TARGET_CHARS:
            break
        content = row.get("content", "")
        for func in extract_pandas_functions(content):
            data_chunks.append(func)
            chunk_len = len(func) + len(EOF_SEP)
            total_chars += chunk_len
            pbar.update(chunk_len)
            if total_chars >= TARGET_CHARS:
                break
    pbar.close()
except Exception as e:
    print(f"Stopped or failed streaming CodeParrot: {e}")

random.seed(42)
random.shuffle(data_chunks)
print(f"Total valid Pandas functions extracted and shuffled: {len(data_chunks):,}")
print(f"Total dataset size in characters: {total_chars:,}")

# 3. Train Custom Line- & Indent-Aware BPE Tokenizer (vocab_size=8192)
print(f"Training custom BPE tokenizer (vocab_size={VOCAB_SIZE})...")
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

newline_indent_regex = Regex(r"\n[ \t]*|[^\n]+")
tokenizer.pre_tokenizer = Sequence([
    Split(pattern=newline_indent_regex, behavior="isolated"),
    ByteLevel(add_prefix_space=False, use_regex=True)
])
tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[EOF]"],
    show_progress=True
)

train_sample = data_chunks[:100_000] if len(data_chunks) > 100_000 else data_chunks
tokenizer.train_from_iterator(train_sample, trainer=trainer)

tokenizer_path = os.path.join(base_dir, 'tokenizer.json')
tokenizer.save(tokenizer_path)

# 4. 90/10 Split & Memory-Safe Batch Encoding to train.bin / val.bin
split_idx = int(len(data_chunks) * 0.9)
train_chunks = data_chunks[:split_idx]
val_chunks = data_chunks[split_idx:]

def encode_and_save_split(chunks, out_filename):
    bin_path = os.path.join(base_dir, out_filename)
    total_tokens = 0
    batch_size = 4096
    
    with open(bin_path, 'wb') as f:
        for i in tqdm(range(0, len(chunks), batch_size), desc=f"Writing {out_filename}"):
            batch = [c + EOF_SEP for c in chunks[i : i + batch_size]]
            encodings = tokenizer.encode_batch(batch)
            flat_ids = [token_id for enc in encodings for token_id in enc.ids]
            arr = np.array(flat_ids, dtype=np.uint16)
            f.write(arr.tobytes())
            total_tokens += len(arr)
            
    return total_tokens

print("Encoding dataset splits to binary files...")
train_token_count = encode_and_save_split(train_chunks, 'train.bin')
val_token_count = encode_and_save_split(val_chunks, 'val.bin')

# 5. Save updated vocab_size and mappings in meta.pkl
vocab = tokenizer.get_vocab()
meta = {
    'vocab_size': VOCAB_SIZE,
    'itos': {v: k for k, v in vocab.items()},
    'stoi': {k: v for k, v in vocab.items()},
}
with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nSaved train.bin ({train_token_count:,} tokens), val.bin ({val_token_count:,} tokens), and meta.pkl (vocab_size={VOCAB_SIZE}).")
print(f"Total tokens generated: {train_token_count + val_token_count:,}")