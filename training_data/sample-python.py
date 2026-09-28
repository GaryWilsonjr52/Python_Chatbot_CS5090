"""
Sample from a trained model, trim mid-file start fragments and cut-off end lines,
validate Python syntax via AST, and save to out_dir/generated_samples.py
"""
import os
import ast
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from tokenizers import Tokenizer, decoders
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = '/content/training_out' # default out_dir, can also be overridden via command line
start = "\n" # use newline so ByteLevel BPE doesn't emit a trailing standalone space token
num_samples = 5 # number of valid samples to draw
max_new_tokens = 800 # gives scripts room to complete
temperature = 0.7 # balanced temperature for valid syntax
top_k = 150 # retain top_k most likely tokens
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model checkpoint
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Load custom Hugging Face BPE tokenizer
tokenizer_path = '/content/Python_Chatbot_CS5090/training_data/tokenizer.json'

if os.path.exists(tokenizer_path):
    print(f"Loading custom BPE tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = decoders.ByteLevel()
    encode = lambda s: tokenizer.encode(s).ids
    decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)
else:
    print("Warning: Custom tokenizer.json not found! Falling back to GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


def clean_and_trim_code(code_str):
    """
    Finds the longest syntactically valid Python block by trimming broken
    mid-file lines from the top and cut-off lines from the bottom.
    """
    lines = code_str.strip().splitlines()
    best_candidate = None
    best_len = 0

    for start_idx in range(len(lines)):
        line = lines[start_idx]
        # Only start at unindented top-level lines
        if line and (line[0] not in (' ', '\t', '"', "'")):
            sub_lines = lines[start_idx:]
            while len(sub_lines) >= 5:
                # Strip trailing blank lines or trailing comments
                while sub_lines and (not sub_lines[-1].strip() or sub_lines[-1].strip().startswith('#')):
                    sub_lines.pop()
                if len(sub_lines) < 5:
                    break

                candidate = "\n".join(sub_lines)
                try:
                    ast.parse(candidate)
                    # Pop dangling cut-off variable names at the very end (like `name_` or `self`)
                    last_token = sub_lines[-1].strip()
                    if last_token.isidentifier() and last_token not in ('pass', 'break', 'continue', 'return', 'yield'):
                        sub_lines.pop()
                        continue

                    if len(sub_lines) > best_len:
                        best_candidate = candidate
                        best_len = len(sub_lines)
                    break
                except SyntaxError:
                    sub_lines.pop()

    if best_candidate:
        return best_candidate, True
    return code_str.strip(), False


# Encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Prepare output file path inside out_dir (/content/training_out/generated_samples.py)
os.makedirs(out_dir, exist_ok=True)
output_file_path = os.path.join(out_dir, 'generated_samples.py')

valid_samples = []
max_attempts = num_samples * 3
attempt = 0

print(f"Generating {num_samples} syntactically valid Python samples...\n")

with torch.no_grad():
    with ctx:
        while len(valid_samples) < num_samples and attempt < max_attempts:
            attempt += 1
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            raw_code = decode(y[0].tolist())

            cleaned_code, is_valid = clean_and_trim_code(raw_code)

            if is_valid:
                sample_num = len(valid_samples) + 1
                print(f"# ==================== SAMPLE {sample_num} (Syntax: VALID, {len(cleaned_code.splitlines())} lines) ====================")
                print(cleaned_code)
                print('---------------')
                valid_samples.append(f"# {'='*20} SAMPLE {sample_num} {'='*20}\n{cleaned_code}\n")
            else:
                print(f"[Attempt {attempt}] Skipped unparsable sample; regenerating...")

# Write all validated samples to /content/training_out/generated_samples.py
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(valid_samples))

print(f"\nSuccessfully saved {len(valid_samples)} valid Python sample(s) to: {output_file_path}")