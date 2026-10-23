"""
Sample raw text from a trained model using a custom prompt,
and save it directly to out_dir/generated_sample.py
"""
import os
from contextlib import nullcontext
import torch
from torch.nn import functional as F
import tiktoken
from tokenizers import Tokenizer, decoders
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = '/content/training_out'
start = "def clean_dataframe(df):\n    "
interactive = False
num_samples = 3
max_new_tokens = 120
min_new_tokens = 6
temperature = 0.4
top_k = 20
repetition_penalty = 1.15
no_repeat_ngram_size = 3 # 3-gram blocking prevents repeating `\n    df['sma'] =`
stop_at_next_function = True
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
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
    model = torch.compile(model)

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

# Handle prompt input source
if interactive:
    user_input = input("Enter unfinished Pandas function (use \\n for newlines): ")
    if user_input.strip():
        start = user_input
elif start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# Sanitize prompt whitespace and strip trailing newlines added by %%writefile
start = start.replace('\xa0', ' ').replace('\r\n', '\n')
start = start.replace('\\n', '\n').replace('\\t', '    ')
start = start.rstrip('\n')

if start.rstrip().endswith(':'):
    start = start.rstrip() + '\n    '

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

os.makedirs(out_dir, exist_ok=True)
output_file_path = os.path.join(out_dir, 'generated_sample.py')

eof_id = tokenizer.token_to_id("[EOF]") if 'tokenizer' in globals() and tokenizer else None
block_size = model.config.block_size

# Identify pure newline/indent tokens so repetition penalty never breaks indentation
indent_token_ids = set()
if 'tokenizer' in globals() and tokenizer:
    for indent_str in ["\n", "\n    ", "\n        ", "\n            ", "    "]:
        for tid in encode(indent_str):
            indent_token_ids.add(tid)

print(f"\nCompleting function from prompt:\n----------------------------------------\n{start}\n----------------------------------------\n")


def generate_code(
    model,
    idx,
    max_new_tokens,
    min_new_tokens=6,
    temperature=0.4,
    top_k=20,
    rep_penalty=1.15,
    ngram_size=3,
    eof_token_id=None,
):
    for step in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # 1. Block [EOF] for the first `min_new_tokens` steps
        if eof_token_id is not None and step < min_new_tokens:
            logits[0, eof_token_id] = -float('Inf')

        seq = idx[0].tolist()

        # 2. Block exact repeated 3-grams across full sequence (including prompt)
        if ngram_size > 0 and len(seq) >= ngram_size:
            prefix = tuple(seq[-(ngram_size - 1):])
            for i in range(len(seq) - ngram_size + 1):
                if tuple(seq[i : i + ngram_size - 1]) == prefix:
                    logits[0, seq[i + ngram_size - 1]] = -float('Inf')

        # 3. Apply repetition penalty across full sequence (excluding indentation tokens)
        if rep_penalty > 1.0:
            for token_id in set(seq):
                if token_id in indent_token_ids:
                    continue
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= rep_penalty
                else:
                    logits[0, token_id] /= rep_penalty

        # 4. Temperature & Top-K sampling
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if eof_token_id is not None and idx_next.item() == eof_token_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def clean_completed_function(prompt_text, full_text):
    if not full_text:
        return prompt_text
    generated_only = full_text[len(prompt_text):]
    stop_markers = ['\ndef ', '\nclass ', '\nif __name__', '\n# %%', '\n@']
    cut_idx = len(generated_only)
    for marker in stop_markers:
        pos = generated_only.find(marker)
        if pos != -1 and pos < cut_idx:
            cut_idx = pos

    trimmed = prompt_text + generated_only[:cut_idx]
    lines = trimmed.split('\n')
    out_lines = []
    seen_lines = set()
    seen_lhs_targets = set()
    prompt_line_count = prompt_text.count('\n')

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Record completed lines (including the completed prompt line at i == prompt_line_count)
        if i > prompt_line_count and stripped:
            # Skip exact duplicate lines
            if stripped in seen_lines:
                continue
            # Skip redundant re-assignments to the exact same DataFrame column (e.g. `df['sma'] = ...`)
            if '=' in stripped and not stripped.startswith(('if ', 'elif ', 'while ', 'return ')):
                lhs = stripped.split('=')[0].strip()
                if lhs.startswith('df[') and lhs in seen_lhs_targets:
                    continue

        if stripped:
            seen_lines.add(stripped)
            if '=' in stripped:
                lhs = stripped.split('=')[0].strip()
                if lhs.startswith('df['):
                    seen_lhs_targets.add(lhs)

        out_lines.append(line)

        # Stop cleanly after a top-level (4-space) return statement
        if i >= prompt_line_count and (line.startswith('    return ') or stripped == 'return'):
            if line.count('(') == line.count(')') and line.count('[') == line.count(']'):
                break

    result = '\n'.join(out_lines).rstrip()
    return result if result else full_text


with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = generate_code(
                model,
                x,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                rep_penalty=repetition_penalty,
                ngram_size=no_repeat_ngram_size,
                eof_token_id=eof_id,
            )
            tokens_list = y[0].tolist()
            raw_code = decode(tokens_list)

            if "[EOF]" in raw_code:
                raw_code = raw_code.split("[EOF]")[0].rstrip()

            if stop_at_next_function:
                raw_code = clean_completed_function(start, raw_code)

            print(f"# ==================== COMPLETED FUNCTION {k+1} ====================")
            print(raw_code)
            print('====================================================================')

            with open(output_file_path, 'a' if k > 0 else 'w', encoding='utf-8') as f:
                f.write(f"# {'='*20} COMPLETED FUNCTION {k+1} {'='*20}\n{raw_code}\n\n")

print(f"\nSuccessfully saved completed function to: {output_file_path}")