"""
Sample raw text from a trained model using a custom prompt or an Excel sheet of prompts.
If --excel_path is provided, reads prompts from Column 1 and writes 6 samples into Column 3.
"""
import os
from contextlib import nullcontext
import torch
from torch.nn import functional as F
import tiktoken
import openpyxl
from openpyxl.styles import Alignment, Font
from tokenizers import Tokenizer, decoders
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = '/content/training_out'
start = "def clean_dataframe(df):\n    "
excel_path = "" # Pass --excel_path="pandas_standard_prompts_20.xlsx" to batch-process an Excel file
excel_out_path = "" # Optional custom output path; defaults to updating excel_path in-place
interactive = False
num_samples = 6 # Default to 6 samples per prompt
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

eof_id = tokenizer.token_to_id("[EOF]") if 'tokenizer' in globals() and tokenizer else None
block_size = model.config.block_size

# Identify pure newline/indent tokens so repetition penalty never breaks indentation
indent_token_ids = set()
if 'tokenizer' in globals() and tokenizer:
    for indent_str in ["\n", "\n    ", "\n        ", "\n            ", "    "]:
        for tid in encode(indent_str):
            indent_token_ids.add(tid)


def sanitize_prompt(raw_prompt):
    """Normalizes whitespace and strips trailing spaces/newlines that confuse BPE."""
    s = str(raw_prompt).replace('\xa0', ' ').replace('\r\n', '\n')
    s = s.replace('\\n', '\n').replace('\\t', '    ')
    s = s.rstrip('\n')
    if s.rstrip().endswith(':'):
        s = s.rstrip() + '\n    '
    elif not s.split('\n')[-1].isspace():
        s = s.rstrip(' ')
    return s


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

        if i > prompt_line_count and stripped:
            if stripped in seen_lines:
                continue
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


def generate_n_samples_for_prompt(prompt_str, n_samples):
    """Generates `n_samples` completions for a single sanitized prompt string."""
    clean_p = sanitize_prompt(prompt_str)
    start_ids = encode(clean_p)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    samples_out = []
    with torch.no_grad():
        with ctx:
            for k in range(n_samples):
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
                    raw_code = clean_completed_function(clean_p, raw_code)

                samples_out.append(f"# --- SAMPLE {k+1} ---\n{raw_code}")
    return samples_out


# ==============================================================================
# MODE 1: Batch Excel Processing (--excel_path="...")
# ==============================================================================
if excel_path:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    print(f"\nLoading Excel workbook: {excel_path}")
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    # Ensure Column 3 header exists
    if not ws.cell(row=1, column=3).value:
        ws.cell(row=1, column=3, value=f"Generated Samples ({num_samples})")

    # Automatically detect if Row 1 is a header or already a `def ` prompt
    first_cell_val = str(ws.cell(row=1, column=1).value or "").strip()
    start_row = 1 if first_cell_val.startswith("def ") else 2

    total_rows = ws.max_row
    processed_count = 0

    for row_idx in range(start_row, total_rows + 1):
        cell_prompt = ws.cell(row=row_idx, column=1).value
        if not cell_prompt or not str(cell_prompt).strip():
            continue

        print(f"Processing Row {row_idx}/{total_rows}...")
        samples_list = generate_n_samples_for_prompt(cell_prompt, num_samples)
        combined_samples_text = "\n\n".join(samples_list)

        out_cell = ws.cell(row=row_idx, column=3, value=combined_samples_text)
        out_cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        out_cell.font = Font(name="Consolas", size=10)
        processed_count += 1

    save_target = excel_out_path if excel_out_path else excel_path
    wb.save(save_target)
    print(f"\nSuccessfully processed {processed_count} prompts and wrote {num_samples} samples each to Column 3 in: {save_target}")

# ==============================================================================
# MODE 2: Single Prompt Processing (--start="..." or FILE:...)
# ==============================================================================
else:
    if interactive:
        user_input = input("Enter unfinished Pandas function (use \\n for newlines): ")
        if user_input.strip():
            start = user_input
    elif start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    clean_start = sanitize_prompt(start)
    os.makedirs(out_dir, exist_ok=True)
    output_file_path = os.path.join(out_dir, 'generated_sample.py')

    print(f"\nCompleting function from prompt:\n----------------------------------------\n{clean_start}\n----------------------------------------\n")
    samples = generate_n_samples_for_prompt(clean_start, num_samples)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for sample_block in samples:
            print(sample_block)
            print('====================================================================')
            f.write(sample_block + "\n\n")

    print(f"\nSuccessfully saved completed functions to: {output_file_path}")