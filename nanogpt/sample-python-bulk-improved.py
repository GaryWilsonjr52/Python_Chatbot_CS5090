"""
Sample raw text from a trained model using a custom prompt or an Excel sheet of prompts.
If --excel_path is provided, reads prompts from Column 1 and writes 6 samples into Column 3.
"""
import os
import re
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
swap_variable_associations = True # Canonicalize non-standard/swapped variable names during inference
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


def build_variable_swap_map(prompt_text):
    """
    Infers the true semantic associations of parameters and local variables from how
    they are used in the prompt (e.g., when a DataFrame is called `col_name`, `dict`,
    `xylophone_matrix`, or `_q`, and a column parameter is called `df` or `_z`) and
    returns (forward_map, reverse_map) for simultaneous variable canonicalization.
    """
    forward_map = {}

    # 1. Analyze function signature parameters and their usage in the body
    sig_match = re.search(r'def\s+[a-zA-Z_]\w*\s*\(([^)]*)\)\s*:', prompt_text)
    if sig_match:
        raw_params = [p.strip() for p in sig_match.group(1).split(',') if p.strip()]
        params = []
        defaults = {}
        for p in raw_params:
            if '=' in p:
                k, v = [x.strip() for x in p.split('=', 1)]
                params.append(k)
                defaults[k] = v
            else:
                params.append(p)

        body = prompt_text[sig_match.end():]

        # Preserve multi-DataFrame signatures like (df1, df2) or (left_df, right_df)
        is_multi_df = len(params) >= 2 and all(
            p in ('df1', 'df2', 'left_df', 'right_df', 'left', 'right') for p in params[:2]
        )

        if not is_multi_df and params:
            df_methods = (
                r'\.(?:dropna|sort_values|replace|groupby|merge|append|flatten|'
                r'select_columns|drop_outliers|withColumn|iterrows|rolling|clip|'
                r'quantile|shift|isin|loc|iloc|assign|to_numpy|reset_index)\b'
            )

            detected_df = None
            detected_col = None
            detected_window = None

            # Detect DataFrame parameter via DataFrame method calls or subscripting
            for p in params:
                if re.search(rf'\b{re.escape(p)}{df_methods}', body) or re.search(rf'\b{re.escape(p)}\s*\[', body):
                    detected_df = p
                    break
            if detected_df is None and len(params) == 1:
                detected_df = params[0]

            # Detect Column parameter via string default, subset=[p], by=p, or df[p] indexing
            for p in params:
                if p == detected_df:
                    continue
                def_val = defaults.get(p, '')
                if (
                    def_val.startswith(("'", '"'))
                    or re.search(rf'subset\s*=\s*\[\s*{re.escape(p)}\s*\]', body)
                    or re.search(rf'\bby\s*=\s*{re.escape(p)}\b', body)
                    or (detected_df and re.search(rf'\b{re.escape(detected_df)}\s*\[\s*{re.escape(p)}\s*\]', body))
                ):
                    detected_col = p
                    break

            # Detect rolling window parameter via integer default in a .rolling() context
            if '.rolling(' in body:
                for p in params:
                    if p in (detected_df, detected_col):
                        continue
                    def_val = defaults.get(p, '')
                    if def_val.isdigit() or re.search(r'rolling\s*\(\s*window\s*=\s*$', prompt_text.strip()):
                        detected_window = p
                        break

            proposed = {}
            if detected_df and detected_df != 'df':
                proposed[detected_df] = 'df'
            if detected_col and detected_col != 'col':
                proposed[detected_col] = 'col'
            if detected_window and detected_window != 'window':
                proposed[detected_window] = 'window'

            # Ensure strict bijection if a target canonical name already exists in the prompt
            existing_identifiers = set(re.findall(r'\b[a-zA-Z_]\w*\b', prompt_text))
            for src, dst in list(proposed.items()):
                if dst in existing_identifiers and dst not in proposed:
                    # Only map if `dst` is another parameter or swap reciprocally
                    if dst in params:
                        proposed[dst] = src
                    else:
                        del proposed[src]

            forward_map.update(proposed)

    # 2. Detect inverted quantile variable associations (e.g., low_val = quantile(0.95), high_val = quantile(0.05))
    q_matches = re.findall(r'\b([a-zA-Z_]\w*)\s*=\s*[^\n]*\.quantile\(\s*(0?\.\d+)\s*\)', prompt_text)
    if len(q_matches) == 2:
        (v1, q1), (v2, q2) = q_matches[0], q_matches[1]
        q1_f, q2_f = float(q1), float(q2)
        if ('low' in v1.lower() and 'high' in v2.lower() and q1_f > 0.5 and q2_f < 0.5) or (
            'high' in v1.lower() and 'low' in v2.lower() and q1_f < 0.5 and q2_f > 0.5
        ):
            forward_map[v1] = v2
            forward_map[v2] = v1

    reverse_map = {v: k for k, v in forward_map.items()}
    return forward_map, reverse_map


def apply_variable_map(text, var_map):
    """
    Simultaneously swaps variable identifiers in `text` using `var_map`, preserving
    string literals, comments, attribute/method names (`.attr`), and call-site keyword
    argument names (`rolling(window=...)`), while swapping parameter declarations inside
    `def ...(...):` headers.
    """
    if not var_map:
        return text

    sig_match = re.search(r'def\s+[a-zA-Z_]\w*\s*\(([^)]*)\)\s*:', text)
    sig_span = sig_match.span(1) if sig_match else (-1, -1)

    pattern = re.compile(
        r'''('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|#[^\n]*|\b(?:'''
        + '|'.join(re.escape(k) for k in sorted(var_map.keys(), key=len, reverse=True))
        + r''')\b)'''
    )

    def replacer(match):
        token = match.group(0)
        if token not in var_map:
            return token
        start, end = match.span()
        # Preserve attribute and method names preceded by '.'
        if start > 0 and text[start - 1] == '.':
            return token
        # Preserve function name immediately following 'def '
        if start >= 4 and text[start - 4 : start] == 'def ':
            return token
        # Outside the `def (...)` parameter list, preserve call-site keyword argument names (e.g. `window=`)
        if not (sig_span[0] <= start and end <= sig_span[1]):
            rest = text[end:].lstrip(' \t')
            if rest.startswith('=') and not rest.startswith('=='):
                prefix_line = text[:start].split('\n')[-1]
                if prefix_line.count('(') > prefix_line.count(')'):
                    return token
        return var_map[token]

    return pattern.sub(replacer, text)


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
    prompt_len = idx.size(1)
    prompt_seq = idx[0, :prompt_len].tolist()
    prompt_token_ids = set(prompt_seq)

    # Record n-grams from the input prompt so multi-token variable names (e.g., `xylophone_matrix`)
    # are never blocked by the 3-gram blocker when referenced in the completion
    prompt_ngrams = set()
    if ngram_size > 0 and len(prompt_seq) >= ngram_size:
        for i in range(len(prompt_seq) - ngram_size + 1):
            prompt_ngrams.add(tuple(prompt_seq[i : i + ngram_size]))

    for step in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # 1. Block [EOF] for the first `min_new_tokens` steps
        if eof_token_id is not None and step < min_new_tokens:
            logits[0, eof_token_id] = -float('Inf')

        # Evaluate ONLY newly generated tokens (excluding the input prompt)
        gen_seq = idx[0, prompt_len:].tolist()

        # 2. Block repeated 3-grams across generated tokens (exempting prompt identifier n-grams)
        if ngram_size > 0 and len(gen_seq) >= ngram_size:
            prefix = tuple(gen_seq[-(ngram_size - 1) :])
            for i in range(len(gen_seq) - ngram_size + 1):
                cand_ngram = tuple(gen_seq[i : i + ngram_size])
                if cand_ngram[:-1] == prefix and cand_ngram not in prompt_ngrams:
                    logits[0, cand_ngram[-1]] = -float('Inf')

        # 3. Apply repetition penalty ONLY to newly generated tokens (excluding prompt & indentation tokens)
        if rep_penalty > 1.0 and gen_seq:
            for token_id in set(gen_seq):
                if token_id in indent_token_ids or token_id in prompt_token_ids:
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

    # Swap non-standard or inverted variable associations into canonical form before inference
    if swap_variable_associations:
        forward_map, reverse_map = build_variable_swap_map(clean_p)
        model_prompt = apply_variable_map(clean_p, forward_map)
    else:
        forward_map, reverse_map = {}, {}
        model_prompt = clean_p

    start_ids = encode(model_prompt)
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
                    raw_code = clean_completed_function(model_prompt, raw_code)

                # Restore the original prompt variable names in the completed function
                if reverse_map:
                    raw_code = apply_variable_map(raw_code, reverse_map)

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