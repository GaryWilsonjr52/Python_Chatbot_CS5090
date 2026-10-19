"""
Sample raw text from a trained model using a custom prompt,
and save it directly to out_dir/generated_sample.py
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from tokenizers import Tokenizer, decoders
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = '/content/training_out' # default out_dir, can also be overridden via command line
start = "def clean_dataframe(df):\n    " # Use --start="..." or --start="FILE:prompt.py"
interactive = False # Set --interactive=True to type/paste prompt directly when running
num_samples = 1 # default to 1 sample based on prompt
max_new_tokens = 512 # gives functions room to complete
temperature = 0.4 # lower temperature (0.2 - 0.5) is more accurate for Pandas code completion
top_k = 50 # retain top_k most likely tokens
seed = 1337
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
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

# Handle prompt input source (Interactive, FILE:, or --start CLI string)
if interactive:
    user_input = input("Enter unfinished Pandas function (use \\n for newlines): ")
    if user_input.strip():
        start = user_input
elif start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# Convert literal '\n' and '\t' passed from command line into real newlines/tabs
start = start.replace('\\n', '\n').replace('\\t', '    ')

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Prepare output file path
os.makedirs(out_dir, exist_ok=True)
output_file_path = os.path.join(out_dir, 'generated_sample.py')

# Get the token ID for [EOF] from the loaded custom tokenizer
eof_id = tokenizer.token_to_id("[EOF]") if 'tokenizer' in globals() and tokenizer else None

print(f"\nCompleting function from prompt:\n----------------------------------------\n{start}\n----------------------------------------\n")

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            tokens_list = y[0].tolist()

            # Truncate token sequence if [EOF] token was generated
            if eof_id is not None and eof_id in tokens_list[len(start_ids):]:
                eof_index = tokens_list.index(eof_id, len(start_ids))
                tokens_list = tokens_list[:eof_index]

            raw_code = decode(tokens_list)

            # Secondary safety strip in case "[EOF]" was decoded as literal text
            if "[EOF]" in raw_code:
                raw_code = raw_code.split("[EOF]")[0].rstrip()

            print(f"# ==================== COMPLETED FUNCTION {k+1} ====================")
            print(raw_code)
            print('====================================================================')
            
            # Write out to file
            with open(output_file_path, 'a' if k > 0 else 'w', encoding='utf-8') as f:
                f.write(f"# {'='*20} COMPLETED FUNCTION {k+1} {'='*20}\n{raw_code}\n\n")

print(f"\nSuccessfully saved completed function to: {output_file_path}")