# Python Chatbot — CS5090

An experimental GPT-based Python code-completion project focused on Pandas and data-analysis workflows. The repository uses a modified [nanoGPT](https://github.com/karpathy/nanoGPT) training loop, prepares a custom Python-function corpus, trains a small GPT model, and completes partial Python functions from text or spreadsheet prompts.

The primary experiment is the **one-head model** described in [`oneHead.ipynb`](oneHead.ipynb). The project is notebook-driven and was authored around Google Colab with a CUDA GPU. It is not currently packaged as a conventional installable chatbot application.

## What the model does

The model is trained from scratch to continue Python functions, especially functions that use Pandas `DataFrame` and `Series` operations. Example prompt types in the notebooks include:

- Calculating a moving average from a `close` column.
- Cleaning and sorting a DataFrame.
- Producing rolling-window breakout signals.

The training process learns next-token prediction over extracted Python functions. The sampling scripts then complete an unfinished function or process a set of prompts in an Excel workbook. Generated code should be reviewed and tested before use; generation is not a guarantee of correctness.

## Repository layout

```text
Python_Chatbot_CS5090/
├── nanoGPT/                       # GPT architecture, training loop, sampling scripts
│   ├── model.py                   # GPT model definition
│   ├── train.py                   # Training and validation loop
│   ├── sample-python.py           # Single-prompt Python completion
│   ├── sample-python-bulk.py      # Excel prompt batch generation
│   ├── sample.py                  # Generic nanoGPT sampler (used by smoke test)
│   ├── configurator.py             # Command-line configuration overrides
│   └── data/shakespeare_char/      # Small nanoGPT example dataset for smoke testing
├── training_data/
│   ├── prepare.py                 # Builds the custom corpus, tokenizer, and train/val files
│   ├── files/                     # Source repositories scanned by prepare.py
│   ├── Prompts/                   # Excel prompt workbooks used for batch sampling
│   ├── train.bin                  # Generated training token IDs (after preparation)
│   ├── val.bin                    # Generated validation token IDs (after preparation)
│   ├── meta.pkl                   # Vocabulary metadata (after preparation)
│   └── tokenizer.json             # Custom BPE tokenizer (after preparation)
├── oneHead.ipynb                  # Main one-head training and prompt experiment
├── prerefined.ipynb               # Earlier six-head training experiment
├── sampler.ipynb                  # Single-prompt and Excel batch sampling
├── test.ipynb                     # nanoGPT/Shakespeare smoke test
├── old/                           # Older project material; not used by the documented workflow
└── .gitattributes                 # Git LFS rules for .pt and .bin files
```

`train.bin`, `val.bin`, `meta.pkl`, and `tokenizer.json` are the outputs of `training_data/prepare.py`; they are not the raw source corpus. The source code repositories are placed under `training_data/files/`. The prompt workbooks referenced by `sampler.ipynb` are:

- `training_data/Prompts/pandas_SWE_prompts_20.xlsx`
- `training_data/Prompts/pandas_exposure_prompts_20.xlsx`

The Shakespeare data under `nanoGPT/data/shakespeare_char/` is separate from the Pandas-focused training corpus and is used only by `test.ipynb` for a basic nanoGPT smoke test.

## Notebook guide

| Notebook | Purpose | Main outputs / data used |
|---|---|---|
| [`oneHead.ipynb`](oneHead.ipynb) | Main experiment. Installs the preparation dependencies, clones the Pandas and quantitative-finance source repositories, runs `training_data/prepare.py`, trains a six-layer GPT with **one attention head**, writes several example function prompts, and samples completions. | Reads source files from `training_data/files/`; creates the corpus files in `training_data/`; writes the model checkpoint to `/content/training_out/ckpt.pt` in the Colab workflow; writes a sample completion to the output directory. |
| [`prerefined.ipynb`](prerefined.ipynb) | Earlier comparison experiment using the same data-preparation workflow, but with **six attention heads** instead of one. Its initial training cell specifies 6,000 iterations and demonstrates a moving-average prompt. The resume cell later in the notebook changes the iteration/decay target to 3,000, so that cell should be reconciled with the checkpoint's saved iteration before use. | Uses the same `training_data/` pipeline and output-directory convention. Treat this as an earlier model configuration rather than the primary one-head recipe. |
| [`sampler.ipynb`](sampler.ipynb) | Uses an existing checkpoint; it does not prepare the corpus or train a model. It samples a hand-written moving-average prompt and runs batch completion against the SWE and exposure Excel prompt sets. | Reads `training_data/tokenizer.json`, `training_out/ckpt.pt`, and the two workbooks in `training_data/Prompts/`. `sample-python-bulk.py` writes generated samples to column 3 of the workbook (by default, it saves back to the same workbook). |
| [`test.ipynb`](test.ipynb) | Smoke-tests the underlying nanoGPT trainer using the built-in Shakespeare character dataset. It prepares that dataset, trains a small four-layer/four-head model for 2,000 iterations, then samples Shakespeare-like text. | Uses `nanoGPT/data/shakespeare_char/`; writes a separate checkpoint under `nanoGPT/shakespeare_char_out/`. It does **not** test the Pandas-specific model or its training corpus. |

## Training data and preparation

### 1. Local code sources

`training_data/prepare.py` recursively scans `training_data/files/` for `.py` and `.ipynb` files. For notebooks, it combines the source code from code cells. It parses Python using the `ast` module and retains standalone functions that:

- Are between 4 and 250 lines long.
- Contain Pandas-related indicators such as `pd.`, `DataFrame`, `Series`, `read_csv`, `groupby`, `rolling`, `dropna`, `merge`, and related operations.

The notebooks clone these source repositories into `training_data/files/`:

**Pandas and data-science examples**

- [leetlab11/Introduction-to-Pandas](https://github.com/leetlab11/Introduction-to-Pandas)
- [Aayush-kotwani/Pandas-masterclass](https://github.com/Aayush-kotwani/Pandas-masterclass)
- [codebasics/py](https://github.com/codebasics/py)
- [guipsamora/pandas_exercises](https://github.com/guipsamora/pandas_exercises)
- [jvns/pandas-cookbook](https://github.com/jvns/pandas-cookbook)

**Quantitative-finance and DataFrame-heavy examples**

- [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)
- [twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta)
- [polakowo/vectorbt](https://github.com/polakowo/vectorbt)
- [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance)
- [quantopian/pyfolio](https://github.com/quantopian/pyfolio)
- [quantopian/alphalens](https://github.com/quantopian/alphalens)
- [pmorissette/bt](https://github.com/pmorissette/bt)
- [bukosabino/ta](https://github.com/bukosabino/ta)

These repositories are cloned at their current default branch by the notebook commands, not pinned to fixed commit IDs. For strict reproducibility, record the commit SHA of each source repository when creating a corpus. Follow the license terms of every source dataset and repository.

### 2. Hugging Face code corpus

In addition to the local repositories, `prepare.py` streams the training split of [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean) through the Hugging Face `datasets` library. It extracts matching Pandas-related functions until it reaches a target of 400,000,000 characters (the script comments estimate this as roughly 100 million tokens). The script catches streaming errors and continues, so check its final summary to make sure the remote data was actually read and the resulting corpus is large enough for your run.

### 3. Generated dataset artifacts

Run `prepare.py` from the repository root:

```bash
python training_data/prepare.py
```

The script:

1. Reads matching functions from local Python files and notebook code cells under `training_data/files/`.
2. Streams additional Python code from `codeparrot/codeparrot-clean` and extracts matching functions.
3. Shuffles the extracted function chunks using `random.seed(42)`.
4. Trains a custom byte-level BPE tokenizer with a vocabulary size of 8,192. Tokenizer training uses up to the first 100,000 shuffled function chunks and includes `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`, and `[EOF]` special tokens.
5. Splits the shuffled function chunks into 90% training and 10% validation partitions.
6. Appends an `[EOF]` separator to each function, tokenizes in batches, and writes the token IDs as `uint16` binary data.

Expected outputs in `training_data/`:

| File | Role |
|---|---|
| `train.bin` | Token IDs used to train the model. |
| `val.bin` | Held-out token IDs used to calculate validation loss. |
| `tokenizer.json` | Custom BPE tokenizer required by the Python sampling scripts. |
| `meta.pkl` | Vocabulary size and token-to-ID / ID-to-token mappings used by the nanoGPT trainer. |

Do not mix these generated files with a different tokenizer or checkpoint. The training data, tokenizer, and checkpoint must come from the same preparation/training run.

## Reproducing the primary one-head model

The notebook uses Google Colab paths such as `/content/Python_Chatbot_CS5090` and a Colab T4 GPU. The commands below preserve the notebook's primary settings while using a local repository path. A CUDA-capable GPU is recommended; the notebook enables `torch.compile` and FP16.

### 1. Clone the repository

The repository's `.gitattributes` marks `*.pt` and `*.bin` for Git LFS. Install Git LFS before cloning if you need any LFS-managed artifacts from the repository:

```bash
git lfs install
git clone https://github.com/GaryWilsonjr52/Python_Chatbot_CS5090.git
cd Python_Chatbot_CS5090
```

### 2. Install Python dependencies

Install a PyTorch build compatible with your CUDA driver and GPU by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install the project packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy datasets tokenizers tqdm tiktoken openpyxl
```

`transformers` and `wandb` are optional for the documented one-head workflow; the Shakespeare test notebook's setup cell also installs them. They are not required by `training_data/prepare.py` or the custom sampling scripts.

### 3. Populate `training_data/files/`

The preparation script needs source code to scan. Run the following once from the repository root. The `--depth 1` clones match the notebook's approach:

```bash
mkdir -p training_data/files
cd training_data/files

git clone --depth 1 https://github.com/leetlab11/Introduction-to-Pandas
git clone --depth 1 https://github.com/Aayush-kotwani/Pandas-masterclass
git clone --depth 1 https://github.com/codebasics/py
git clone --depth 1 https://github.com/guipsamora/pandas_exercises
git clone --depth 1 https://github.com/jvns/pandas-cookbook

git clone --depth 1 https://github.com/stefan-jansen/zipline-reloaded
git clone --depth 1 https://github.com/twopirllc/pandas-ta
git clone --depth 1 https://github.com/polakowo/vectorbt
git clone --depth 1 https://github.com/ranaroussi/yfinance
git clone --depth 1 https://github.com/quantopian/pyfolio
git clone --depth 1 https://github.com/quantopian/alphalens
git clone --depth 1 https://github.com/pmorissette/bt
git clone --depth 1 https://github.com/bukosabino/ta

cd ../..
```

Skip any clone whose directory is already present. Re-running `git clone` in an existing destination will fail.

### 4. Prepare the Pandas corpus

From the repository root:

```bash
python training_data/prepare.py
```

Confirm that the script reports saved `train.bin`, `val.bin`, `meta.pkl`, and `tokenizer.json` in `training_data/`. If the CodeParrot stream fails, the command can still finish using only the local code it successfully extracted; inspect the printed function and token counts before proceeding.

### 5. Train the one-head model

Run `train.py` from the `nanoGPT/` directory. The absolute dataset and output paths below avoid relying on Colab's `/content` directory:

```bash
REPO_ROOT="$(pwd)"
cd "$REPO_ROOT/nanoGPT"

python train.py \
  --out_dir="$REPO_ROOT/training_out" \
  --dataset="$REPO_ROOT/training_data" \
  --always_save_checkpoint=False \
  --log_interval=10 \
  --eval_interval=200 \
  --block_size=512 \
  --batch_size=16 \
  --gradient_accumulation_steps=4 \
  --n_layer=6 \
  --n_head=1 \
  --n_embd=384 \
  --dropout=0.1 \
  --learning_rate=1e-3 \
  --min_lr=1e-4 \
  --lr_decay_iters=6000 \
  --warmup_iters=200 \
  --device=cuda \
  --compile=True \
  --dtype=float16 \
  --max_iters=6000
```

This reproduces the principal settings in `oneHead.ipynb`:

| Parameter | Value |
|---|---:|
| Transformer layers | 6 |
| Attention heads per layer | 1 |
| Embedding width | 384 |
| Context / block size | 512 tokens |
| Micro-batch size | 16 |
| Gradient accumulation | 4 steps |
| Effective tokens per iteration | 32,768 |
| Dropout | 0.1 |
| Peak / minimum learning rate | `1e-3` / `1e-4` |
| Warmup / decay / maximum iterations | 200 / 6,000 / 6,000 |
| Device / precision | CUDA / FP16 |

`train.py` evaluates the training and validation splits every 200 iterations. With `--always_save_checkpoint=False`, it saves `training_out/ckpt.pt` when validation loss improves; the training loop does not write a checkpoint at iteration zero, so the first checkpoint is produced at a later evaluation step. That checkpoint is required by the sampling scripts.

**Resume an interrupted run:** rerun the same command with `--init_from=resume`. Keep the same output directory and model dimensions. The configured `--max_iters=6000` is the total target iteration number, not an additional 6,000 iterations; increase `--max_iters` and, as appropriate, `--lr_decay_iters` if you intend to train beyond the original target.

### 6. Generate a completion from one prompt

The sampling scripts currently hard-code the Colab tokenizer location:

```python
'/content/Python_Chatbot_CS5090/training_data/tokenizer.json'
```

For a local checkout, update the `tokenizer_path` assignment in **both** `nanoGPT/sample-python.py` and `nanoGPT/sample-python-bulk.py` to resolve the tokenizer relative to the script:

```python
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tokenizer_path = os.path.join(repo_root, "training_data", "tokenizer.json")
```

Then, from `nanoGPT/`, run a prompt directly. Re-establish `REPO_ROOT` in this shell if you opened a new terminal:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/nanoGPT"

python sample-python.py \
  --init_from=resume \
  --out_dir="$REPO_ROOT/training_out" \
  --start="def calculate_moving_average(df, window=20):\n    df = df.dropna(subset=['close'])\n    " \
  --num_samples=6 \
  --temperature=0.1 \
  --top_k=20
```

Or put the unfinished function in a text file and pass `--start="FILE:/absolute/path/to/prompt.py"`. The single-prompt sampler writes its generated result to `training_out/generated_sample.py`.

Always use the matching `training_data/tokenizer.json` created by the corpus preparation step. Although the sampler contains a fallback to GPT-2's tokenizer if the custom tokenizer is missing, GPT-2 token IDs are not interchangeable with the custom vocabulary used to train this model.

### 7. Run batch prompt evaluation from Excel

The notebook's batch examples use two workbooks under `training_data/Prompts/`. From `nanoGPT/`, for example (set `REPO_ROOT` again if this is a new shell):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/nanoGPT"

python sample-python-bulk.py \
  --out_dir="$REPO_ROOT/training_out" \
  --device=cuda \
  --excel_path="$REPO_ROOT/training_data/Prompts/pandas_SWE_prompts_20.xlsx" \
  --excel_out_path="$REPO_ROOT/training_out/pandas_SWE_results.xlsx" \
  --num_samples=6 \
  --temperature=0.1 \
  --top_k=20
```

For the exposure prompt set, change `--excel_path` to `pandas_exposure_prompts_20.xlsx` and use a different `--excel_out_path`. The script reads prompts from column 1 and writes the generated completions to column 3. If `--excel_out_path` is omitted, it saves to the input workbook path and overwrites that workbook.

## Tests and validation

There are two kinds of checks in the notebooks, and they serve different purposes:

1. **nanoGPT smoke test — `test.ipynb`.** From `nanoGPT/`, run:

   ```bash
   python data/shakespeare_char/prepare.py
   python train.py \
     --out_dir=shakespeare_char_out \
     --dataset=shakespeare_char \
     --device=cuda \
     --compile=False \
     --eval_iters=20 \
     --log_interval=10 \
     --block_size=64 \
     --batch_size=12 \
     --n_layer=4 \
     --n_head=4 \
     --n_embd=128 \
     --max_iters=2000
   python sample.py --out_dir=shakespeare_char_out --device=cuda
   ```

   This confirms that the nanoGPT data loader, training loop, checkpoint saving, and generic sampler can run. The notebook's saved output shows a successful 2,000-iteration run followed by text generation. It does not validate Pandas code generation.

2. **Pandas completion checks — `oneHead.ipynb` and `sampler.ipynb`.** Generate completions for the moving-average, DataFrame-cleaning, breakout-signal, SWE, and exposure prompts. Review the outputs for correct Pandas operations, sensible handling of missing values, and consistent indentation. For a syntax-only check of a generated Python file, use `python -m py_compile path/to/generated_sample.py`.

There is no documented `pytest`/`unittest` suite or automated correctness metric for the Pandas-generated code in the inspected workflow. The spreadsheet sampling notebooks are manual/qualitative evaluations, not a scored benchmark. The `test.ipynb` file is a Shakespeare baseline experiment, not a test suite for the target model.

## Reproducibility notes and known notebook assumptions

- **Colab paths:** the notebooks use `/content/Python_Chatbot_CS5090` and `/content/training_out`. Substitute local absolute paths when running outside Colab.
- **Tokenizer path:** both custom sampler scripts hard-code the Colab location for `tokenizer.json`; update that path for local execution as shown above.
- **Checkpoint filename:** `sampler.ipynb` has a markdown note saying `cpt.pt`, but the actual sampling scripts load `ckpt.pt`. Use `training_out/ckpt.pt`.
- **GPU requirements:** the primary notebook uses `--device=cuda`, `--dtype=float16`, and `--compile=True`. For a CPU smoke test, use the separate Shakespeare workflow with `--device=cpu --compile=False`; the full Pandas training recipe is intended for a CUDA GPU.
- **Earlier six-head resume settings:** `prerefined.ipynb` initially configures 6,000 iterations, but its later resume cell sets both `--max_iters` and `--lr_decay_iters` to 3,000. nanoGPT treats `max_iters` as an absolute iteration target, so check the saved checkpoint's `iter_num` and correct those values before resuming that experiment.
- **External data is mutable:** because the source repositories are cloned from their current default branches and the Hugging Face dataset is streamed at run time, a future corpus build may not be byte-for-byte identical. Pin source commit SHAs and record package versions for a controlled experiment.
- **Incomplete corpus warning:** `prepare.py` catches CodeParrot streaming exceptions and can continue with local files only. Verify the final extracted-function/token counts and the saved output files.
- **Git LFS:** `.gitattributes` configures `*.pt` and `*.bin` for Git LFS. Install Git LFS before cloning or pull LFS objects when required; otherwise large files may be present only as pointer files.

## Acknowledgements

This project adapts the nanoGPT training code and uses public Pandas/quantitative-finance source repositories plus the Hugging Face CodeParrot dataset. Refer to the licenses and terms of those upstream projects and datasets before redistributing derived training artifacts.
