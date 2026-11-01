# Python Chatbot — CS 5090

A small GPT-style Python code language model trained from scratch with [nanoGPT](https://github.com/karpathy/nanoGPT). The project focuses on generating and completing Python code that uses Pandas and DataFrame operations. It was developed as the CS 5090 Project 1: **Build and Evaluate a Small Code Language Model**. Project discription is included as a pdf

The repository contains a six-attention-head baseline model, a one-head ablation of that model, a Google Colab/nanoGPT setup test, and a notebook for sampling model outputs against real software-engineering prompts.

> **Reproduction entry point:** Start with [`prerefined.ipynb`](prerefined.ipynb). This is the primary notebook and produces the six-head baseline. Run its cells in order; the notebook runs `training_data/prepare.py` and then `nanoGPT/train.py`. Do not use `oneHead.ipynb` as the baseline reproduction notebook: it is the one-head ablation.

## Project workflow at a glance

| Purpose | Notebook or script | What it does |
|---|---|---|
| Primary model / baseline | [`prerefined.ipynb`](prerefined.ipynb) | Builds the training data by running `training_data/prepare.py`, then trains the primary six-head model by running `nanoGPT/train.py`. |
| Ablation | [`oneHead.ipynb`](oneHead.ipynb) | Repeats the model workflow with the attention-head count changed from six to one. This tests the effect of using one attention head instead of six. |
| Sampling and model use | [`sampler.ipynb`](sampler.ipynb) | Loads the trained checkpoint and generates completions from either a 20-prompt Excel workbook or a `prompt.py` file created/used in the sampler workflow. |
| nanoGPT / Colab setup test | [`test.ipynb`](test.ipynb) | Checks that the underlying Google Colab code and nanoGPT training/sampling path work. It is a setup/smoke test, not an evaluation of the Pandas model. |

## Repository layout

```text
Python_Chatbot_CS5090/
├── nanoGPT/                              # nanoGPT implementation, configs, and example data
├── nanogpt/
│   └── sample-python-bulk-improved.py    # Improved inference-time bulk sampler
├── training_data/
│   ├── prepare.py                        # Dataset preparation and tokenizer training
│   ├── files/                            # Cloned source repositories
│   └── Prompts/                          # Original 20-prompt workbooks
├── data + results/
│   ├── checkpoints and loss records/    # Checkpoints, logs, and loss workbook
│   ├── training data/                    # train.bin, val.bin, meta.pkl, tokenizer.json
│   ├── pandas_SWE_prompts_20_generated.xlsx
│   ├── pandas_exposure_prompts_evaluated_20_prerefined.xlsx
│   └── pandas_exposure_prompts_20_post_fixed_sample.xlsx
├── prerefined.ipynb                      # Primary six-head baseline workflow
├── oneHead.ipynb                         # One-head ablation workflow
├── sampler.ipynb                         # Original sampling workflow
├── test.ipynb                            # nanoGPT/Colab setup smoke test
├── Project1.pdf                          # Project specification
├── Project1_report.docx                  # Project report
└── .gitattributes                        # Git LFS rules, where configured
```

The `.bin`, `.pkl`, and tokenizer files under `training_data/` are generated artifacts. They are not the original source corpus. The original local code sources are placed under `training_data/files/`.

## 1. Reproducing the six-head baseline

The intended reproduction path is to run [`prerefined.ipynb`](prerefined.ipynb) from top to bottom in Google Colab.

### Environment

1. Open `prerefined.ipynb` in Google Colab.
2. Select CPU only runtime for making the training data, as that is not gpu bound, saving your gpu hours
3. Select a GPU runtime. A GPU is required for training; taking around an hour for the training to finsh per model
4. Run the notebook cells in their existing order. The notebook contains the setup, data preparation, and model-training steps.
5. Keep the same repository version, source data, tokenizer, and configuration when reproducing the run.

### What the baseline notebook runs

The notebook orchestrates the two main Python stages:

1. **Data preparation — `training_data/prepare.py`**
   - Reads and filters local Python source files and notebook code under `training_data/files/`.
   - Streams additional code from the Hugging Face CodeParrot dataset.
   - Extracts Pandas-related Python functions.
   - Builds the tokenizer and writes the training and validation token files.
2. **Model training — `nanoGPT/train.py`**
   - Loads the prepared dataset and vocabulary metadata.
   - Trains the six-layer, six-attention-head GPT baseline.
   - Evaluates training and validation loss during training.
   - Saves the model checkpoint used by `sampler.ipynb`.

The notebook is the authoritative run sequence and configuration. The important baseline architecture settings are:

| Setting | Baseline |
|---|---:|
| Transformer layers | 6 |
| Attention heads | 6 |
| Embedding size | 384 |
| Context length / block size | 512 tokens |
| Tokenizer | Byte-pair encoding (BPE) |
| Tokenizer vocabulary | 8,192 |
| Data split | 90% training / 10% validation |
| Training script | `nanoGPT/train.py` |
| Data preparation script | `training_data/prepare.py` |

The project brief recommends fitting the tokenizer using the training split only. See [Data preparation notes](#data-preparation-notes) below for a caveat about the current preparation pipeline.

### Files produced by preparation and training

| Artifact | Location | Purpose |
|---|---|---|
| Training tokens | `data + results/training data/train.bin` | Tokenized examples used for model training. |
| Validation tokens | `data + results/training data/val.bin` | Held-out examples used for validation loss. |
| Vocabulary metadata | `data + results/training data/meta.pkl` | Token/vocabulary mappings expected by nanoGPT. |
| BPE tokenizer | `data + results/training data/tokenizer.json` | Tokenizer used by the custom Python samplers. |
| Model checkpoint | `training_out/ckpt.pt` in the Colab workflow | Trained model loaded during sampling. |

Keep the checkpoint, tokenizer, and prepared dataset from the same run together. A checkpoint trained with one tokenizer cannot safely be sampled using a different tokenizer vocabulary.

### Reproducing a comparable run

For a meaningful reproduction, record the following with the notebook run:

- Repository commit or version.
- Colab runtime type, GPU, Python version, and installed package versions.
- The source repository commit IDs used to build `training_data/files/`.
- Whether the CodeParrot stream completed, along with extracted function and token counts.
- The exact baseline configuration, training duration/iteration count, and final training/validation loss.
- The location of the checkpoint and the corresponding tokenizer/data artifacts.

The final report should include the model configuration, important data-processing decisions, and a training/validation loss curve.

## 2. Training data and corpus preparation

### Local source repositories

The notebook setup clones the following repositories into `training_data/files/`. `prepare.py` scans the local tree for Python files and Jupyter notebooks.

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

The notebook clones these repositories from their default branches. Their contents can change over time; for strict reproducibility, record the commit SHA for each source repository. These repos were lasted pulled on **October 31st, 2026**

### Hugging Face dataset

In addition to the local repositories, `training_data/prepare.py` streams the training split of [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean) through the Hugging Face `datasets` library. The script extracts additional Pandas-related functions until it reaches a target of 400,000,000 characters (the script comments estimate this at roughly 100 million tokens). The stream can fail or be interrupted, so inspect the preparation output and record the final counts rather than assuming the full remote corpus was collected.

### How `prepare.py` constructs the dataset

The preparation script:

1. Recursively scans `training_data/files/` for `.py` and `.ipynb` files.
2. Extracts notebook code cells and parses Python source using the `ast` module.
3. Retains standalone functions between 4 and 250 lines that contain Pandas-related indicators, such as `pd.`, `DataFrame`, `Series`, `read_csv`, `groupby`, `rolling`, `dropna`, or `merge`.
4. Adds matching functions extracted from the streamed CodeParrot dataset.
5. Shuffles the collected function chunks using a fixed random seed.
6. Trains a custom byte-level BPE tokenizer with an 8,192-token vocabulary, using up to the first 100,000 shuffled function chunks and the special tokens `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`, and `[EOF]`.
7. Splits the function data into training and validation partitions (90/10).
8. Appends an end-of-function separator, tokenizes the content, and writes `train.bin` and `val.bin` as `uint16` token IDs. It also writes `meta.pkl` and `tokenizer.json`.

### Data preparation notes

The project brief asks that the data sheet document filtering, deduplication, tokenization, and splitting, and specifies that the tokenizer should be trained on the training split only.

Two points should be addressed when documenting or refining the current pipeline:

- **Tokenizer fit and split order:** the current `prepare.py` workflow appears to fit the tokenizer before making the 90/10 train/validation split. That is not the order specified in the project brief. For strict compliance, split the function chunks first, fit the tokenizer only on the training partition, and then use that tokenizer to encode both partitions.
- **Deduplication:** the preparation steps described above filter and shuffle functions, but do not describe a separate deduplication pass. State clearly in the project data sheet whether duplicates were removed. Do not claim deduplication unless it was actually performed.

Also record the source licenses and any restrictions on redistributing the resulting corpus or trained artifacts.

## 3. One-head ablation

[`oneHead.ipynb`](oneHead.ipynb) is the ablation notebook. It changes the baseline from **six attention heads to one attention head** while retaining the general model/data workflow.

The purpose of this experiment is to study what is gained by using multiple attention heads. It is not the main reproduction path.

For a valid comparison:

1. Train and save the six-head baseline by running `prerefined.ipynb`.
2. Record the predicted effect of reducing the attention-head count **before** running the ablation.
3. Use the same prepared training/validation data, tokenizer, context length, and other training settings for both runs wherever possible.
4. Run `oneHead.ipynb` and retain its configuration and checkpoint separately from the baseline.
5. Compare the final validation loss and at least one side-by-side generation using the same prompt and sampling settings.

If the ablation notebook rebuilds the corpus, preserve the baseline data artifacts or otherwise verify that both models use the same data split. Rebuilding from mutable remote repositories or a live Hugging Face stream can change the training data and weaken the comparison.

## 4. Sampling the trained model

After training the baseline or ablation, use [`sampler.ipynb`](sampler.ipynb) for inference. Sampling does not train the model or rebuild the corpus; it loads the saved checkpoint and the matching tokenizer.

The sampler supports two prompt workflows:

### Option A: Excel workbook with 20 prompts

Use a workbook containing the 20 software-engineering prompts. The repository workflow references:

- `training_data/Prompts/pandas_SWE_prompts_20.xlsx`
- `training_data/Prompts/pandas_exposure_prompts_20.xlsx`

Choose the workbook in the sampler notebook and run the batch-generation cells. The batch sampler reads the prompts and writes the generated completions to the workbook/output location configured in the notebook. Preserve an untouched copy of the original prompt sheet so that the inputs and model outputs remain traceable.

This 20-prompt workflow supports the project brief's Phase 2 evaluation. For each prompt, record one judgment:

- **Accept as-is**
- **Accept after editing**
- **Reject**

Report the count for each outcome. Select three examples for closer discussion: the best result, the worst result, and a near-miss that initially appeared correct but was not.

### Option B: `prompt.py` for an individual prompt

For an individual completion, create or use a `prompt.py` file in the sampler workflow containing the Python prompt to continue. Run the relevant cells in `sampler.ipynb` to pass that prompt through the model and inspect the generated code.

Keep the exact prompt and generated output together. Generated code should be reviewed and tested; fluent output is not evidence that the code is syntactically or logically correct.

### Sampling artifacts and paths

The custom sampling scripts use the tokenizer produced by `prepare.py` and the trained checkpoint (the Colab workflow uses `training_out/ckpt.pt`). The Python sampler scripts currently contain Colab-specific paths. When running outside Colab, update those paths to match the local checkout and checkpoint location.

The bulk sampler can write results back to the source workbook if configured to do so. Use a separate output workbook when you need to preserve the original 20 prompts.

## 5. `test.ipynb`: Google Colab and nanoGPT setup test

[`test.ipynb`](test.ipynb) is used to confirm that the underlying Google Colab setup and nanoGPT code path work. It exercises nanoGPT's example training/sampling workflow using the small Shakespeare character dataset.

This notebook is a **smoke test of the environment and nanoGPT integration**:

- It checks that the notebook can install/import the required packages.
- It exercises nanoGPT data preparation and the generic training loop.
- It confirms that a checkpoint can be produced and that the generic sampler can generate output.

It does **not** test the Pandas training corpus, the six-head model's code-generation quality, or whether generated Python functions are correct. It is also not a `pytest` or `unittest` suite.

The Pandas model is evaluated through the prompt workflows in `sampler.ipynb` and the project's Phase 2/3 evaluation records.

## 6. Improved bulk sampler

The updated sampler is located at:

```text
nanogpt/sample-python-bulk-improved.py
```

This is an inference-time improvement to the bulk generation workflow. It does not retrain the model or change checkpoint weights. The script generates six candidates per prompt and includes configurable controls for:

- Temperature (`0.4` by default) and top-k sampling (`20` by default). Temprature is overriden to `0.2` to make code more determinisic
- Repetition penalty (`1.15` by default), applied to generated tokens rather than prompt tokens.
- Three-gram repetition blocking.
- Minimum and maximum generated-token limits (6 and 120 by default).
- Stopping when a subsequent function begins.
- Optional variable-association swapping: inferred parameter/local-variable roles are temporarily normalized during generation and restored afterward.
- Prompt cleanup and completed-function cleanup, including handling of repeated lines and trailing content.

### Running the improved sampler

The script's defaults are designed around the project's Colab directory layout. Before execution, verify the checkpoint, tokenizer, input workbook, and output workbook paths in the script. In particular, the defaults refer to a checkpoint under `/content/training_out/` and has to be named `ckpt.pt` in order for the sampler to find it. `tokenizer.json` under the repository's `training_data/` path; update them if your files are stored elsewhere, this is also need or the sampler won't be able to decode the tokens.

Use the same checkpoint and tokenizer pair used for the corresponding model run. Keep the original prompt workbook unchanged and write generated results to a separate output workbook when preserving evaluation traceability matters.

The improved sampler should be evaluated with the same prompts and comparable sampling settings as the original sampler. Record the exact script version and settings used for each before/after comparison.

## 7. Current experiment workbooks

The current repository includes these result files under `data + results/`:

| File | Purpose |
|---|---|
| `pandas_SWE_prompts_20_generated.xlsx` | Generated candidates and manual judgments for the 20 standard Pandas software-engineering prompts. |
| `pandas_exposure_prompts_evaluated_20_prerefined.xlsx` | Prerefinement exposure-probe evaluations. |
| `pandas_exposure_prompts_20_post_fixed_sample.xlsx` | Post-refinement exposure-probe generations. |
| `checkpoints and loss records/training_step_losses_and_charts.xlsx` | Training-step loss records and charts. |

The standard-prompt workbook records 20 prompts and six candidates per prompt. The reviewed outcomes are 4 accepted as-is, 10 accepted after editing, and 6 rejected. These are manual review labels, not proof of correctness across all inputs or Pandas versions.

For the exposure evaluation, distinguish manually assigned severity and unit-test-detectability judgments from experimentally measured test coverage. Do not describe a failure as caught by a unit test unless the test was implemented and run.

## Reproducibility notes

- **Use the correct baseline notebook:** `prerefined.ipynb` is the six-head baseline; `oneHead.ipynb` is the one-head ablation.
- **Run the notebook in order:** the baseline notebook orchestrates data preparation followed by model training. Sampling is a separate inference step.
- **Colab paths:** the notebooks and sampling scripts use paths such as `/content/Python_Chatbot_CS5090` and `/content/training_out`. Update paths when moving to a local environment.
- **Checkpoint name:** the sampling scripts load `ckpt.pt`; a notebook note referring to `cpt.pt` is a typo.
- **Tokenizer/checkpoint pairing:** always use the `tokenizer.json` generated for the same model run.
- **External data can change:** the source repositories are cloned from their current default branches and CodeParrot is streamed at runtime. Record source commit IDs and dataset/preparation details.
- **No dedicated unit-test suite is documented:** `test.ipynb` verifies the Colab/nanoGPT path; it is not a correctness test for generated Pandas code.
- **Dependencies and versions:** record the Python, PyTorch, CUDA, `datasets`, `tokenizers`, and other package versions used for each reproduction. A pinned dependency file is preferable for repeatability.
- **Large artifacts:** check `.gitattributes` and the repository's GitHub settings to determine whether large checkpoints or binary artifacts are stored with Git LFS. Install Git LFS if the repository uses it.

## Acknowledgements

This project uses/adapts [nanoGPT](https://github.com/karpathy/nanoGPT), public Python/Pandas and quantitative-finance repositories, and the Hugging Face [CodeParrot Clean dataset](https://huggingface.co/datasets/codeparrot/codeparrot-clean). Consult the upstream licenses and dataset terms before redistributing source data, derived corpora, or model artifacts.
