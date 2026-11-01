# Python Chatbot — CS 5090

A small GPT-style Python code language model trained from scratch with [nanoGPT](https://github.com/karpathy/nanoGPT). The project focuses on generating and completing Python code that uses Pandas and DataFrame operations. It was developed as the CS 5090 Project 1: **Build and Evaluate a Small Code Language Model**.

The repository contains a six-attention-head baseline model, a one-head ablation, a Google Colab/nanoGPT setup test, and a notebook for sampling model outputs against real software-engineering prompts.

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
├── nanoGPT/
│   ├── model.py                    # GPT model and attention implementation
│   ├── train.py                    # Training and validation loop
│   ├── sample.py                   # Generic nanoGPT sampler (used by setup test)
│   ├── sample-python.py            # Single-prompt Python code sampling
│   ├── sample-python-bulk.py       # Excel-based batch prompt sampling
│   └── data/
│       └── shakespeare_char/       # nanoGPT example dataset for test.ipynb
├── training_data/
│   ├── prepare.py                  # Filters source code, trains tokenizer, builds train/val data
│   ├── files/                      # Cloned source repositories scanned by prepare.py
│   ├── Prompts/                    # Excel prompt workbooks used by sampler.ipynb
│   ├── train.bin                   # Generated training token IDs
│   ├── val.bin                     # Generated validation token IDs
│   ├── meta.pkl                    # Vocabulary metadata for nanoGPT
│   └── tokenizer.json              # Custom BPE tokenizer used for sampling
├── prerefined.ipynb                # Primary six-head baseline
├── oneHead.ipynb                   # One-head ablation
├── sampler.ipynb                  # Inference and prompt-based evaluation
├── test.ipynb                     # Google Colab/nanoGPT setup smoke test
├── old/                            # Earlier project material
└── .gitattributes                  # Git LFS rules for model and binary artifacts
```

The `.bin`, `.pkl`, and tokenizer files under `training_data/` are generated artifacts. They are not the original source corpus. The original local code sources are placed under `training_data/files/`.

## 1. Reproducing the six-head baseline

The intended reproduction path is to run [`prerefined.ipynb`](prerefined.ipynb) from top to bottom in Google Colab.

### Environment

1. Open `prerefined.ipynb` in Google Colab.
2. Select a GPU runtime. A GPU is strongly recommended for training; the course brief estimates a successful run may take approximately 2–6 GPU-hours, depending on the environment.
3. Run the notebook cells in their existing order. The notebook contains the setup, data preparation, and model-training steps.
4. Keep the same repository version, source data, tokenizer, and configuration when reproducing the run.

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
| Training tokens | `training_data/train.bin` | Tokenized examples used for model training. |
| Validation tokens | `training_data/val.bin` | Held-out examples used for validation loss. |
| Vocabulary metadata | `training_data/meta.pkl` | Token/vocabulary mappings expected by nanoGPT. |
| BPE tokenizer | `training_data/tokenizer.json` | Tokenizer used by the custom Python samplers. |
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

The notebook clones these repositories from their default branches. Their contents can change over time; for strict reproducibility, record the commit SHA for each source repository.

### Hugging Face dataset

In addition to the local repositories, `training_data/prepare.py` streams the **training split** of [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean) through the Hugging Face `datasets` library. The script extracts additional Pandas-related functions until it reaches a target of 400,000,000 characters (the script comments estimate this at roughly 100 million tokens). The stream can fail or be interrupted, so inspect the preparation output and record the final counts rather than assuming the full remote corpus was collected.

There are two distinct splits to keep straight:

- **CodeParrot's upstream split:** the project script requests the dataset's `train` split from Hugging Face.
- **This project's 90/10 split:** after collecting and filtering code from the local repositories and CodeParrot, `prepare.py` creates its own training and validation partitions. The explanation below refers to this project-level split, not to the upstream Hugging Face split.

### Third-party source licenses

The table below records the license information visible in each upstream repository or its license/metadata file when checked on **October 31, 2026**. The licenses apply to the upstream source projects, not automatically to this repository as a whole. The notebook clones current default branches, so the license and source content at the exact commit used for a training run should be saved alongside the experiment.

| Source repository | License found | License / source reference |
|---|---|---|
| [leetlab11/Introduction-to-Pandas](https://github.com/leetlab11/Introduction-to-Pandas) | **No explicit repository license found** | [Repository root](https://github.com/leetlab11/Introduction-to-Pandas). No `LICENSE` file or license declaration was visible in the root listing when checked. |
| [Aayush-kotwani/Pandas-masterclass](https://github.com/Aayush-kotwani/Pandas-masterclass) | **No explicit repository license found** | [Repository root](https://github.com/Aayush-kotwani/Pandas-masterclass). No `LICENSE` file or license declaration was visible in the root listing when checked. |
| [codebasics/py](https://github.com/codebasics/py) | **No explicit repository license found** | [Repository root](https://github.com/codebasics/py). No `LICENSE` file or license declaration was visible in the root listing when checked. |
| [guipsamora/pandas_exercises](https://github.com/guipsamora/pandas_exercises) | **BSD 3-Clause** | [LICENSE](https://github.com/guipsamora/pandas_exercises/blob/master/LICENSE) |
| [jvns/pandas-cookbook](https://github.com/jvns/pandas-cookbook) | **CC BY-SA 4.0** | [License statement in README](https://github.com/jvns/pandas-cookbook#license) |
| [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | **Apache License 2.0** | [LICENSE](https://github.com/stefan-jansen/zipline-reloaded/blob/main/LICENSE) and [package metadata](https://github.com/stefan-jansen/zipline-reloaded/blob/main/pyproject.toml) |
| [twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta) | **MIT** | [LICENCE](https://github.com/aarigs/pandas-ta?tab=MIT-1-ov-file) |
| [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | **Apache License 2.0 + Commons Clause** | [LICENSE.md](https://github.com/polakowo/vectorbt/blob/master/LICENSE.md). The Commons Clause adds a restriction on selling the software or a product/service whose value derives substantially from it. |
| [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) | **Apache License 2.0** | [LICENSE.txt](https://github.com/ranaroussi/yfinance/blob/main/LICENSE.txt); the [project README](https://github.com/ranaroussi/yfinance#legal-stuff) identifies the Apache license. |
| [quantopian/pyfolio](https://github.com/quantopian/pyfolio) | **Apache License 2.0** | [LICENSE](https://github.com/quantopian/pyfolio/blob/master/LICENSE) |
| [quantopian/alphalens](https://github.com/quantopian/alphalens) | **Apache License 2.0** | [LICENSE](https://github.com/quantopian/alphalens/blob/master/LICENSE) |
| [pmorissette/bt](https://github.com/pmorissette/bt) | **MIT** | [LICENSE](https://github.com/pmorissette/bt/blob/master/LICENSE) |
| [bukosabino/ta](https://github.com/bukosabino/ta) | **MIT** | [MIT license](https://github.com/bukosabino/ta/blob/master/LICENSE) |

**Important: repositories without an explicit license.** The three repositories marked above have no license grant visible in the repository listing checked. They are all learning based repositories so they should be ok. This project is **not for redistribution**

### Hugging Face CodeParrot licensing

The [`codeparrot/codeparrot-clean` dataset card](https://huggingface.co/datasets/codeparrot/codeparrot-clean) describes a collection of Python files from GitHub and lists `license` as a **per-record field** in its schema. The viewer shows multiple license identifiers, including `apache-2.0`, `bsd-3-clause`, `gpl-2.0`, `gpl-3.0`, and `mit`; it exposes 15 distinct values in the current viewer. The dataset card does **not** provide one blanket license that relicenses every code file in the collection under a single permissive license.

For that reason, CodeParrot should be treated as a **mixed-license dataset**. A file's terms depend on its source repository and the `license` value recorded for that row. The fact that the dataset is hosted on Hugging Face does not replace the original source licenses. In addition, the current preparation workflow extracts matching functions from the streamed records without an evident per-record license allowlist. Unless that is changed, the generated corpus may contain snippets under different license terms, and the exact source-license distribution is not known from the model checkpoint alone.

For future data rebuilds, preserve the CodeParrot row metadata (`repo_name`, `path`, and `license`) for every extracted function. Apply and document an explicit license policy before including records, skip records with missing/unknown license metadata unless approved, and save a manifest mapping each included function back to its source row. This is especially important before publishing or redistributing the corpus, generated artifacts, or trained weights.

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

### Why use a training/validation split?

The split separates the data used to **fit the model** from data used to **check how well the trained model generalizes**:

- **Training partition (90%):** `train.py` uses these tokens to calculate the training loss and update the model weights through backpropagation.
- **Validation partition (10%):** the model is evaluated on these tokens without using them to update its weights. Validation loss estimates next-token prediction performance on examples that were held out from the weight updates.

This serves three purposes. First, it provides a check on whether the model is learning patterns that carry over to unseen examples rather than simply becoming better at the training corpus. Second, comparing training loss with validation loss helps reveal overfitting: training loss may continue to fall while validation loss stops improving or increases. Third, the same held-out validation data gives a consistent basis for comparing the six-head baseline and one-head ablation. Using 90% for training leaves most of the collected corpus available to learn from while retaining a meaningful 10% sample for this check.

The validation partition is **not a second training set** and should not be used for gradient updates. It is also not the same as the 20 real software-engineering prompts or the Phase 3 probe set: those are separate task-level evaluations of generated code. For a fair ablation, keep the same `train.bin` and `val.bin` data (and tokenizer) for both models so that the head-count change is the main experimental difference.

A random split is only a useful holdout when examples are sufficiently independent. Duplicate or near-duplicate functions, or functions copied from the same repository appearing in both partitions, can make validation loss look more representative than it really is. The preparation script should therefore document its deduplication policy and consider source-aware grouping if the goal is to estimate performance on entirely new repositories.

### Data preparation notes

The project brief asks that the data sheet document filtering, deduplication, tokenization, and splitting, and specifies that the tokenizer should be trained on the training split only.

Two points should be addressed when documenting or refining the current pipeline:

- **Tokenizer fit and split order:** the current `prepare.py` workflow appears to fit the tokenizer before making the 90/10 train/validation split. That is not the order specified in the project brief and weakens the holdout: although validation tokens are not used for gradient updates, the tokenizer vocabulary/merges have already been influenced by validation text. For strict compliance, split the function chunks first, fit the tokenizer only on the training partition, and then use that tokenizer to encode both partitions.
- **Deduplication:** the preparation steps described above filter and shuffle functions, but do not describe a separate deduplication pass. State clearly in the project data sheet whether duplicates were removed. Do not claim deduplication unless it was actually performed.

Use the [third-party source license inventory](#third-party-source-licenses) and [CodeParrot licensing notes](#hugging-face-codeparrot-licensing) when recording source provenance and deciding whether any corpus or trained artifact can be redistributed.

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

## 6. Evaluation and project evidence

This repo represents a course project that is organized into four phases. Model quality alone is not the target; the project is assessed on understanding, experimental design, evidence, analysis, and reproducibility.

### Phase 1 — Build and understand

- Reproducible six-head baseline and one-head ablation.
- Model architecture/configuration and training environment.
- Data filtering, deduplication status, tokenizer training, and train/validation split details.
- Training and validation loss results, including a loss curve.
- Ablation prediction, exact change, validation-loss comparison, and side-by-side generation example.
- Limitations and data/licensing considerations.

### Phase 2 — Use the model on 20 real inputs

- The source and purpose of the 20 prompts.
- The software-engineering task being attempted (in this repository, Python/Pandas code completion).
- The meaning of an acceptable output for that task.
- Accept-as-is, accept-after-editing, and reject counts.
- The best, worst, and near-miss examples with short explanations.
- A brief assessment of where the model saved time and where verification cost more than writing the code manually.

### Phase 3 — Evaluate and break

sample approximately 20–30 probes designed to expose model weaknesses. Include ordinary prompts and targeted cases such as misleading comments or names, long context, security-related prompts, prompt variations, and prompts similar to the training data.

Record at least 10 failure instances in a CSV. Use these category names exactly:

| Category | Meaning |
|---|---|
| `syntax` | Output does not parse. |
| `invented-api` | Calls an API or function that does not exist. |
| `wrong-logic` | Parses/runs but produces the wrong behavior. |
| `insecure` | Introduces a security weakness. |
| `memorised` | Reproduces training data nearly verbatim. |
| `prompt-sensitive` | A small rewording changes the answer substantially. |
| `context-ignored` | Ignores information clearly present in the input. |
| `other` | Use only when none of the above applies; explain clearly. |

this repo goes over 4 in 20 different probing prompts:

prompt-sensitive
unkown variable names
invented-api
syntax

### Phase 4 — Improve and re-evaluate

the sampler is improved, noted as `sample-python-bulk-improved.py` in the nanogpt folder

any of the required files can be found in `data + results`

## Reproducibility notes

- **Use the correct baseline notebook:** `prerefined.ipynb` is the six-head baseline; `oneHead.ipynb` is the one-head ablation.
- **Run the notebook in order:** the baseline notebook orchestrates data preparation followed by model training. Sampling is a separate inference step.
- **Colab paths:** the notebooks and sampling scripts use paths such as `/content/Python_Chatbot_CS5090` and `/content/training_out`. Update paths when moving to a local environment. Also make sure to add `tokenizer.json` to `/content/Python_Chatbot_CS5090/training_data` before runing a checkpoint
- **Checkpoint name:** the sampling scripts load `ckpt.pt`; a notebook note referring to `cpt.pt` is a typo.
- **Tokenizer/checkpoint pairing:** always use the `tokenizer.json` generated for the same model run.
- **External data can change:** the source repositories are cloned from their current default branches and CodeParrot is streamed at runtime. Record source commit IDs and dataset/preparation details.
- **No dedicated unit-test suite is documented:** `test.ipynb` verifies the Colab/nanoGPT path; it is not a correctness test for generated Pandas code.
- **Dependencies and versions:** record the Python, PyTorch, CUDA, `datasets`, `tokenizers`, and other package versions used for each reproduction. A pinned dependency file is preferable for repeatability.
- **Git LFS:** `.gitattributes` marks model and binary artifacts for Git LFS. Install Git LFS before cloning or retrieving large tracked artifacts.

## Acknowledgements

This project uses/adapts [nanoGPT](https://github.com/karpathy/nanoGPT), public Python/Pandas and quantitative-finance repositories, and the Hugging Face [CodeParrot Clean dataset](https://huggingface.co/datasets/codeparrot/codeparrot-clean). Third-party components and source snippets remain subject to their respective licenses. Review the [license inventory](#third-party-source-licenses) and [CodeParrot licensing notes](#hugging-face-codeparrot-licensing) before redistributing source data, derived corpora, or model artifacts.
