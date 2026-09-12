# Code Catalyst

**A self-hosted coding assistant.** A LLaMA-2 / CodeLlama model fine-tuned on real Stack
Overflow question–answer pairs, quantised to 4-bit so it runs on a single consumer GPU, and
served through a Flask web app with accounts and conversation memory.

Started in mid-2023 — before ChatGPT made this category obvious.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?logo=sqlalchemy&logoColor=white)

---

## The idea

A general-purpose model answering "how do I do X in Python?" gives you prose with code buried
in it. What you usually want is the snippet.

CoNaLa is a dataset built for exactly that: natural-language programming intents paired with
the Python one-liners that satisfy them, mined from Stack Overflow. Fine-tuning on it teaches
a model to answer in the shape the question implies — code first, minimal preamble.

The second constraint was hardware. A 7B model in full precision needs more VRAM than a
single consumer card has, and fine-tuning one needs several times that again. The whole
project is built around **QLoRA**, which makes both fit.

## The fine-tuning

### Quantisation

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit_fp32_cpu_offload=True,
)
```

Each choice does specific work:

- **`nf4`** — NormalFloat4, a 4-bit type whose quantisation levels are spaced to match the
  normal distribution that neural network weights actually follow. It loses less than
  uniform 4-bit integer quantisation at the same bit width.
- **`double_quant`** — quantises the quantisation constants themselves, saving roughly another
  0.4 bits per parameter. Small, free, no quality cost.
- **`bfloat16` compute dtype** — weights are stored in 4-bit but dequantised to bf16 for the
  actual matmul. bf16 keeps fp32's exponent range, so it doesn't overflow the way fp16 does
  during backward passes.
- **`fp32_cpu_offload`** — lets layers that won't fit spill to CPU rather than failing
  outright.

The base weights stay frozen; only low-rank adapters train. That's what makes a 7B fine-tune
possible on one GPU.

### Training configuration

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # effective batch size 8
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    weight_decay=0.001,
    optim="paged_adamw_32bit",
    fp16=True,
    group_by_length=True,
    save_steps=500,
    save_total_limit=2,
)
```

The interesting parameters are the ones fighting memory:

- **`per_device_train_batch_size=1` with `gradient_accumulation_steps=8`** — an effective batch
  of 8 without ever holding 8 sequences in memory at once. Gradients accumulate across eight
  forward/backward passes before a single optimiser step.
- **`paged_adamw_32bit`** — Adam keeps two fp32 moments per parameter, which for a 7B model is
  more memory than the model itself. The paged optimiser moves that state between GPU and CPU
  on demand, turning an out-of-memory crash into a slowdown.
- **`group_by_length=True`** — batches similar-length sequences together so padding tokens
  don't dominate the batch. On a dataset of short snippets with a long tail, this is a
  significant throughput win for one line of config.
- **`lr_scheduler_type="constant"` with `warmup_ratio=0.03`** — a brief warmup then a flat
  2e-4. LoRA adapters are small and randomly initialised, so they tolerate a much higher
  learning rate than full fine-tuning would, and don't need the decay schedule a full
  fine-tune depends on.
- **`save_total_limit=2`** — keeps only the two most recent checkpoints, because 3 epochs at
  500-step saves would otherwise fill the disk.

### The dataset

`conala-paired-train (1).json` — **2,379 records** (JSON Lines, one object per line):

```json
{
  "intent": "How to convert a list of multiple integers into a single integer?",
  "rewritten_intent": "Concatenate elements of a list 'x' of multiple integers to a single integer",
  "snippet": "sum(d * 10 ** i for i, d in enumerate(x[::-1]))",
  "question_id": 41067960
}
```

Each record carries the raw Stack Overflow title (`intent`), a hand-cleaned restatement
(`rewritten_intent`), and the answer (`snippet`). The rewritten field matters: raw titles are
often vague or badly worded, while the rewritten version names the actual variables, which
gives the model a much cleaner mapping from request to code.

Note that multiple records share a `question_id` — one question with several valid answers.
That's useful signal, since it teaches the model that a single intent has more than one
correct implementation.

## The application

`app.py` is a Flask app doing four things:

**Model serving.** Loads the fine-tuned model once at startup with the quantisation config
above and pins it to `cuda:0`, then generates with a repetition penalty of 1.1 to suppress
the looping that small quantised models are prone to.

**Accounts.** SQLAlchemy models with WTForms-validated registration and login, and passwords
stored as hashes rather than plaintext.

**Conversation memory.** Turns are appended to a history list and the whole exchange is
replayed into each prompt, so follow-up questions like "now make it handle negatives" resolve
against what came before instead of arriving contextless.

**Runtime generation controls.** `GET`/`POST /update-model-config` change temperature and
`max_new_tokens` without a restart, so decoding behaviour can be tuned while the model stays
loaded — reloading a 4-bit 7B model takes long enough that restarting to try a different
temperature isn't practical.

### Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Chat interface |
| `GET`/`POST` | `/register` | Create an account |
| `GET`/`POST` | `/login` | Sign in |
| `POST` | `/submit-prompt` | Send a prompt; `NEW_CHAT` clears history |
| `GET` | `/get-model-config` | Read current temperature and token limit |
| `POST` | `/update-model-config` | Change them at runtime |
| `GET` | `/users` | List registered users |
| `GET` | `/initdb` | Create database tables |

## Project structure

```
.
├── app.py                          # Flask backend: model loading, routes, auth, chat
├── configuration.py                # Fine-tuning TrainingArguments
├── conala-paired-train (1).json    # 2,379 intent→snippet pairs (JSONL)
├── Templates/                      # index, login, add_user, list_users
├── static/                         # Stylesheet and images
└── Fine-tuned LLaMa2               # Link to the trained weights
```

## The model weights

The fine-tuned model is too large for Git and is hosted separately —
see [`Fine-tuned LLaMa2`](./Fine-tuned%20LLaMa2) for the download link. It's shared openly;
a citation is appreciated if you use it.

## Running it

**Prerequisites:** Python 3.10+, a CUDA GPU with ~6 GB VRAM, PyTorch with CUDA support.

```bash
pip install flask flask-sqlalchemy flask-wtf transformers torch bitsandbytes accelerate

# Download the weights (see link above), then point app.py at them:
#   path = "/path/to/your/model"

python app.py          # then open http://localhost:5000
```

Visit `/initdb` once to create the user tables.

## Current state

This is a 2023–2024 project brought online later, and it hasn't been run against current
library versions. Known issues, tracked honestly:

- **Login is broken.** `check_password_hash(user.password.data, ...)` calls `.data` on a string
  — `user.password` is already the hash. Should be `check_password_hash(user.password, ...)`.
- **`generate_password_hash(..., method='sha256')` no longer exists** in Werkzeug 3.x, and
  plain SHA-256 was never appropriate for passwords. Drop the argument and take the default
  (scrypt).
- **`register.html` is missing** — the route renders a template that isn't in the repo, so
  `/register` returns a 500.
- **The template folder is `Templates/`, capitalised.** Flask looks for `templates/`, so this
  only works on case-insensitive filesystems (Windows, default macOS) and fails on Linux.
- **Chat history and generation settings are module-level globals**, shared across every
  visitor — two users see one conversation, and one changing the temperature changes it for
  everyone. Both belong in the Flask session or the database, keyed by user.
- **History grows without bound.** The full transcript is replayed into every prompt, so a
  long conversation eventually exceeds the context window. It needs truncation or
  summarisation.
- **Authentication doesn't gate anything.** `session['user_id']` is set at login but never
  checked, so `/`, `/submit-prompt`, and `/users` are open to anyone. There's no `/logout`.
- **`SECRET_KEY = os.urandom(24)`** is regenerated on every start, invalidating all sessions
  on restart and breaking entirely across multiple workers. Load it from the environment.
- **Hardcoded model path** (`/mnt/c/Users/...`) and a placeholder HF token in the source.
- **`debug=True`** in production would expose the Werkzeug debugger.
- **The LoRA adapter config isn't in the repo.** `BitsAndBytesConfig` and `TrainingArguments`
  are both here, but the `LoraConfig` — rank, alpha, dropout, target modules — is the heart of
  a QLoRA setup and is missing. Worth adding for anyone reproducing the fine-tune.

## Roadmap

- [ ] Fix the login comparison and password hashing method
- [ ] Add the missing `register.html`; rename `Templates/` → `templates/`
- [ ] Move chat history and generation settings into per-user session state
- [ ] Truncate or summarise history to stay inside the context window
- [ ] Add a `@login_required` decorator and a logout route
- [ ] Read the secret key, model path, and HF token from environment variables
- [ ] Commit the `LoraConfig` alongside the training arguments
- [ ] Publish the adapter weights to the Hugging Face Hub instead of Drive
- [ ] Evaluate against CoNaLa's test split (BLEU / exact match) to quantify the fine-tune

## License

MIT

---

Questions, bugs, or suggestions are welcome — open an issue.
