# Financial Document NER Prototype (Step-by-step)

## Step 1: Inference sanity check (no training yet)

### Setup
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run
```bash
python -m src.infer --text "On Jan 31, 2026, Apple Inc. reported revenue of $119.6B and net income of $33.9B."
```
