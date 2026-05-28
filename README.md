# Benchmarking OpenHands with any LLM

## Prepare venv

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip uv

git submodule update --init --recursive
uv sync --dev --active
```

## Azure OpenAI identity login

```bash
source venv/bin/activate
python -m pip install openai azure-identity-broker --upgrade
```


## Rollout
```bash
source venv/bin/activate
python main.py \
    --config config/default.yaml \
    --run-id debug \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --n-limit 1

nohup python main.py \
    --config config/ds4pro.yaml \
    --run-id multilang \
    --dataset datasets/multilang.jsonl \
    --num-workers 1 \
    > log-ds4pro.out 2>&1 &
```

Outputs are written under `logs/<model>/<run-id>/`. The wrapper also writes `preds.json` in the same directory with `instance_id` and `model_patch`, matching the local evaluation format used by the other agents.

