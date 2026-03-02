# Aria Designer API

Contract-first backend for:

- component catalog lifecycle,
- workflow validation/compile/run requests,
- Aria patch proposal + approval flow.

This scaffold keeps all data in memory for standalone development.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8091
```

## Endpoints (scaffold)

- `GET /health`
- `GET /components`
- `POST /components`
- `POST /workflows/validate`
- `POST /workflows/compile`
- `POST /workflows/run`
- `POST /aria/propose-patch`
- `POST /aria/apply-patch`
