# Post-update training checklist (solar/wind)

Use this after a successful server update (`git pull`, `migrate`, service restart).

## 1) Verify repo and migrations

```bash
cd ~/forecast_platform
source venv/bin/activate
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
python3 manage.py makemigrations --check --dry-run
python3 manage.py showmigrations wind | tail -n 20
```

Expected:
- `makemigrations --check --dry-run` => `No changes detected`
- `showmigrations wind` => all migrations `[X]` (including `0004`)

## 2) Verify app process and listener

```bash
pgrep -af "gunicorn.*backend.wsgi"
ss -ltnp | rg ":8000"
```

Expected:
- gunicorn master/workers are running
- port `8000` is listening

## 3) Train models

Solar models:

```bash
python3 manage.py train_station_models --all
```

or one station:

```bash
python3 manage.py train_station_models --station-id <ID>
```

## 4) Run one scheduler tick

```bash
bash deploy/run_scheduler_tick.sh
```

## 5) Smoke-check forecast

Open station page and confirm:
- recent forecast exists
- no stack traces in logs
- MAPE starts updating with new forecast windows

