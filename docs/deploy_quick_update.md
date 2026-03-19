```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m 'manual-before-update'
git pull --rebase
source venv/bin/activate
python3 manage.py migrate
python3 manage.py collectstatic --noinput
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файла нет:
# GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```
