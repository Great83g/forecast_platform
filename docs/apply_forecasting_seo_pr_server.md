# Как применить изменения PR на сервере

Скопируйте весь блок и выполните на сервере:

```bash
cd ~/forecast_platform
source venv/bin/activate
set -euo pipefail

git fetch --all --tags --prune
git checkout main
git reset --hard origin/main
git log -1 --oneline

python3 manage.py migrate
python3 manage.py collectstatic --noinput

if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
fi

curl -I http://127.0.0.1:8000/login/?next=/dashboard/
```
