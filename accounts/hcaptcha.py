import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from django.conf import settings


def verify_hcaptcha(token: str, remote_ip: str | None = None) -> tuple[bool, str]:
    secret = getattr(settings, "HCAPTCHA_SECRET_KEY", "")
    if not secret:
        return False, "HCAPTCHA_SECRET_KEY не настроен на сервере."
    if not token:
        return False, "Подтвердите, что вы не робот."

    payload = {"secret": secret, "response": token}
    if remote_ip:
        payload["remoteip"] = remote_ip

    req = Request(
        "https://hcaptcha.com/siteverify",
        data=urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
    except Exception:
        return False, "Не удалось проверить hCaptcha. Попробуйте еще раз."

    if data.get("success"):
        return True, ""
    return False, "Проверка hCaptcha не пройдена."
