from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404


def brand_logo(request):
    """Serve navbar logo via Django so it works even when static files are misconfigured."""
    base_dir = Path(settings.BASE_DIR)
    candidates = [
        base_dir / "dashboard/static/dashboard/img/logo.png",
        base_dir / "dashboard/static/dashboard/img/intech-logo.svg",
    ]

    for logo_path in candidates:
        if logo_path.exists():
            content_type = "image/png" if logo_path.suffix.lower() == ".png" else "image/svg+xml"
            return FileResponse(logo_path.open("rb"), content_type=content_type)

    raise Http404("Brand logo file not found")
