import json
import logging
from urllib.error import URLError
from urllib.request import Request, urlopen

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .serializers import CalculatorRequestSerializer
from .services.calculator_engine import calculate

logger = logging.getLogger(__name__)


def _bitrix_endpoint(method: str) -> str:
    base_url = str(getattr(settings, "BITRIX_WEBHOOK_URL", "") or "").strip()
    if not base_url:
        raise ValueError("Bitrix webhook URL is not configured.")
    return f"{base_url.rstrip('/')}/{method}.json"


def _lead_comments(payload: dict[str, str]) -> str:
    rows = [
        ("Комментарий", payload.get("comment")),
        ("Выбранный план", payload.get("selected_plan")),
        ("Цена", payload.get("price")),
        ("Панелей", payload.get("panel_count")),
        ("Мощность", payload.get("system_power_kw")),
        ("Окупаемость", payload.get("payback_years")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in rows if value)


def _send_bitrix_lead(payload: dict[str, str]) -> int | None:
    fields: dict[str, object] = {
        "TITLE": f"Заявка с калькулятора СЭС — {payload['name']}",
        "NAME": payload["name"],
        "SOURCE_ID": "WEB",
        "SOURCE_DESCRIPTION": "Калькулятор солнечных панелей",
        "PHONE": [{"VALUE": payload["phone"], "VALUE_TYPE": "WORK"}],
    }
    if payload.get("email"):
        fields["EMAIL"] = [{"VALUE": payload["email"], "VALUE_TYPE": "WORK"}]

    comments = _lead_comments(payload)
    if comments:
        fields["COMMENTS"] = comments

    body = json.dumps(
        {"fields": fields, "params": {"REGISTER_SONET_EVENT": "Y"}},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        _bitrix_endpoint("crm.lead.add"),
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    with urlopen(request, timeout=10) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    if response_payload.get("error"):
        raise ValueError(response_payload.get("error_description") or response_payload["error"])

    return response_payload.get("result")


def calculator_page(request):
    return render(request, "solar_calculator/calculator_page.html")


def seo_solar_panels_kazakhstan(request):
    return render(request, "solar_calculator/seo_solar_panels_kazakhstan.html", {"h1": "Солнечные панели в Казахстане"})


def seo_solar_panels_almaty(request):
    return render(request, "solar_calculator/seo_solar_panels_almaty.html", {"h1": "Солнечные панели в Алматы"})


def seo_sell_electricity_kz(request):
    return render(request, "solar_calculator/seo_sell_electricity_kz.html", {"h1": "Продажа электроэнергии в сеть в Казахстане"})


def seo_solar_580w(request):
    return render(request, "solar_calculator/seo_solar_580w.html", {"h1": "Солнечные панели 580 Вт"})


def sitemap_xml(request):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://intech-forecast.com/</loc></url>
  <url><loc>https://intech-forecast.com/solar-calculator</loc></url>
  <url><loc>https://intech-forecast.com/solar-panels-kazakhstan</loc></url>
  <url><loc>https://intech-forecast.com/solar-panels-almaty-price</loc></url>
  <url><loc>https://intech-forecast.com/sell-electricity-kazakhstan</loc></url>
  <url><loc>https://intech-forecast.com/solar-580w-panels</loc></url>
</urlset>"""
    return HttpResponse(xml, content_type="application/xml")


def robots_txt(request):
    body = "User-agent: *\nAllow: /\nSitemap: https://intech-forecast.com/sitemap.xml\n"
    return HttpResponse(body, content_type="text/plain")


@api_view(["POST"])
def lead_api(request):
    name = str(request.data.get("name") or "").strip()
    phone = str(request.data.get("phone") or "").strip()

    if not name or not phone:
        return Response({"success": False, "error": "Заполните имя и телефон."}, status=400)

    lead_payload = {
        "name": name,
        "phone": phone,
        "email": str(request.data.get("email") or "").strip(),
        "comment": str(request.data.get("comment") or "").strip(),
        "selected_plan": str(request.data.get("selected_plan") or "").strip(),
        "price": str(request.data.get("price") or "").strip(),
        "panel_count": str(request.data.get("panel_count") or "").strip(),
        "system_power_kw": str(request.data.get("system_power_kw") or "").strip(),
        "payback_years": str(request.data.get("payback_years") or "").strip(),
    }
    try:
        bitrix_lead_id = _send_bitrix_lead(lead_payload)
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        logger.exception(
            "Solar calculator lead submission to Bitrix failed",
            extra={"lead": lead_payload},
        )
        return Response({"success": False, "error": str(exc)}, status=502)

    logger.info(
        "Solar calculator lead submitted",
        extra={"lead": lead_payload, "bitrix_lead_id": bitrix_lead_id},
    )
    return Response({"success": True, "lead_id": bitrix_lead_id})


@api_view(["POST"])
def calculate_api(request):
    serializer = CalculatorRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    payload = serializer.validated_data
    output = calculate(payload["mode"], payload["inputs"])
    return Response(output)
