import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .serializers import CalculatorRequestSerializer, LeadRequestSerializer
from .services.calculator_engine import calculate


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
def calculate_api(request):
    serializer = CalculatorRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    payload = serializer.validated_data
    output = calculate(payload["mode"], payload["inputs"])
    return Response(output)


@api_view(["POST"])
def create_lead_api(request):
    serializer = LeadRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    payload = serializer.validated_data

    selected_plan = payload.get("selected_plan") or "Без пакета"
    title = f"Заявка с калькулятора InTech ({selected_plan})"
    comments = (
        "=== Solar калькулятор ===\n\n"
        f"Пакет: {selected_plan}\n"
        f"Цена: {payload.get('price') or '—'}\n"
        f"Панели: {payload.get('panel_count') or '—'}\n"
        f"Мощность: {payload.get('system_power_kw') or '—'}\n"
        f"Окупаемость: {payload.get('payback_years') or '—'}\n\n"
        f"Комментарий клиента: {payload.get('comment') or '—'}"
    )
    fields = {
        "TITLE": title,
        "NAME": payload["name"],
        "PHONE": [{"VALUE": payload["phone"], "VALUE_TYPE": "WORK"}],
        "ASSIGNED_BY_ID": 1,
        "SOURCE_ID": "WEB",
        "COMMENTS": comments,
    }
    if payload.get("email"):
        fields["EMAIL"] = [{"VALUE": payload["email"], "VALUE_TYPE": "WORK"}]

    bitrix_url = f"{settings.BITRIX_WEBHOOK_URL.rstrip('/')}/crm.lead.add.json"
    body = json.dumps({"fields": fields}).encode("utf-8")
    req = Request(bitrix_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        return Response({"success": False, "error": f"Bitrix HTTP {exc.code}"}, status=502)
    except URLError as exc:
        return Response({"success": False, "error": f"Bitrix unavailable: {exc.reason}"}, status=502)
    except Exception as exc:
        return Response({"success": False, "error": str(exc)}, status=500)

    if data.get("error"):
        return Response({"success": False, "error": data.get("error_description") or data["error"]}, status=502)
    return Response({"success": True, "bitrix_lead_id": data.get("result")})
