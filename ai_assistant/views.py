from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from typing import Optional

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_POST

from stations.models import Organization, OrganizationMember, Station

from . import services

logger = logging.getLogger(__name__)

INTENT_GET_YESTERDAY_GENERATION = "get_yesterday_generation"
INTENT_GET_TOMORROW_FORECAST = "get_tomorrow_forecast"
INTENT_GET_TODAY_PLANFACT = "get_today_planfact"
INTENT_GET_YESTERDAY_PLANFACT = "get_yesterday_planfact"
INTENT_OPEN_TOMORROW_FORECAST = "open_tomorrow_forecast"
INTENT_OPEN_PLANFACT_TODAY = "open_planfact_today"
INTENT_OPEN_PLANFACT_YESTERDAY = "open_planfact_yesterday"

READ_INTENTS = {
    INTENT_GET_YESTERDAY_GENERATION,
    INTENT_GET_TOMORROW_FORECAST,
    INTENT_GET_TODAY_PLANFACT,
    INTENT_GET_YESTERDAY_PLANFACT,
}
NAVIGATION_INTENTS = {
    INTENT_OPEN_TOMORROW_FORECAST,
    INTENT_OPEN_PLANFACT_TODAY,
    INTENT_OPEN_PLANFACT_YESTERDAY,
}


def _station_queryset_for_user(user):
    org_ids = Organization.objects.filter(owner=user).values_list("id", flat=True)
    member_org_ids = OrganizationMember.objects.filter(user=user).values_list("organization_id", flat=True)
    return Station.objects.filter(org_id__in=org_ids.union(member_org_ids), station_kind=Station.KIND_SOLAR).distinct()


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def _detect_intent(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    wants_open = any(word in normalized for word in ("открой", "открыть", "перейди", "покажи страницу", "перейти"))
    mentions_tomorrow = "завтра" in normalized or "завтраш" in normalized
    mentions_yesterday = "вчера" in normalized or "вчераш" in normalized
    mentions_today = "сегодня" in normalized or "сегодняш" in normalized
    mentions_forecast = "прогноз" in normalized
    mentions_planfact = any(word in normalized for word in ("план факт", "план-факт", "план/факт", "планфакт"))
    mentions_generation = any(word in normalized for word in ("выработка", "генерация", "сгенерировала", "производство"))

    if wants_open and mentions_forecast and mentions_tomorrow:
        return INTENT_OPEN_TOMORROW_FORECAST
    if wants_open and mentions_planfact and mentions_yesterday:
        return INTENT_OPEN_PLANFACT_YESTERDAY
    if wants_open and mentions_planfact:
        return INTENT_OPEN_PLANFACT_TODAY
    if mentions_forecast and mentions_tomorrow:
        return INTENT_GET_TOMORROW_FORECAST
    if mentions_planfact and mentions_yesterday:
        return INTENT_GET_YESTERDAY_PLANFACT
    if mentions_planfact and (mentions_today or not mentions_yesterday):
        return INTENT_GET_TODAY_PLANFACT
    if mentions_generation and mentions_yesterday:
        return INTENT_GET_YESTERDAY_GENERATION
    return None


def _resolve_station(text: str, user) -> Optional[Station]:
    stations = list(_station_queryset_for_user(user).order_by("sort_order", "id"))
    if not stations:
        return None

    explicit_id = re.search(r"(?:станци(?:я|и|ю)|station|id)\s*#?\s*(\d+)(?![\.,]\d)", text, flags=re.IGNORECASE)
    if explicit_id:
        station_id = int(explicit_id.group(1))
        for station in stations:
            if station.pk == station_id:
                return station

    normalized = _normalize_text(text)
    for station in stations:
        if _normalize_text(station.name) in normalized:
            return station

    decimal_values = re.findall(r"\d+(?:[\.,]\d+)?", text)
    for raw_value in decimal_values:
        value = raw_value.replace(",", ".")
        for station in stations:
            if value in _normalize_text(station.name):
                return station
            try:
                if abs(float(value) - float(station.capacity_mw)) < 0.01:
                    return station
            except (TypeError, ValueError):
                continue

    return stations[0]


def _format_kwh(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "н/д"
    return f"{value:.1f}%"


def _answer_for_read_intent(intent: str, station_id: int) -> str:
    if intent == INTENT_GET_YESTERDAY_GENERATION:
        result = services.get_yesterday_generation(station_id)
        if result.points_count == 0:
            return f"За вчера ({result.date:%d.%m.%Y}) нет данных выработки по {result.station_name}."
        return f"Вчера выработка по {result.station_name} составила {_format_kwh(result.energy_kwh)} кВт·ч."

    if intent == INTENT_GET_TOMORROW_FORECAST:
        result = services.get_tomorrow_forecast(station_id)
        if result.points_count == 0:
            return f"На завтра ({result.date:%d.%m.%Y}) нет сохраненного прогноза по {result.station_name}."
        return f"Прогноз выработки на завтра по {result.station_name}: {_format_kwh(result.energy_kwh)} кВт·ч."

    if intent == INTENT_GET_TODAY_PLANFACT:
        result = services.get_today_planfact(station_id)
        return (
            f"План/факт за сегодня по {result.station_name}: факт {_format_kwh(result.fact_kwh)} кВт·ч, "
            f"план {_format_kwh(result.plan_kwh)} кВт·ч, отклонение {_format_kwh(result.deviation_kwh)} кВт·ч "
            f"({_format_percent(result.deviation_percent)})."
        )

    if intent == INTENT_GET_YESTERDAY_PLANFACT:
        result = services.get_yesterday_planfact(station_id)
        return (
            f"План/факт за вчера по {result.station_name}: факт {_format_kwh(result.fact_kwh)} кВт·ч, "
            f"план {_format_kwh(result.plan_kwh)} кВт·ч, отклонение {_format_kwh(result.deviation_kwh)} кВт·ч "
            f"({_format_percent(result.deviation_percent)})."
        )

    return "Не удалось подготовить ответ по выбранному намерению."


@login_required
@require_POST
def assistant_query(request):
    received_at = timezone.now()
    success = False
    intent = None
    station_id = None
    text = ""

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
        text = str(payload.get("text") or "").strip()
        if not text:
            return JsonResponse({"answer": "Введите вопрос для ассистента.", "action": None}, status=400)

        intent = _detect_intent(text)
        station = _resolve_station(text, request.user)
        station_id = station.pk if station else None

        if intent is None:
            return JsonResponse(
                {
                    "answer": "Пока я умею отвечать только по выработке, прогнозу и план/факту за сегодня, вчера или завтра.",
                    "action": None,
                },
                status=400,
            )
        if station is None:
            return JsonResponse({"answer": "Не найдена доступная солнечная станция для вашего пользователя.", "action": None}, status=404)

        if intent in READ_INTENTS:
            answer = _answer_for_read_intent(intent, station.pk)
            action = None
        elif intent in NAVIGATION_INTENTS:
            action = services.build_navigation_action(intent, station.pk)
            if intent == INTENT_OPEN_TOMORROW_FORECAST:
                target_date = timezone.localdate() + timedelta(days=1)
                answer = f"Открываю прогноз на завтра ({target_date:%d.%m.%Y}) по {station.name}."
            elif intent == INTENT_OPEN_PLANFACT_YESTERDAY:
                target_date = timezone.localdate() - timedelta(days=1)
                answer = f"Открываю план/факт за вчера ({target_date:%d.%m.%Y}) по {station.name}."
            else:
                answer = f"Открываю план/факт за сегодня ({timezone.localdate():%d.%m.%Y}) по {station.name}."
        else:
            return JsonResponse({"answer": "Этот intent не разрешен на первом этапе.", "action": None}, status=400)

        success = True
        return JsonResponse({"answer": answer, "action": action})
    except json.JSONDecodeError:
        return JsonResponse({"answer": "Некорректный JSON в запросе.", "action": None}, status=400)
    except Exception:
        logger.exception("AI assistant query failed")
        return JsonResponse({"answer": "Ассистент временно не смог обработать запрос.", "action": None}, status=500)
    finally:
        logger.info(
            "ai_assistant_query",
            extra={
                "question": text,
                "intent": intent,
                "station_id": station_id,
                "received_at": received_at.isoformat(),
                "success": success,
                "user_id": getattr(request.user, "id", None),
            },
        )
