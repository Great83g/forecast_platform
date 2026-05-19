from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_POST

from stations.models import Organization, OrganizationMember, Station

from . import services
from .date_parser import parse_period

logger = logging.getLogger(__name__)

INTENT_GET_YESTERDAY_GENERATION = "get_yesterday_generation"
INTENT_GET_TOMORROW_FORECAST = "get_tomorrow_forecast"
INTENT_GET_TODAY_PLANFACT = "get_today_planfact"
INTENT_GET_YESTERDAY_PLANFACT = "get_yesterday_planfact"
INTENT_GET_GENERATION_PERIOD = "get_generation_period"
INTENT_GET_FORECAST_PERIOD = "get_forecast_period"
INTENT_GET_PLANFACT_PERIOD = "get_planfact_period"
INTENT_OPEN_TOMORROW_FORECAST = "open_tomorrow_forecast"
INTENT_OPEN_PLANFACT_TODAY = "open_planfact_today"
INTENT_OPEN_PLANFACT_YESTERDAY = "open_planfact_yesterday"

READ_INTENTS = {
    INTENT_GET_YESTERDAY_GENERATION,
    INTENT_GET_TOMORROW_FORECAST,
    INTENT_GET_TODAY_PLANFACT,
    INTENT_GET_YESTERDAY_PLANFACT,
    INTENT_GET_GENERATION_PERIOD,
    INTENT_GET_FORECAST_PERIOD,
    INTENT_GET_PLANFACT_PERIOD,
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


_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)

_STATION_STOP_WORDS = {
    "ao",
    "id",
    "mw",
    "mvt",
    "pv",
    "ses",
    "solar",
    "station",
    "stantsiya",
    "too",
    "ао",
    "мвт",
    "сес",
    "сэс",
    "станция",
    "тоо",
}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def _station_search_text(text: str) -> str:
    normalized = _normalize_text(text).replace(",", ".")
    latin = normalized.translate(_CYRILLIC_TO_LATIN)
    latin = re.sub(r"\b(?:сес|сэс|ses)\b", " ses ", latin)
    latin = re.sub(r"[^a-z0-9.]+", " ", latin)
    return " ".join(latin.split())


def _station_search_tokens(text: str) -> list[str]:
    return [token for token in _station_search_text(text).split() if len(token) > 1]


def _important_station_tokens(station: Station) -> list[str]:
    return [
        token
        for token in _station_search_tokens(station.name)
        if token not in _STATION_STOP_WORDS and not token.replace(".", "", 1).isdigit()
    ]


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.92
    return SequenceMatcher(None, left, right).ratio()


def _station_name_score(station: Station, query_tokens: list[str], query_search_text: str) -> float:
    station_search_text = _station_search_text(station.name)
    if station_search_text and station_search_text in query_search_text:
        return 1.0

    station_tokens = _important_station_tokens(station)
    if not station_tokens or not query_tokens:
        return 0.0

    best_token_scores = [
        max(_token_similarity(station_token, query_token) for query_token in query_tokens)
        for station_token in station_tokens
    ]
    matched_scores = [score for score in best_token_scores if score >= 0.78]
    if not matched_scores:
        return 0.0

    # One distinctive station token is enough for names like "СЭС Балхаш 50 МВт"
    # when the user says "ses balhash" or just "балхаш".
    return max(matched_scores)




def _has_period_hint(normalized: str) -> bool:
    if any(phrase in normalized for phrase in ("за неделю", "за месяц", "с ", " по ")):
        if "сегодня" not in normalized and "вчера" not in normalized and "завтра" not in normalized:
            return True
    if re.search(r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{4})?\b", normalized):
        return True
    if re.search(r"\b\d{1,2}\s+[а-я]+\b", normalized):
        return True
    return False

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
    if mentions_planfact and mentions_today and not _has_period_hint(normalized):
        return INTENT_GET_TODAY_PLANFACT
    if mentions_planfact and _has_period_hint(normalized):
        return INTENT_GET_PLANFACT_PERIOD
    if mentions_generation and mentions_yesterday:
        return INTENT_GET_YESTERDAY_GENERATION
    if mentions_generation:
        return INTENT_GET_GENERATION_PERIOD
    if mentions_forecast:
        return INTENT_GET_FORECAST_PERIOD
    if mentions_planfact:
        return INTENT_GET_PLANFACT_PERIOD
    return None




@dataclass(frozen=True)
class StationResolution:
    station: Optional[Station]
    needs_clarification: bool


def _resolve_station(text: str, user) -> StationResolution:
    stations = list(_station_queryset_for_user(user).order_by("sort_order", "id"))
    if not stations:
        return StationResolution(station=None, needs_clarification=False)

    explicit_id = re.search(r"(?:станци(?:я|и|ю)|station|id)\s*#?\s*(\d+)(?![\.,]\d)", text, flags=re.IGNORECASE)
    if explicit_id:
        station_id = int(explicit_id.group(1))
        for station in stations:
            if station.pk == station_id:
                return StationResolution(station=station, needs_clarification=False)

    normalized = _normalize_text(text)
    query_search_text = _station_search_text(text)
    query_tokens = _station_search_tokens(text)
    for station in stations:
        if _normalize_text(station.name) in normalized or _station_search_text(station.name) in query_search_text:
            return StationResolution(station=station, needs_clarification=False)

    decimal_values = re.findall(r"\d+(?:[\.,]\d+)?", text)
    for raw_value in decimal_values:
        value = raw_value.replace(",", ".")
        for station in stations:
            if value in _station_search_text(station.name):
                return StationResolution(station=station, needs_clarification=False)
            try:
                if abs(float(value) - float(station.capacity_mw)) < 0.01:
                    return StationResolution(station=station, needs_clarification=False)
            except (TypeError, ValueError):
                continue

    scored_stations = [
        (_station_name_score(station, query_tokens, query_search_text), station)
        for station in stations
    ]
    best_score, best_station = max(scored_stations, key=lambda item: item[0])
    if best_score >= 0.78:
        return StationResolution(station=best_station, needs_clarification=False)

    if len(stations) == 1:
        return StationResolution(station=stations[0], needs_clarification=False)
    return StationResolution(station=None, needs_clarification=True)


def _format_kwh(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "н/д"
    return f"{value:.1f}%"




def _answer_for_period_intent(intent: str, station_id: int, text: str) -> str:
    period = parse_period(text)
    if intent == INTENT_GET_GENERATION_PERIOD:
        result = services.get_generation_for_period(station_id, period.date_from, period.date_to)
        if result.points_count == 0:
            return f"За период {period.label} нет данных выработки по {result.station_name}."
        return f"Выработка за период {period.label} по {result.station_name}: {_format_kwh(result.energy_kwh)} кВт·ч (точек: {result.points_count})."
    if intent == INTENT_GET_FORECAST_PERIOD:
        result = services.get_forecast_for_period(station_id, period.date_from, period.date_to)
        if result.points_count == 0:
            return f"За период {period.label} нет сохраненного прогноза по {result.station_name}."
        return f"Прогноз выработки за период {period.label} по {result.station_name}: {_format_kwh(result.energy_kwh)} кВт·ч (точек: {result.points_count})."
    result = services.get_planfact_for_period(station_id, period.date_from, period.date_to)
    return (
        f"План/факт за период {period.label} по {result.station_name}: факт {_format_kwh(result.fact_kwh)} кВт·ч, "
        f"план {_format_kwh(result.plan_kwh)} кВт·ч, отклонение {_format_kwh(result.deviation_kwh)} кВт·ч "
        f"({_format_percent(result.deviation_percent)}), точек: {result.points_count}."
    )

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
        station_resolution = _resolve_station(text, request.user)
        station = station_resolution.station
        station_id = station.pk if station else None

        if intent is None:
            return JsonResponse(
                {
                    "answer": "Пока я умею отвечать только по выработке, прогнозу и план/факту за сегодня, вчера или завтра.",
                    "action": None,
                },
                status=400,
            )
        if station_resolution.needs_clarification:
            return JsonResponse({"answer": "Уточните станцию: у вас несколько станций, а в вопросе не удалось однозначно определить нужную.", "action": None}, status=400)
        if station is None:
            return JsonResponse({"answer": "Не найдена доступная солнечная станция для вашего пользователя.", "action": None}, status=404)

        if intent in READ_INTENTS:
            if intent in {INTENT_GET_GENERATION_PERIOD, INTENT_GET_FORECAST_PERIOD, INTENT_GET_PLANFACT_PERIOD}:
                answer = _answer_for_period_intent(intent, station.pk, text)
            else:
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
