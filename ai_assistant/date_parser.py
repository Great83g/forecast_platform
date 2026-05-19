from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from django.utils import timezone

_MONTHS = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "ма": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}


@dataclass(frozen=True)
class ParsedPeriod:
    date_from: date
    date_to: date
    label: str


def _today() -> date:
    return timezone.localdate()


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_period(text: str) -> ParsedPeriod:
    normalized = " ".join((text or "").lower().replace("ё", "е").split())
    today = _today()

    if "вчера" in normalized:
        target = today - timedelta(days=1)
        return ParsedPeriod(target, target, f"{target:%d.%m.%Y}")
    if "завтра" in normalized:
        target = today + timedelta(days=1)
        return ParsedPeriod(target, target, f"{target:%d.%m.%Y}")
    if "сегодня" in normalized:
        return ParsedPeriod(today, today, f"{today:%d.%m.%Y}")

    if "за последние 7" in normalized or "последние 7 дней" in normalized or "за неделю" in normalized:
        date_from = today - timedelta(days=6)
        return ParsedPeriod(date_from, today, f"{date_from:%d.%m.%Y}–{today:%d.%m.%Y}")

    if "с начала месяца" in normalized or "за месяц" in normalized:
        date_from = today.replace(day=1)
        return ParsedPeriod(date_from, today, f"{date_from:%d.%m.%Y}–{today:%d.%m.%Y}")

    if "за прошлый месяц" in normalized:
        first_this_month = today.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        first_prev_month = last_prev_month.replace(day=1)
        return ParsedPeriod(first_prev_month, last_prev_month, f"{first_prev_month:%d.%m.%Y}–{last_prev_month:%d.%m.%Y}")

    range_match = re.search(r"с\s*(\d{1,2})\s*по\s*(\d{1,2})\s*([а-я]+)", normalized)
    if range_match:
        day_from = int(range_match.group(1))
        day_to = int(range_match.group(2))
        month_word = range_match.group(3)
        month = next((m for k, m in _MONTHS.items() if month_word.startswith(k)), None)
        if month is not None:
            d1 = _safe_date(today.year, month, day_from)
            d2 = _safe_date(today.year, month, day_to)
            if d1 and d2:
                if d2 < d1:
                    d1, d2 = d2, d1
                return ParsedPeriod(d1, d2, f"{d1:%d.%m.%Y}–{d2:%d.%m.%Y}")

    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", normalized)
    if iso_match:
        d = _safe_date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        if d:
            return ParsedPeriod(d, d, f"{d:%d.%m.%Y}")

    full_match = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", normalized)
    if full_match:
        d = _safe_date(int(full_match.group(3)), int(full_match.group(2)), int(full_match.group(1)))
        if d:
            return ParsedPeriod(d, d, f"{d:%d.%m.%Y}")

    short_match = re.search(r"\b(\d{1,2})[./](\d{1,2})\b", normalized)
    if short_match:
        d = _safe_date(today.year, int(short_match.group(2)), int(short_match.group(1)))
        if d:
            return ParsedPeriod(d, d, f"{d:%d.%m.%Y}")

    month_match = re.search(r"\b(\d{1,2})\s+([а-я]+)\b", normalized)
    if month_match:
        day = int(month_match.group(1))
        month_word = month_match.group(2)
        month = next((m for k, m in _MONTHS.items() if month_word.startswith(k)), None)
        if month is not None:
            d = _safe_date(today.year, month, day)
            if d:
                return ParsedPeriod(d, d, f"{d:%d.%m.%Y}")

    return ParsedPeriod(today, today, f"{today:%d.%m.%Y}")
