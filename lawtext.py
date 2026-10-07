# from blind-counsel 8b1bf8f lawtext.py
"""Read numbers, dates and money out of legal text, and do calendar arithmetic.

Used on both sides: the cloud's central audit checks that every precomputed
parameter's value really appears in its statutory quote, and the device checks
that every extracted client fact really appears in the client's own words.
"""
from __future__ import annotations

import datetime as dt
import re

_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def numbers_in(text: str) -> set[float]:
    """Every number written in the text, in digits, money or words."""
    t = (text or "").lower().replace("’", "'")
    out: set[float] = {float(m.group(1).replace(",", ""))
                       for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)", t)}
    if "one and a half" in t:
        out.add(1.5)
    if re.search(r"\bhalf a\b", t):
        out.add(0.5)
    for m in re.finditer(r"\b(twenty|thirty|forty|fifty|sixty)-(one|two|three|four|five|six|seven|eight|nine)\b", t):
        out.add(float(_TENS[m.group(1)] + _UNITS[m.group(2)]))
    for w, v in {**_UNITS, **_TENS}.items():
        if re.search(rf"\b{w}\b", t):
            out.add(float(v))
    return out


def dates_in(text: str) -> list[dt.date]:
    """Dates written like '4 August 2023' or '4th August, 2023'."""
    out = []
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})\b", text or ""):
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            try:
                out.append(dt.date(int(m.group(3)), mon, int(m.group(1))))
            except ValueError:
                pass
    return out


def amounts_in(text: str) -> list[float]:
    return [float(m.group(1).replace(",", "")) for m in re.finditer(r"£\s?(\d[\d,]*(?:\.\d+)?)", text or "")]


def add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(y, m, min(d.day, last))


def period_end(start: dt.date, months: int) -> dt.date:
    """Last day of a period of `months` months beginning with `start`: the day before the
    corresponding date, or -- where that month has no corresponding date (e.g. 31 June)
    -- the last day of that month."""
    target = add_months(start, months)
    return target if target.day != start.day else target - dt.timedelta(days=1)


def complete_years(start: dt.date, end: dt.date) -> int:
    """Complete years of employment from `start` up to and including `end`."""
    y = 0
    while period_end(start, 12 * (y + 1)) <= end:
        y += 1
    return y
