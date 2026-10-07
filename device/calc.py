# from blind-counsel 8b1bf8f device/calc.py
"""Device-side deterministic calculators (local; no model does arithmetic).

Inputs are only:
  * client facts the device model extracted AND that were verified against the
    client's own words (the date/amount must appear in the quoted sentence), and
  * statutory parameters precomputed by the cloud AND re-verified on the device
    against the PIR-fetched statute (the number must appear in the statute quote).
If an input is missing, the calculator says so instead of guessing.
"""
from __future__ import annotations

import datetime as dt

from lawtext import complete_years, period_end

P = "uk/ukpga/1996/18/"


class Missing(Exception):
    pass


def _need(d: dict, *keys):
    for k in keys:
        if d.get(k) is None:
            raise Missing(k)
    return [d[k] for k in keys]


def _param(params: dict, sec: str, name: str) -> float:
    try:
        return params[P + sec][name]["value"]
    except KeyError:
        raise Missing(f"statutory parameter {sec}.{name}") from None


def _statutory_notice_weeks(start: dt.date, material: dt.date, params) -> int:
    yrs = complete_years(start, material)
    if yrs < 2:
        return int(_param(params, "s86", "notice_weeks_under_2y"))
    if yrs < 12:
        return int(yrs * _param(params, "s86", "notice_weeks_per_year"))
    return int(_param(params, "s86", "notice_weeks_max"))


def _material_date(f: dict) -> dt.date:
    # s97(3) / s145(6): the date notice was given, or the termination date if none
    if f.get("notice_given"):
        return _need(f, "notice_given_date")[0]
    return _need(f, "termination_date")[0]


def ud_qualifying(f, params):
    start, edt = _need(f, "start_date", "termination_date")
    material = _material_date(f)
    weeks = _statutory_notice_weeks(start, material, params)
    s97_2 = max(edt, material + dt.timedelta(weeks=weeks))         # s97(2) extension
    need = int(_param(params, "s108", "qualifying_years"))
    ok = s97_2 >= period_end(start, 12 * need)
    ext = f" (extended under s97(2) by {weeks} week(s) of unpaid statutory notice to {s97_2})" if s97_2 > edt else ""
    return ("yes" if ok else "no", None,
            f"Continuous employment from {start} to the effective date of termination {edt}{ext}: "
            f"{complete_years(start, s97_2)} complete year(s) against the {need}-year requirement in s108(1).")


def time_limit(f, params):
    edt, advice = _need(f, "termination_date", "advice_date")
    months = int(_param(params, "s111", "time_limit_months"))
    deadline = period_end(edt, months)
    presented = f.get("claim_presented_date")
    if presented:
        ok = presented <= deadline
        how = f"claim presented {presented}"
    else:
        ok = advice <= deadline
        how = (f"no claim presented yet; {(deadline - advice).days} day(s) remain at the advice date {advice}"
               if ok else f"no claim presented and the deadline passed {(advice - deadline).days} day(s) before the advice date {advice}")
    return ("yes" if ok else "no", deadline.isoformat(),
            f"Primary limit under s111(2)(a): {months} months beginning with the EDT {edt}, "
            f"i.e. on or before {deadline}; {how}. (ACAS early-conciliation extensions under s207B "
            f"and the 'not reasonably practicable' extension are not computed.)")


def compensation_cap(f, params):
    if f.get("annual_pay") is not None:
        week = f["annual_pay"] / 52
    else:
        week = _need(f, "weekly_pay")[0]
    fixed = _param(params, "s124", "cap_fixed_gbp")
    weeks = _param(params, "s124", "cap_weeks")
    cap = round(min(fixed, weeks * week))
    return ("yes", cap,
            f"s124(1ZA): the lower of £{fixed:,.0f} and {weeks:.0f} x a week's pay "
            f"(£{week:,.2f}) = £{cap:,}. This caps the compensatory award; it does not decide "
            f"whether there is a loss.")


def _relevant_dates(f, params):
    start, rd = _need(f, "start_date", "termination_date")
    material = _material_date(f)
    weeks = _statutory_notice_weeks(start, material, params)
    return start, rd, max(rd, material + dt.timedelta(weeks=weeks)), weeks


def relevant_date(f, params):
    _, rd, ext, weeks = _relevant_dates(f, params)
    more = f"; for ss.155 and 162 it is extended under s145(5) to {ext} ({weeks} weeks' statutory notice)" if ext > rd else ""
    return ("yes", rd.isoformat(), f"Relevant date under s145(2): {rd}{more}.")


def red_qualifying(f, params):
    start, _, ext, _ = _relevant_dates(f, params)
    need = int(_param(params, "s155", "qualifying_years"))
    yrs = complete_years(start, ext)
    return ("yes" if yrs >= need else "no", None,
            f"{yrs} complete year(s) of continuous employment ending with the relevant date {ext} "
            f"against the {need}-year requirement in s155.")


def redundancy_amount(f, params):
    start, _, ext, _ = _relevant_dates(f, params)
    age, week = _need(f, "age", "weekly_pay")
    if complete_years(start, ext) < int(_param(params, "s155", "qualifying_years")):
        return ("no", 0, "No entitlement under s155, so no payment.")
    g = lambda n: _param(params, "s162", n)  # noqa: E731
    limit = _param(params, "s227", "week_pay_limit_gbp")
    years = min(complete_years(start, ext), int(g("max_years")))
    weeks = 0.0
    for k in range(years):               # reckoning backwards (s162(1)(b))
        start_age = int(age) - 1 - k     # age at the start of that year, from age at relevant date
        weeks += (g("rate_upper_weeks") if start_age >= g("age_upper") else
                  g("rate_middle_weeks") if start_age >= g("age_lower") else g("rate_lower_weeks"))
    pay = min(week, limit)
    amount = round(weeks * pay, 2)
    return ("yes", amount,
            f"{years} year(s) reckoned back from {ext}: {weeks:g} weeks' pay x £{pay:,.2f} "
            f"(week's pay capped at £{limit:,.0f} by s227) = £{amount:,.2f}. Ages per year are "
            f"inferred from the stated age (date of birth not given).")


CALCULATORS = {"ud_qualifying": ud_qualifying, "time_limit": time_limit,
               "compensation_cap": compensation_cap, "relevant_date": relevant_date,
               "red_qualifying": red_qualifying, "redundancy_amount": redundancy_amount}


def run(name: str, facts: dict, params: dict):
    """Returns (answer, value, explanation) or raises Missing(<what>)."""
    return CALCULATORS[name](facts, params)
