import datetime as dt
import glob
import json
import os

import pytest

from device import calc

HERE = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def params():
    out = {}
    for f in glob.glob(os.path.join(HERE, "..", "results", "precomputed", "calc_*.json")):
        e = json.load(open(f))
        out[e["coordinate"]] = e["params"]
    return out


D = dt.date.fromisoformat


def test_ud_qualifying_s97_extension(params):
    # dismissed without notice 3 days short of two years: s97(2) adds a week
    f = dict(start_date=D("2024-06-01"), termination_date=D("2026-05-28"), notice_given=False)
    assert calc.run("ud_qualifying", f, params)[0] == "yes"
    f = dict(start_date=D("2024-10-05"), termination_date=D("2026-03-31"), notice_given=False)
    assert calc.run("ud_qualifying", f, params)[0] == "no"


def test_time_limit(params):
    f = dict(termination_date=D("2026-02-02"), advice_date=D("2026-05-05"))
    ans, deadline, _ = calc.run("time_limit", f, params)
    assert (ans, deadline) == ("no", "2026-05-01")


def test_compensation_cap(params):
    assert calc.run("compensation_cap", dict(annual_pay=182000.0), params)[1] == 123543
    assert calc.run("compensation_cap", dict(annual_pay=41600.0), params)[1] == 41600


def test_redundancy_amount(params):
    f = dict(start_date=D("2020-07-01"), termination_date=D("2026-07-31"), notice_given=True,
             notice_given_date=D("2026-07-03"), age=30, weekly_pay=500.0)
    assert calc.run("redundancy_amount", f, params)[1] == 3000


def test_notice_without_verified_date_refuses(params):
    f = dict(start_date=D("2020-07-01"), termination_date=D("2026-07-31"), notice_given=True)
    with pytest.raises(calc.Missing):
        calc.run("relevant_date", f, params)
