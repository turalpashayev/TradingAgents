"""Tests for ta_exit deterministic P&L helpers."""

import pytest

from ta_exit.pnl import format_pnl_line, pnl_at_price


def test_pnl_at_price_loss():
    row = pnl_at_price(100.0, 92.0, quantity=10)
    assert row.pnl_per_unit == pytest.approx(-8.0)
    assert row.pnl_pct == pytest.approx(-8.0)
    assert row.pnl_total == pytest.approx(-80.0)


def test_pnl_at_price_gain():
    row = pnl_at_price(50.0, 55.0, quantity=2.0)
    assert row.pnl_pct == pytest.approx(10.0)
    assert row.pnl_total == pytest.approx(10.0)


def test_format_pnl_line_includes_total():
    line = format_pnl_line(100.0, 90.0, quantity=5.0)
    assert "-10.00" in line or "-10.0000" in line
    assert "5" in line


def test_entry_must_be_positive():
    with pytest.raises(ValueError):
        pnl_at_price(0, 10.0)
