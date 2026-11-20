import math
import pytest

from dpa4_xray_hrmc.rmc.phase3 import KB_EV_K, inline_delta, metropolis_accept


def test_inline_zero_energy_reduces_to_phase25():
    assert inline_delta(2.5, 0.0, 300.0) == 2.5


def test_inline_uses_total_energy_and_kbt_once():
    assert inline_delta(1.0, KB_EV_K * 300.0, 300.0) == pytest.approx(2.0)


def test_metropolis_boundaries_and_overflow_safety():
    assert metropolis_accept(-1.0, 0.999)[0]
    accepted, probability = metropolis_accept(2.0, math.exp(-2.0) * 0.5)
    assert accepted and probability == pytest.approx(math.exp(-2.0))
    assert not metropolis_accept(1000.0, 0.5)[0]

