"""
Unit test cho Dual-Tier Edge Gate (src/decision/edge_gate.py)
"""
import json
import pytest
from pathlib import Path
from src.decision.edge_gate import EdgeGate


def test_edge_gate_initialization(tmp_path):
    policy_file = tmp_path / "evaluation_policy.json"
    gate = EdgeGate(policy_path=policy_file)
    res = gate.check()
    
    assert res["pass"] is False
    assert res["status"] == "PENDING"
    assert res["action"] == "PAPER_TRADE"


def test_edge_gate_tier1_pass_tier2_fail(tmp_path):
    policy_file = tmp_path / "evaluation_policy.json"
    gate = EdgeGate(policy_path=policy_file)

    # Tier 1 PASS (delta_brier_upper_95 < 0, ECE <= 0.0800), Tier 2 FAIL (roi_lower_95 <= 0)
    res = gate.update(
        roi_lower_95=-0.05,
        evaluation_period="2025-01-01 to 2026-07-25",
        n_days=180,
        total_bets=100,
        total_hits=25,
        roi=-0.02,
        delta_brier_upper_95=-0.0125,
        ece_score=0.0450,
        brier_score=0.1820
    )

    assert res["pass"] is False
    assert res["tier1_pass"] is True
    assert res["tier2_pass"] is False
    assert res["status"] == "TIER1_PASS"
    assert res["action"] == "PAPER_TRADE_APPROVED"


def test_edge_gate_both_tiers_pass(tmp_path):
    policy_file = tmp_path / "evaluation_policy.json"
    gate = EdgeGate(policy_path=policy_file)

    # Tier 1 PASS and Tier 2 PASS (roi_lower_95 > 0)
    res = gate.update(
        roi_lower_95=0.035,
        evaluation_period="2025-01-01 to 2026-07-25",
        n_days=180,
        total_bets=100,
        total_hits=35,
        roi=0.12,
        delta_brier_upper_95=-0.0150,
        ece_score=0.0350,
        brier_score=0.1780
    )

    assert res["pass"] is True
    assert res["tier1_pass"] is True
    assert res["tier2_pass"] is True
    assert res["status"] == "PASS"
    assert res["action"] == "BET"


def test_edge_gate_tier1_fail(tmp_path):
    policy_file = tmp_path / "evaluation_policy.json"
    gate = EdgeGate(policy_path=policy_file)

    # Tier 1 FAIL (delta_brier_upper_95 >= 0)
    res = gate.update(
        roi_lower_95=-0.10,
        evaluation_period="2025-01-01 to 2026-07-25",
        n_days=180,
        total_bets=100,
        total_hits=20,
        roi=-0.08,
        delta_brier_upper_95=0.0050,  # CI95 contains 0 -> FAIL
        ece_score=0.0950,
        brier_score=0.2310
    )

    assert res["pass"] is False
    assert res["tier1_pass"] is False
    assert res["status"] == "FAIL"
    assert res["action"] == "PAPER_TRADE"
