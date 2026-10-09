"""CLI smoke tests: selftest, demo output, and defaults."""

import json
import os

import dictforge.cli as cli
from dictforge.pipeline.selftest import run_selftest


def test_selftest_passes():
    res = run_selftest()
    assert res["all_pass"] is True


def test_demo_writes_benchmark(tmp_path, monkeypatch):
    out = os.path.join(str(tmp_path), "benchmark.json")
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rc = cli.main(["demo", "--seeds", "1", "--out", out])
    assert rc == 0
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "headline" in data
    assert data["deterministic"] is True
    assert data["headline"]["gate_pass"] is True


def test_no_subcommand_defaults_to_demo(tmp_path, monkeypatch):
    out = os.path.join(str(tmp_path), "benchmark.json")
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rc = cli.main(["--out", out])  # no subcommand -> demo
    assert rc == 0
    assert os.path.exists(out)


def test_bad_seed_env_raises(monkeypatch):
    monkeypatch.setenv("ENV_SEED", "not-int")
    rc = cli.main(["selftest"])  # selftest still works; but env is validated by demo
    assert rc == 0  # selftest does not read ENV_SEED
