# DictForge Makefile — one-click reproducible workflow.
# Determinism: pin BLAS/LAPACK threads to 1 before any NumPy import.

PY := python
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

.PHONY: help install dev lint format test selftest demo ci clean

help:
	@echo "DictForge targets:"
	@echo "  install   install package + deps"
	@echo "  dev       install dev deps (pytest, ruff)"
	@echo "  lint      ruff check + format --check"
	@echo "  test      run unit tests"
	@echo "  selftest  run mathematical invariant self-test"
	@echo "  demo      run benchmark -> benchmark.json"
	@echo "  ci        run the full CI closure locally"
	@echo "  clean     remove caches"

install:
	pip install -e .

dev:
	pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest -q

selftest:
	$(PY) -m dictforge.cli selftest

demo:
	$(PY) -m dictforge.cli demo --out benchmark.json

ci: lint selftest test demo
	@echo "CI closure OK"

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
