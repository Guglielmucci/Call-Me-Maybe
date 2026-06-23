.PHONY: install run run-verbose debug clean lint lint-strict test run-bonus run-bonus-verbose help

install:
	uv sync

run:
	uv run python -m src $(ARGS)

run-verbose:
	uv run python -m src --verbose

run-bonus:
	uv run python -m src --model Qwen25

run-bonus-verbose:
	uv run python -m src --model Qwen25 --verbose

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	rm -rf .pytest_cache .mypy_cache

lint:
	uv run flake8 src/ tests/ --exclude .venv,llm_sdk
	uv run mypy src/ tests/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/ tests/ --exclude .venv,llm_sdk
	uv run mypy src/ tests/ --strict

test:
	uv run pytest

help:
	@echo "Makefile commands:"
	@echo "  install            - Installa le dipendenze con uv sync"
	@echo "  run                - Esegue il modulo src (usa ARGS=... per passare argomenti)"
	@echo "  run-verbose        - Esecuzione in modalità verbosa"
	@echo "  run-bonus          - Esegue con modello Qwen25"
	@echo "  run-bonus-verbose  - Esecuzione con Qwen25 in modalità verbosa"
	@echo "  debug              - Avvia il debugger pdb sul modulo src"
	@echo "  clean              - Rimuove cache e directory __pycache__, .pytest_cache, .mypy_cache"
	@echo "  lint               - Linting con flake8 e mypy (opzioni moderate)"
	@echo "  lint-strict        - Linting con mypy --strict"
	@echo "  test               - Esegue pytest (senza coverage)"
	@echo "  help               - Mostra questa mappa"
