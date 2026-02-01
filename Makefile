.PHONY: help setup data-check train infer clean

help:
	@echo "Targets:"
	@echo "  make setup       - create venv + install deps (Windows: uses python)"
	@echo "  make data-check  - run loader + alignment sanity checks"
	@echo "  make train       - train with train/valid split and save model"
	@echo "  make infer       - run inference using saved model"
	@echo "  make clean       - remove artifacts"

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

data-check:
	python -m src.data_loader
	python -m src.align

train:
	python -m src.train

infer:
	python -m src.infer --model artifacts/fin_ner_model --text 'Microsoft reported revenue of $$85.2B in Q3 2024.'

clean:
	@if exist artifacts rmdir /s /q artifacts
	@if exist __pycache__ rmdir /s /q __pycache__
