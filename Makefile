.PHONY: install install-all fmt lint nb

install:
	python3 -m pip install -U pip
	pip install -r requirements.txt
	pre-commit install || true

install-all:
	python3 -m pip install -U pip
	pip install -r requirements/all.txt
	pre-commit install || true

fmt:
	black src scripts
	isort src scripts

lint:
	flake8 src scripts

nb:
	jupyter lab


