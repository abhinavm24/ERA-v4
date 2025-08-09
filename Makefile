.PHONY: install install-all fmt lint mypy nb nbqa test test-cov check precommit precommit-run

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

mypy:
	mypy src

nbqa:
	nbqa black notebooks
	nbqa isort notebooks
	nbqa flake8 notebooks

test:
	pytest -q

test-cov:
	pytest --cov=era_v4 --cov-report=term-missing

check: fmt lint mypy nbqa test

nb:
	jupyter lab
