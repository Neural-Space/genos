# Setup the environment

SYSTEM_DEPENDENCIES := poetry==1.1.3 pre-commit coveralls flake8

.PHONY: check-py3
check-py3:
	./utility-scripts/check_python37.sh

.PHONY: install-system-deps
install-system-deps:
	pip install -U $(SYSTEM_DEPENDENCIES)


.PHONY: install-system-deps-user
install-system-deps-user:
	pip install --user -U $(SYSTEM_DEPENDENCIES)

## To install system level dependencies
.PHONY: bootstrap
bootstrap: check-py3 install-system-deps
	curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

## Install system dependencies in user dir (Linux)
.PHONY: bootstrap-user
bootstrap-user: check-py3 install-system-deps-user
	curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

.PHONY: bootstrap-mac
bootstrap-mac: check-py3 install-system-deps
	brew install azure-cli

## Install system dependencies in user dir (Linux)
.PHONY: bootstrap-user-mac
bootstrap-user-mac: check-py3 install-system-deps-user
	brew install azure-cli

## Setup poetry
.PHONY: poetry-setup
poetry-setup:
	poetry config virtualenvs.in-project true
	poetry run pip install pip==20.0.2
	poetry install --no-root
	poetry run pip install azure-keyvault-secrets azure.identity
	poetry install

## Setup pre-commit
.PHONY: pre-commit-setup
pre-commit-setup:
	pre-commit install


# Setup virtual environment and dependencies
.PHONY: install
install: pre-commit-setup poetry-setup

# Environment setup for Conda
.PHONY: conda-env-setup
conda-env-setup:
	mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
	mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
	touch ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
	touch ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
	echo "export CONDA_OLD_PATH=${PATH}" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
	echo "export PATH=${HOME}/.local/bin:${PATH}" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
	echo "unset PATH" >> ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
	echo "export PATH=${CONDA_OLD_PATH}" >> ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh

# Format code
.PHONY: format
format:
	# calling make _format within poetry make it so that we only init poetry once
	poetry run isort -rc -y src/my_package tests
	poetry run black src/my_package tests


# Flake8 to check code formatting
.PHONY: lint
lint:
	poetry run flake8 src/my_package tests

N_THREADS=1
# Run tests
.PHONY: test
test:
	PYTHONPATH='./src/' poetry run pytest tests/ -s -n ${N_THREADS} -vv

# Run coverage
.PHONY: coverage
coverage:
	PYTHONPATH='./src/' poetry run coverage run --concurrency=multiprocessing -m pytest tests/ -s
	poetry run coverage combine
	poetry run coverage report -m


# Run tests and coverage
.PHONY: test-coverage
test-coverage: test coverage
