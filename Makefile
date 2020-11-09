# Setup the environment

SYSTEM_DEPENDENCIES := poetry==1.1.3 pre-commit coveralls flake8

.PHONY: check-py3
check-py3:
	./utility-scripts/check_python37.sh

.PHONY: install-system-python-deps
install-system-python-deps:
	pip install -U $(SYSTEM_DEPENDENCIES)


.PHONY: install-system-python-deps
install-system-python-deps-user:
	pip install --user -U $(SYSTEM_DEPENDENCIES)

## To install system level dependencies
.PHONY: install-system-deps
install-system-deps: check-py3 install-system-python-deps


## Setup poetry
.PHONY: poetry-setup
poetry-setup:
	poetry config virtualenvs.in-project true
	poetry run pip install pip==20.0.2
	poetry install

## Setup pre-commit
.PHONY: pre-commit-setup
pre-commit-setup:
	pre-commit install


# Setup virtual environment and dependencies
.PHONY: install-deps
install-deps: pre-commit-setup poetry-setup


# Format code
.PHONY: code-format
code-format:
	# calling make _format within poetry make it so that we only init poetry once
	poetry run isort -rc -y src/genos tests
	poetry run black src/genos tests


# Flake8 to check code formatting
.PHONY: quality-check
quality-check:
	poetry run flake8 src/genos tests


# Run tests
.PHONY: unit-test
unit-test:
	PYTHONPATH='./src/' poetry run pytest tests/ -s

# Run coverage
.PHONY: coverage
coverage:
	PYTHONPATH='./src/' poetry run coverage run --concurrency=multiprocessing -m pytest tests/ -s
	poetry run coverage combine
	poetry run coverage report -m


# Run tests and coverage
.PHONY: test-coverage
test-coverage: unit-test coverage


.PHONY: verify-version-tag
verify-version-tag:
	PYTHONPATH='./src/' poetry run python setup.py verify

.PHONY: package-upload
package-upload:
	if [ -d "./build" ]; then rm -rf "./build"; fi
	if [ -d "./dist" ]; then rm -rf "./dist"; fi
	if [ -d "./src/panini.egg-info" ]; then rm -rf "./src/panini.egg-info"; fi

	PYTHONPATH=./src poetry run python setup.py sdist bdist_wheel
	twine upload dist/*

	if [ -d "./build" ]; then rm -rf ./build; fi
	if [ -d "./dist" ]; then rm -rf ./dist; fi
	if [ -d "./src/panini.egg-info" ]; then rm -rf ./src/panini.egg-info; fi