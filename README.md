# Template Python Project
This is a template for all python projects. It has the following:

- **A Makefile** with various helpful targets. E.g.,
  ```bash
  # to install system level dependencies
  make bootstrap
  
  # install virtual environment and project level dependencies
  make install
  
  # run unit tests
  make test
  
  # run black code formatting and isort
  make format
  
  # to run flake8 and validate code formatting
  make lint
  ```
- **A pre-commit config** to validate code formatting before commits are made.
- **A Pull Request (PR) Template** with a checklist for PRs
- A Dockerfile
- A Docker-Compose file
- A `setup.py` file in case you want to package it.
- A Coverage config in `.coveragerc`
- A `.gitignore` file
- A `.dockerignore` file
- A CircleCI config file `.circleci/config.yml`


# Project Structure

All source files go inside the `./src/my_package/`

# `PYTHONPATH` setup

- Pycharm: Mark `./src` as content root
- Others: Set this environment variable `export PYTHONPATH=./src`

# Configuring NS Private PyPi repo

Get username and password from the project Admin

```bash
poetry config http-basic.neuralspace <private-pypi-username> <private-pypi-password>
```

# Renaming this Project

Note that the name of this python package is `my_package`. Take a look at the `./src` folder.
Hence, `my_package` has been used in the following files.
- Makefile
- Dockerfile
- `.circleci/config.yml`

Make sure to rename `my_package` to `<your-package-name>` while using this template.
