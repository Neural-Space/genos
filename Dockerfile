FROM python:3.7.7-stretch AS BASE

RUN apt-get update \
    && apt-get --assume-yes --no-install-recommends install \
        build-essential \
        curl \
        git \
        jq \
        libgomp1 \
        vim

WORKDIR /app

# upgrade pip version
RUN pip install --no-cache-dir --upgrade pip==20.1.1

# install poetry
RUN pip install poetry==1.1.3

# config poetry
RUN poetry config virtualenvs.in-project true
RUN poetry config virtualenvs.create true

# install python dependencies
ADD pyproject.toml .
ADD poetry.lock .
ADD src/my_package/__init__.py my_package/__init__.py

RUN poetry run pip install pip==20.0.2
RUN poetry install --no-dev

ADD Makefile .

RUN rm -rf /root/.cache/pip \
    && rm -rf /root/.cache/pypoetry/cache

ADD src/my_package ./my_package
ADD utility-scripts ./utility-scripts

FROM BASE as DEV

RUN poetry install
ADD tests ./tests
RUN make test