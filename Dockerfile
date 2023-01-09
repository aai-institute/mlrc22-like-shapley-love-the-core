FROM rayproject/ray:2.2.0-py310-cpu as BASE-IMAGE

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    HOME=/home/ray

RUN mkdir $HOME/code

WORKDIR $HOME/code

RUN pip install poetry==1.2.0

COPY README.md poetry.lock pyproject.toml ./
COPY src/ ./src/

RUN poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi

FROM rayproject/ray:2.2.0-py310-cpu as MAIN-IMAGE

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/ray \
    VENV_DIR=/home/ray/code/.venv

RUN mkdir $HOME/code
WORKDIR $HOME/code/

COPY --from=BASE-IMAGE $VENV_DIR $VENV_DIR
COPY src/ ./src/

ENV PATH="$VENV_DIR/bin:$PATH"
