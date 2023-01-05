# The same version to google colab.
FROM python:3.8.16

RUN mkdir /workspace
WORKDIR /workspace

RUN apt update
RUN apt install -y fluidsynth make bash-completion

RUN echo "[[ $PS1 && -f /usr/share/bash-completion/bash_completion ]] && \
    . /usr/share/bash-completion/bash_completion" >> ~/.bashrc

RUN pip install -U pip

RUN pip install poetry
RUN poetry config virtualenvs.create false
COPY pyproject.toml /workspace/pyproject.toml
RUN poetry install
