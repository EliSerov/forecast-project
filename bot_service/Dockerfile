FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml ./

RUN uv lock && uv sync

COPY src/ ./src/

CMD ["uv", "run", "python", "src/bot.py"]
