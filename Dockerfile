FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
ENV UV_LINK_MODE=copy UV_NO_DEV=1 PYTHONUNBUFFERED=1
RUN uv sync --locked --no-editable
EXPOSE 8501
CMD ["uv", "run", "--locked", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
