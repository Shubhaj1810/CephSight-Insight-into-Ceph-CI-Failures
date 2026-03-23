FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY tests/ ./tests/

# Default: run the analyzer CLI
ENTRYPOINT ["python3", "main.py"]
