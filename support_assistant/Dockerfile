FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 --retries=5 -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
