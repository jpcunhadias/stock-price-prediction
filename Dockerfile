FROM python:3.10-slim

WORKDIR /app

COPY 'src/api/' .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]