FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install fastapi uvicorn[standard] pandas scikit-learn jinja2 python-multipart

EXPOSE 2003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2003"]
