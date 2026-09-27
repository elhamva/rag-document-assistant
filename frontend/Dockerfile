FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/frontend/requirements.txt

COPY frontend /app/frontend

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "frontend/app.py"]
