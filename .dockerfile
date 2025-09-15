# This file forces Docker detection
FROM python:3.11-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8050
CMD ["python", "simple_forecast_app.py"]
