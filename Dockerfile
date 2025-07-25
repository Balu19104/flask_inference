# Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Expose port for Render (can be any, but 10000+ is safe)
EXPOSE 5050

# Run your app
CMD ["python", "app.py"]