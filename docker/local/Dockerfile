FROM python:3.7.6-buster

ENV PYTHONUNBUFFERED 1

# We install dev tools into the local Docker instance
COPY requirements/ /requirements
RUN pip install --upgrade pip

# Don't cache installers to reduce image size
RUN pip install --no-cache-dir -r /requirements/ray_requirements.txt

WORKDIR /app
