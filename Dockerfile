# Multi-stage build for production deployment
FROM python:3.11-slim as backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    pkg-config \
    libpq-dev \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Frontend build stage
FROM node:18-alpine as frontend

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Final production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies from backend stage
COPY --from=backend /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin

# Copy built frontend from frontend stage
COPY --from=frontend /app/dist ./dist

# Copy Python application
COPY *.py ./
COPY requirements.txt ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000

CMD ["gunicorn", "web_server:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300"]