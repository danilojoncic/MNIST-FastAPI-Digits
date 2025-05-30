# Stage 1: Build dependencies
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache build-base

# Set working directory
WORKDIR /app

# Install pipenv or just use requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-deps -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-alpine

WORKDIR /app

# Install runtime dependencies only
RUN apk add --no-cache libstdc++

# Copy installed wheels from builder
COPY --from=builder /app /app
RUN pip install --no-cache-dir *.whl && rm -f *.whl

# Copy app code
COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
