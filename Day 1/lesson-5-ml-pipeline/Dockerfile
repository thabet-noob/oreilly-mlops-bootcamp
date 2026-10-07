# Use an official Python runtime as a parent image
# (must match the Python/library versions used to train the model in model/)
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies first so this layer is cached between code changes
# Optional: trust an extra CA (e.g. corporate TLS inspection) passed with --secret id=corp_ca,src=<pem>
COPY requirements.txt .
RUN --mount=type=secret,id=corp_ca \
    if [ -f /run/secrets/corp_ca ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/corp_ca > /tmp/ca.pem && export PIP_CERT=/tmp/ca.pem; \
    fi && \
    pip install --no-cache-dir -r requirements.txt && rm -f /tmp/ca.pem

# Copy the application code and trained model (see .dockerignore for what is excluded)
COPY . .

# Listen on all interfaces so the app is reachable from outside the container
ENV HOST=0.0.0.0

# Expose port for Flask app
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
