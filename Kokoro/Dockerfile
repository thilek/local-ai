# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    curl \
    gnupg2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify espeak-ng installation
RUN espeak-ng --version

# Download the model file using curl
RUN curl -L --retry 5 --output /app/kokoro-v0_19.pth \
    https://huggingface.co/geneing/Kokoro/resolve/f610f07c62f8baa30d4ed731530e490230e4ee83/kokoro-v0_19.pth

# Validate the file is downloaded
RUN ls -lh /app/kokoro-v0_19.pth

# Copy requirements file to cache dependencies
COPY requirements.txt /app/

# Create a virtual environment and set PATH
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn phonemizer torch transformers scipy munch soundfile nltk

# Install NLTK data for tokenization and ensure all required data is available
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" && \
    mkdir -p /app/.venv/nltk_data && \
    mv /root/nltk_data/* /app/.venv/nltk_data/

# Copy the rest of the application files
COPY . /app/

# Add a non-root user
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8880

# Command to run the application
CMD ["uvicorn", "kokoro_service:app", "--host", "0.0.0.0", "--port", "8880"]
