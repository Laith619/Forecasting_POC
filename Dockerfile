# Use a Python base image with optimization for numerical libraries
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    libssl-dev \
    # Add dependencies that help with numerical package builds
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with optimized approach
RUN pip install --upgrade pip && \
    # Install core dependencies with binary wheels where possible
    pip install numpy==1.24.3 pandas==1.5.3 scipy==1.10.1 && \
    # Install Prophet
    pip install prophet && \
    # Install pmdarima with binary wheel if possible (removing no-binary flag)
    pip install pmdarima==2.0.3 && \
    # Then install remaining packages
    pip install -r requirements.txt

# Copy the application code
COPY . .

# Create a script to start Streamlit with the correct port
RUN echo '#!/bin/bash\n\
PORT="${PORT:-8501}"\n\
streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose the port that will be used (handled by the PORT environment variable in Cloud Run)
EXPOSE 8080

# Add a healthcheck that uses the PORT environment variable
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8080}/_stcore/health || exit 1

# Set the entry point to run the application using our wrapper script
CMD ["/app/start.sh"]