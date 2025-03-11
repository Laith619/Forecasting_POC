#!/bin/bash

# Stop and remove existing container if it exists
echo "Checking for existing containers..."
if [ "$(docker ps -aq -f name=forecasting-app)" ]; then
    echo "Stopping and removing existing container..."
    docker stop forecasting-app
    docker rm forecasting-app
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t forecasting-app:latest .

# Run the container
echo "Running container..."
docker run -d \
    --name forecasting-app \
    -p 8501:8501 \
    forecasting-app:latest

echo "Container is starting..."
echo "You can access the application at http://localhost:8501"
echo "View logs with: docker logs -f forecasting-app" 