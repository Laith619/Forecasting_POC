#!/bin/bash

# Check if docker-compose or docker compose is available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "Neither docker-compose nor docker compose found. Please install Docker Compose."
    exit 1
fi

# Stop any existing services
echo "Stopping any existing services..."
$DOCKER_COMPOSE down

# Build and start the services
echo "Building and starting services..."
$DOCKER_COMPOSE up -d --build

echo "Services are starting..."
echo "You can access the application at http://localhost:8501"
echo "View logs with: $DOCKER_COMPOSE logs -f"
echo "Stop services with: $DOCKER_COMPOSE down" 