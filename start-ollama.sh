#!/bin/bash

# Start Ollama service
ollama serve &

# Wait for Ollama to fully start up
echo "Waiting for Ollama to start..."
until curl -s --head http://localhost:11434/api/version | grep "200 OK" > /dev/null; do 
  echo "Waiting for Ollama API to become available..."
  sleep 1
done
echo "Ollama service started!"

# List of models to download - add any additional models you need
echo "Starting model downloads..."
ollama pull deepseek-r1:8b
# Add more models if needed:
# ollama pull llama2
# ollama pull codellama:7b
echo "Model downloads completed!"

# Keep the script running to keep the container alive
wait