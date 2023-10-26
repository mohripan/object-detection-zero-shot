# Use the updated base image with the model cached
FROM zero-shot-object-detection-base-with-model

# Copy the local project files into the container
COPY . /app

# Set the command to run your application
CMD ["python", "./app.py"]