# Use the updated base image with the model cached
FROM zero-shot-object-detection-base-with-model

# Copy the local project files into the container
COPY . /app

# Expose the streamlit port
EXPOSE 8501

# Set the command to run Streamlit
CMD ["streamlit", "run", "./app.py"]