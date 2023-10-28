docker build -f Dockerfile.base -t zero-shot-object-detection-base .
docker run --name temp_container zero-shot-object-detection-base python download_model.py
docker commit temp_container zero-shot-object-detection-base-with-model
docker rm temp_container
docker build -t zero-shot-object-detection-app .
docker run --env-file .env -p 8501:8501 zero-shot-object-detection-app