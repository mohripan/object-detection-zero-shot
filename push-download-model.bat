docker run --name temp_container zero-shot-object-detection-base python download_model.py
docker commit temp_container zero-shot-object-detection-base-with-model
docker tag zero-shot-object-detection-base-with-model:latest mohripan/zero-shot-object-detection-base-with-model:latest
docker push mohripan/zero-shot-object-detection-base-with-model:latest
docker rm temp_container