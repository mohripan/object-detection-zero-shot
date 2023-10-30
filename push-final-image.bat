docker build -t zero-shot-object-detection-app .
docker tag zero-shot-object-detection-app:latest mohripan/zero-shot-object-detection-app:latest
docker push mohripan/zero-shot-object-detection-app:latest