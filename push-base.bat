docker build -f Dockerfile.base -t zero-shot-object-detection-base .
docker tag zero-shot-object-detection-base:latest mohripan/zero-shot-object-detection-base:latest
docker push mohripan/zero-shot-object-detection-base:latest