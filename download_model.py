from transformers import pipeline

# This will download and cache the model
checkpoint = "google/owlvit-base-patch32"
_ = pipeline(model=checkpoint, task="zero-shot-object-detection")