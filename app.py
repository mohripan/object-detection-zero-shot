import streamlit as st
import requests
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import os
import io
import replicate
import base64
from dotenv import load_dotenv
import requests

load_dotenv()

# Load the model and processor
checkpoint = "google/owlvit-base-patch32"

def load_model_and_processor():
    print("Function load_model_and_processor called!")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor

model, processor = load_model_and_processor()

# Fetch API token from environment variable
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
IMGBB_API_KEY = os.environ.get('IMGBB_KEY')

if REPLICATE_API_TOKEN is None:
    raise ValueError("REPLICATE_API_TOKEN environment variable not set")

if IMGBB_API_KEY is None:
    raise ValueError("IMGBB_KEY environment variable not set")


MAX_IMAGE_DIMENSION = 400

def resize_image(image, max_dimension=MAX_IMAGE_DIMENSION):
    """Resizes the input PIL Image to the specified max_dimension while maintaining aspect ratio."""
    w, h = image.size
    if max(w, h) > max_dimension:
        if w > h:
            new_w = max_dimension
            new_h = int((new_w / w) * h)
        else:
            new_h = max_dimension
            new_w = int((new_h / h) * w)
        image = image.resize((new_w, new_h))
    return image

def detect_objects(im, text_queries):
    inputs = processor(text=text_queries, images=im, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    draw = ImageDraw.Draw(im)
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    boxes = results["boxes"].tolist()

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

    return im

def detect_one_shot(target_image, query_image):
    inputs = processor(images=target_image, query_images=query_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
        target_sizes = torch.tensor([target_image.size[::-1]])
        results = processor.post_process_image_guided_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)[0]

    draw = ImageDraw.Draw(target_image)
    scores = results["scores"].tolist()
    boxes = results["boxes"].tolist()
    
    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)
        draw.text((xmin, ymin), f"{round(score,2)}", fill="white")

    return target_image, boxes


def remove_background(img_url):
    output = replicate.run(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        input={"image": img_url}
    )
    return output

def upload_image_to_imgbb(image_data):
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": IMGBB_API_KEY,
        "image": base64.b64encode(image_data).decode("utf-8"),
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json()["data"]["url"]
    else:
        raise Exception("Error uploading image to imgbb")


# Navigation bar
st.sidebar.title("Object Detection Type")
option = st.sidebar.radio("", ["Zero-Shot Object Detection", "One-Shot Object Detection"])

# Threshold slider
threshold = st.sidebar.slider('Detection Threshold', 0.0, 1.0, 0.1)

st.title(option)

if option == "Zero-Shot Object Detection":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image = resize_image(image)
        queries = st.text_input("Input the objects you want to detect (comma separated)", "")
        text_queries = [q.strip() for q in queries.split(",")]
        
        col1, col2 = st.columns(2)
        col1.header("Uploaded Image")
        col1.image(image, use_column_width=True)

        if st.button("Detect"):
            
            detected_img = detect_objects(image, text_queries)  # Make sure to adjust the detect_objects function to use the threshold slider value
            col2.header("Detected Objects")
            col2.image(detected_img, use_column_width=True)

else:  # One-Shot Object Detection
    st.write("In One-Shot Object Detection, you provide an example image (query image) of the object you want to detect and then provide another image (target image) where you want to detect the object.")
    
    st.write("### Upload Example Image (Query Image)")
    query_file = st.file_uploader("Upload example image of the object you want to detect", type=['jpg', 'png'])

    st.write("### Target Image")
    target_file = st.file_uploader("Upload the image where you want to detect the object", type=['jpg', 'png'])
    
    if target_file and query_file:
        query_image = Image.open(query_file)
        query_image = resize_image(query_image)
        target_image = Image.open(target_file)
        target_image = resize_image(target_image)

        col1, col2 = st.columns(2)
        col1.header("Example Image (Query Image)")
        col1.image(query_image, use_column_width=True)
        
        col2.header("Target Image")
        col2.image(target_image, use_column_width=True)

        # Add checkbox for removing background
        remove_bg = st.checkbox("Remove background from target image")

        if st.button("Detect"):
            # Check if remove_bg is selected by user
            if remove_bg:
                # Convert the PIL image to bytes
                byte_io = io.BytesIO()
                query_image.save(byte_io, format="JPEG")
                img_bytes = byte_io.getvalue()
                
                # Upload the image bytes to imgbb
                imgbb_url = upload_image_to_imgbb(img_bytes)

                # Remove background
                bg_removed_uri = remove_background(imgbb_url)
                response = requests.get(bg_removed_uri)
                query_image = Image.open(io.BytesIO(response.content))
                    
                if query_image.mode != "RGB":
                    query_image = query_image.convert("RGB")
                        
                col1.header("Example Image (Query Image)")
                col1.image(query_image, use_column_width=True)
                
            detected_img, detected_boxes = detect_one_shot(target_image.copy(), query_image)
            boxes = [(box[0], box[1], box[2], box[3]) for box in detected_boxes]

            draw = ImageDraw.Draw(target_image)
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)

            st.header("Detected Objects in Target Image")
            st.image(target_image, use_column_width=True)
            
# Footer
st.markdown("""
--- 
Created by Mohammad Ripan Saiful Mansur
""")