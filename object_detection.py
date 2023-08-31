from pathlib import Path
import os

os.environ["TRANSFORMERS_USE_DEPRECATED_TFOD_API"] = "FALSE"

import torch

from PIL import Image, ImageDraw, ImageFont
from transformers import YolosForObjectDetection, YolosImageProcessor

root_directory = Path(__file__).parent
# print(root_directory)
picture_path = root_directory / "assets" / "bar-board-chairs-coffee-shop.jpg"
image = Image.open(picture_path)

image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(
    outputs, threshold=0.7, target_sizes=target_sizes
)[0]

draw = ImageDraw.Draw(image)
font_path = root_directory / "assets" / "OpenSans-ExtraBold.ttf"
font_path = "/home/kareem/Desktop/web_dev/fastapi_power/image_detection_wtih_fastapi/assets/OpenSans-ExtraBold.ttf"
font = ImageFont.truetype(font_path, 20)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.7:
        box_values = box.tolist()
        label = model.config.id2label[label.item()]
        draw.rectangle(box_values, outline="red", width=4)
        draw.text(box_values[0:2], label, font=font, fill="red")
image.show()
