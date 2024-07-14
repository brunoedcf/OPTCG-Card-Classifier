from ultralytics import YOLO
import torch
from PIL import Image
import yaml


def load_class_names(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["names"]


def classify_image(image_path, class_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = YOLO("best.pt").to(device)

    img = Image.open(image_path).convert("RGB")

    results = model.predict(img)

    predictions = results[0].boxes
    for pred in predictions:
        label_idx = int(pred.cls.item())
        label_name = class_names[label_idx]
        confidence = pred.conf.item()
        print(f"Class: {label_name}, Confidence: {confidence:.2f}")


if __name__ == "__main__":
    config_path = "config.yaml"
    class_names = load_class_names(config_path)
    image_path = "test_assets\optcgteste_1.jpeg"
    classify_image(image_path, class_names)
