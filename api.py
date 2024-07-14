from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
from PIL import Image
import yaml
import io
from dotenv import dotenv_values

config = dotenv_values(".env")

app = FastAPI()


def load_class_names(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["names"]


def classify_image(image_data, class_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = YOLO("best.pt").to(device)

    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    results = model.predict(img)

    predictions = results[0].boxes
    if len(predictions) > 0:
        pred = predictions[0]
        label_idx = int(pred.cls.item())  # Convert tensor to int
        label_name = class_names[label_idx]
        confidence = pred.conf.item()  # Convert tensor to float
        return {"class": label_name, "confidence": confidence}
    else:
        return {"class": None, "confidence": None}


config_path = "config.yaml"
class_names = load_class_names(config_path)


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_data = await file.read()
    result = classify_image(image_data, class_names)
    print(result)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config["HOST"], port=config["PORT"])
