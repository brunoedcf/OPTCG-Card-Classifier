from ultralytics import YOLO
import torch

EPOCHS=50

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = YOLO("assets\models\OPTCG_prototype_V1.pt").to(device)
    results = model.train(data="config.yaml", epochs=EPOCHS)


if __name__ == "__main__":
    train_model()
