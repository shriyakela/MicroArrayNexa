from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy import ndimage
import csv
from visualizations import generate_all_visualizations

IMG_SIZE = 256
MODEL_PATH = "unet_trained_model.pkl"


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()
        )

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


def load_model(path):
    # Prefer loading a state-dict if present (safer for CPU-only environments)
    state_candidates = [
        os.path.join(os.path.dirname(path) or '.', 'unet_trained_state.pth'),
        os.path.join('.', 'unet_trained_state.pth')
    ]
    for state_path in state_candidates:
        if os.path.exists(state_path):
            try:
                print(f"Loading state_dict from {state_path}")
                state = torch.load(state_path, map_location=torch.device('cpu'))
                net = UNet()
                if isinstance(state, dict) and ('state_dict' in state or 'model_state_dict' in state):
                    sd = state.get('state_dict', state.get('model_state_dict'))
                else:
                    sd = state
                net.load_state_dict(sd)
                net.eval()
                print('Loaded UNet from state_dict successfully')
                return net
            except Exception as e:
                import traceback
                print('State-dict load failed:', e)
                traceback.print_exc()

    # If no state-dict loaded, attempt to load the full pickled model (legacy)
    if not os.path.exists(path):
        return None

    try:
        import sys
        # Ensure UNet is available under __main__ for unpickling (models pickled in notebooks)
        main_mod = sys.modules.get("__main__")
        injected = False
        if main_mod is not None and not hasattr(main_mod, "UNet"):
            setattr(main_mod, "UNet", UNet)
            injected = True
        model = torch.load(path, map_location=torch.device('cpu'))
        if injected:
            try:
                delattr(main_mod, "UNet")
            except Exception:
                pass
        model.eval()
        return model
    except Exception as e:
        import traceback
        print("Error loading model:", e)
        traceback.print_exc()
        return None


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["RESULT_FOLDER"] = os.path.join("static", "results")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

device = torch.device("cpu")
model = load_model(MODEL_PATH)
last_csv_data = None
last_mean_intensities = None
last_visualizations = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", heading="MicroarrayNexa", result=None)


def preprocess_pil(img: Image.Image):
    img = img.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor, arr


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", heading="MicroarrayNexa", error="Model file not found or failed to load.")

    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    tensor, arr = preprocess_pil(img)

    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()

    binary = (pred > 0.1).astype(np.uint8) * 255

    # compute labels and mean intensities using original resized image
    labels, num_spots = ndimage.label(binary)

    mean_vals = []
    for i in range(1, num_spots + 1):
        region = labels == i
        intensity = (arr * 255)[region]
        if intensity.size == 0:
            continue
        mean_vals.append(float(np.mean(intensity)))

    # Generate CSV data
    csv_data = io.StringIO()
    writer = csv.writer(csv_data)
    writer.writerow(['Spot Number', 'Mean Intensity'])
    for i, val in enumerate(mean_vals, 1):
        writer.writerow([i, val])
    global last_csv_data, last_mean_intensities, last_visualizations
    last_csv_data = csv_data.getvalue()
    last_mean_intensities = mean_vals
    
    # Generate visualizations
    try:
        last_visualizations = generate_all_visualizations(
            mean_vals, 
            app.config["RESULT_FOLDER"]
        )
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        last_visualizations = None

    # save original and mask images
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.png")
    result_mask_path = os.path.join(app.config["RESULT_FOLDER"], "mask.png")
    Image.fromarray((arr * 255).astype(np.uint8)).save(upload_path)
    Image.fromarray(binary).save(result_mask_path)

    stats = {
        "num_spots": int(num_spots),
        "mean_intensities": mean_vals[:50]
    }

    return render_template(
        "index.html",
        heading="MicroarrayNexa",
        result={
            "upload": upload_path.replace("\\", "/"),
            "mask": result_mask_path.replace("\\", "/"),
            "stats": stats,
            "visualizations": last_visualizations
        }
    )


@app.route("/download_csv")
def download_csv():
    if last_csv_data is None:
        return "No data available", 404
    output = io.BytesIO()
    output.write(last_csv_data.encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='spots.csv')


@app.route("/visualizations")
def visualizations():
    if last_visualizations is None:
        return "No visualizations available. Please analyze an image first.", 404
    
    return render_template(
        "visualizations.html",
        heading="Analysis Visualizations",
        visualizations=last_visualizations
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
