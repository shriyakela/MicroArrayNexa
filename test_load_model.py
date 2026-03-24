import pickle
import torch
print("torch:", torch.__version__)
try:
    with open("unet_trained_model.pkl","rb") as f:
        m = pickle.load(f)
    print("Loaded model type:", type(m))
except Exception as e:
    print("Failed to load model:", e)
