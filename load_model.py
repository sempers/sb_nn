import os
import gdown

MODEL_URL = "https://drive.google.com/uc?export=download&id=1UzuYps6TJ_i_lEngDv2UNo4xCPcmxRBR"

if not os.path.exists("model.h5"):
    gdown.download(MODEL_URL, "model.h5")
    print("Model loaded")
else:
    print("Model is already there")