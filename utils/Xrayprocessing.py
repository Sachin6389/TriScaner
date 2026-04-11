import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(image_path, img_size):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(img_size)

        img = np.array(img, dtype=np.float32)

        # EfficientNet preprocessing
        img = preprocess_input(img)

        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        print("Preprocess Error:", e)
        return None