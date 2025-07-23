from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0  # atau MobileNetV2 jika Anda pakai itu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ubah ke domain frontend jika sudah produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load model arsitektur dan bobot pretrained dari ImageNet ===
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# === Load weights hasil training ===
model.load_weights("model/best_model_weights_2.h5")


# === Label Kelas ===
LABELS = ['Basal Cell Carcinoma', 'Nevus', 'Melanoma', 'Benign Keratosis Lesion', 'Actinic Keratosis', 'Normal']

# === Pydantic model ===
class ImagePayload(BaseModel):
    image_base64: str

# === Fungsi pra-pemrosesan gambar ===
def preprocess_image(base64_str: str):
    try:
        image_data = base64.b64decode(base64_str.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Gagal memproses gambar.")

@app.post("/predict")
def predict(payload: ImagePayload):
    try:
        image_array = preprocess_image(payload.image_base64)
        prediction = model.predict(image_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class]) * 100

        diagnosis = LABELS[predicted_class]
        recommendation = (
            "Segera periksa ke dokter kulit untuk evaluasi lebih lanjut."
            if diagnosis in ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratosis"]
            else "Pantau kondisi kulit secara berkala."
        )

        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
    
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5024)