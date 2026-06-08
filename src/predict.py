import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


CLASS_MAPPING_FALLBACK = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def preprocess_single_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)

    img_array = np.array(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_single_inference(image_path, model_path):

    print(f"\nLoading model: {model_path}")

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
    except Exception as e:
        print(f"\nError loading model:\n{e}")
        sys.exit(1)

    processed_tensor = preprocess_single_image(image_path)

    predictions = model.predict(
        processed_tensor,
        verbose=0
    )

    predicted_idx = np.argmax(predictions[0])

    confidence = float(
        predictions[0][predicted_idx] * 100
    )

    predicted_class = CLASS_MAPPING_FALLBACK[predicted_idx]

    print("\n" + "=" * 50)
    print("PLANT DISEASE PREDICTION")
    print("=" * 50)
    print(f"Predicted Disease : {predicted_class}")
    print(f"Confidence        : {confidence:.2f}%")
    print("=" * 50)

    display_img = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(8, 6))
    plt.imshow(display_img)
    plt.axis("off")

    plt.title(
        f"{predicted_class}\nConfidence: {confidence:.2f}%",
        fontsize=12
    )

    os.makedirs("results", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = f"results/{base_name}_prediction.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nSaved image to: {output_file}")

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plant Disease Prediction"
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to leaf image"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="plant_disease_model.keras",
        help="Path to trained model"
    )

    args = parser.parse_args()

    run_single_inference(
        args.image_path,
        args.model
    )