import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# Setup deterministic fallback dataset maps inside dictionary variable to bypass external module dependency structural requirements
CLASS_MAPPING_FALLBACK = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def preprocess_single_image(image_path, target_size=(224, 224)):
    """Loads, resizes and scales image pixels uniformly inside functional range."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Requested leaf target asset file cannot be located: {image_path}")
        
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    
    # Preprocessing identical to MobileNetV2 standards (Rescaling inputs between [-1, 1])
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension wrapper
    return img_array

def run_single_inference(image_path, model_path):
    """Executes prediction cycle for single leaf image target."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model binary: {e}")
        sys.exit(1)
        
    # Standard input mapping initialization pipeline checks
    processed_tensor = preprocess_single_image(image_path)
    
    predictions = model.predict(processed_tensor, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class_idx] * 100
    
    predicted_class_name = CLASS_MAPPING_FALLBACK[predicted_class_idx]
    
    print("\n" + "="*40)
    print("        INFERENCE PIPELINE RESULTS      ")
    print("="*40)
    print(f"Predicted Disease:\n{predicted_class_name}")
    print(f"\nConfidence:\n{confidence_score:.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plant Disease Single Target Inference Trigger Module Pipeline")
    parser.add_argument('image_path', type=str, help='Absolute or relative file address pointing towards leaf image source file target asset')
    parser.add_argument('--model', type=str, default='models/plant_disease_classifier.keras', help='Path to compiled model binary')
    
    args = parser.parse_args()
    run_single_inference(args.image_path, args.model)