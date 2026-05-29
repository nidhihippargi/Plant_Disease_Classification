import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PATHS — dataset ka exact structure
# ─────────────────────────────────────────────
BASE_DIR  = r"C:\Users\tanis\OneDrive\Apps\DL project\data\raw\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "valid")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

# ─────────────────────────────────────────────
# AUGMENTATION — Train generator
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# Validation — only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# ─────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
# Test generator
TEST_DIR = os.path.join(BASE_DIR, "test")

test_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ─────────────────────────────────────────────
# CLASS INFO
# ─────────────────────────────────────────────
class_indices = train_generator.class_indices
class_names   = list(class_indices.keys())
num_classes   = len(class_names)

print(f"\n✅ Number of classes : {num_classes}")
print(f"✅ Train samples     : {train_generator.samples}")
print(f"✅ Val samples       : {val_generator.samples}")

# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights_array))
print(f"✅ Class weights computed for {len(class_weights)} classes")

# ─────────────────────────────────────────────
# VERIFY BATCH SHAPES
# ─────────────────────────────────────────────
def verify_batch(generator, name="Generator"):
    batch_images, batch_labels = next(generator)
    print(f"\n[{name}] Image shape  : {batch_images.shape}")
    print(f"[{name}] Label shape  : {batch_labels.shape}")
    print(f"[{name}] Pixel range  : {batch_images.min():.2f} - {batch_images.max():.2f}")
    assert batch_images.shape[1:] == (224, 224, 3), "❌ Image size wrong!"
    assert batch_labels.shape[1]  == num_classes,   "❌ Class count wrong!"
    print(f"✅ {name} batch verified!")

verify_batch(train_generator, "Train")
verify_batch(val_generator,   "Val")
verify_batch(test_generator,  "Test")

print("\n✅ Data pipeline ready! Generators can be imported for MobileNetV2 training.")
