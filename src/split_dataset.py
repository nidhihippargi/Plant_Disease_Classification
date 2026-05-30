
import os
import shutil
import random
from pathlib import Path

# PATHS
BASE_DIR = "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TEST_DIR  = os.path.join(BASE_DIR, "test")

TEST_SPLIT = 0.50  # valid ka 50% test mein jayega
SEED = 42
random.seed(SEED)

print("Creating test set from valid folder...\n")

for class_name in os.listdir(VALID_DIR):
    class_path = Path(VALID_DIR) / class_name
    if not class_path.is_dir():
        continue

    images = list(class_path.glob("*.jpg")) + \
             list(class_path.glob("*.JPG")) + \
             list(class_path.glob("*.png"))

    random.shuffle(images)
    n_test = int(len(images) * TEST_SPLIT)
    test_images = images[:n_test]

    # Test folder banao
    test_class_dir = Path(TEST_DIR) / class_name
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # Files copy karo
    for img in test_images:
        shutil.copy(img, test_class_dir / img.name)

    print(f"{class_name}: {len(test_images)} test images")

print(f"\n✅ Test set created at: {TEST_DIR}")
print("✅ split_dataset.py done!")