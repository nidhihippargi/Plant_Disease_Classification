import tensorflow as tf
from pathlib import Path

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

BASE_DIR = Path(
    "dataset/New Plant Diseases Dataset(Augmented)"
    "/New Plant Diseases Dataset(Augmented)"
)

TRAIN_DIR = BASE_DIR / "train"
VALID_DIR = BASE_DIR / "valid"
TEST_DIR = BASE_DIR / "test"


def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    valid_ds = valid_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, valid_ds, test_ds, class_names