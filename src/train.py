from src.data_loader import load_datasets
from src.model import build_model
from src.losses import weighted_focal_loss

import tensorflow as tf


def main():

    train_ds, valid_ds, test_ds, class_names = load_datasets()

    model = build_model(
        num_classes=len(class_names)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=weighted_focal_loss(
            alpha=0.25,
            gamma=2.0
        ),
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=5
    )

    model.save("plant_disease_model.keras")

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()