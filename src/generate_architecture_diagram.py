import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 10))

ax.axis("off")

layers = [
    "Input Leaf Image\n224×224×3",
    "Rescaling\n[-1,1]",
    "MobileNetV2\n(ImageNet Weights)\nFrozen Backbone",
    "GlobalAveragePooling2D",
    "Dropout\n0.3",
    "Dense Layer\n38 Classes",
    "Softmax",
    "Disease Prediction"
]

y = 0.9

for i, layer in enumerate(layers):

    ax.text(
        0.5,
        y,
        layer,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5")
    )

    if i < len(layers)-1:
        ax.arrow(
            0.5,
            y-0.04,
            0,
            -0.08,
            head_width=0.02,
            head_length=0.02,
            length_includes_head=True
        )

    y -= 0.12

plt.title(
    "Plant Disease Classification Architecture",
    fontsize=14
)

plt.savefig(
    "results/architecture_diagram.png",
    bbox_inches="tight",
    dpi=300
)

print("Saved: results/architecture_diagram.png")