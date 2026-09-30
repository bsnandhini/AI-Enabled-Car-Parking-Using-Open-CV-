import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model
from keras.utils import to_categorical

from data_loader import X, Z


# Create output folder
os.makedirs("output_images", exist_ok=True)


# Load the trained model
model = load_model("models/first_cnn.h5")


# Label encoding
le = LabelEncoder()
Y = le.fit_transform(Z)
Y = to_categorical(Y, 2)

X = np.array(X)
X = X / 255.0


# Separate data
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42
)


# --------------------------------------------------
# Model Evaluation - Training Data
# --------------------------------------------------

train_loss, train_accuracy = model.evaluate(
    x_train,
    y_train,
    verbose=1
)

print("\nTraining Accuracy:", train_accuracy)
print("Training Accuracy (%):", train_accuracy * 100)
print("Training Loss:", train_loss)


# --------------------------------------------------
# Model Evaluation - Test Data
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nTest Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)
print("Test Loss:", test_loss)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

pred = model.predict(x_test)

pred_digits = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)


# --------------------------------------------------
# Properly Classified Images
# --------------------------------------------------

prop_class = []

for i in range(len(y_test)):

    if y_true[i] == pred_digits[i]:

        prop_class.append(i)

        if len(prop_class) == 8:
            break


# --------------------------------------------------
# Misclassified Images
# --------------------------------------------------

mis_class = []

for i in range(len(y_test)):

    if y_true[i] != pred_digits[i]:

        mis_class.append(i)

        if len(mis_class) == 8:
            break


# --------------------------------------------------
# Display & Save Properly Classified Images
# --------------------------------------------------

if len(prop_class) > 0:

    fig, ax = plt.subplots(4, 2)

    fig.set_size_inches(15, 15)

    count = 0

    for i in range(4):

        for j in range(2):

            if count < len(prop_class):

                index = prop_class[count]

                image = np.clip(x_test[index], 0, 1)

                ax[i, j].imshow(x_test[index] * 255)

                actual_label = le.inverse_transform(
                    [y_true[index]]
                )[0]

                predicted_label = le.inverse_transform(
                    [pred_digits[index]]
                )[0]

                ax[i, j].set_title(
                    "Actual: " + actual_label +
                    " | Predicted: " + predicted_label
                )

                ax[i, j].axis("off")

                count += 1

            else:

                ax[i, j].axis("off")

    plt.tight_layout()

    plt.savefig(
        "output_images/properly_classified_images.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)


# --------------------------------------------------
# Display & Save Misclassified Images
# --------------------------------------------------

if len(mis_class) > 0:

    fig, ax = plt.subplots(4, 2)

    fig.set_size_inches(15, 15)

    count = 0

    for i in range(4):

        for j in range(2):

            if count < len(mis_class):

                index = mis_class[count]

                image = np.clip(x_test[index], 0, 1)

                ax[i, j].imshow(x_test[index] * 255)

                actual_label = le.inverse_transform(
                    [y_true[index]]
                )[0]

                predicted_label = le.inverse_transform(
                    [pred_digits[index]]
                )[0]

                ax[i, j].set_title(
                    "Actual: " + actual_label +
                    " | Predicted: " + predicted_label
                )

                ax[i, j].axis("off")

                count += 1

            else:

                ax[i, j].axis("off")

    plt.tight_layout()

    plt.savefig(
        "output_images/misclassified_images.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

else:

    print("\nNo misclassified images found.")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

label = ["Free", "Full"]

cm = confusion_matrix(
    y_true,
    pred_digits
)

plt.figure(figsize=(9, 9))

ax = sns.heatmap(
    cm,
    cmap="rocket_r",
    fmt="d",
    annot=True,
    annot_kws={'size': 16},
    xticklabels=label,
    yticklabels=label
)

ax.set_ylabel(
    "Actual",
    fontsize=20
)

ax.set_xlabel(
    "Predicted",
    fontsize=20
)

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "output_images/confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

plt.close()


# --------------------------------------------------
# Accuracy and Loss Graph
# --------------------------------------------------
# These values are from the current evaluation.

metrics = ["Training", "Test"]

accuracy_values = [
    train_accuracy,
    test_accuracy
]

loss_values = [
    train_loss,
    test_loss
]


# Accuracy Graph
plt.figure(figsize=(8, 6))

plt.bar(
    metrics,
    accuracy_values
)

plt.ylim(0, 1)

plt.ylabel("Accuracy")

plt.title("Model Accuracy")

plt.tight_layout()

plt.savefig(
    "output_images/model_accuracy.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

plt.close()


# Loss Graph
plt.figure(figsize=(8, 6))

plt.bar(
    metrics,
    loss_values
)

plt.ylabel("Loss")

plt.title("Model Loss")

plt.tight_layout()

plt.savefig(
    "output_images/model_loss.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

plt.close()


# --------------------------------------------------
# Final Output
# --------------------------------------------------

print("\nEvaluation completed successfully.")

print("Test Accuracy:", test_accuracy * 100, "%")

print("\nOutput images saved in:")

print("output_images/")