import tensorflow as tf
from prepare_dataset import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset("dataset/")
    model = tf.keras.models.load_model("models/gender_digit_classifier.h5")

    preds = model.predict(X_test)
    if isinstance(preds, list) and len(preds) == 2:
        y_pred_digit, y_pred_gender = preds
    else:
        # unexpected: can't evaluate
        raise RuntimeError("Model did not return two outputs on predict()")

    y_true_digit, y_true_gender = y_test

    # Convert one-hot to labels
    y_pred_digit = np.argmax(y_pred_digit, axis=1)
    y_true_digit = np.argmax(y_true_digit, axis=1)

    y_pred_gender = np.argmax(y_pred_gender, axis=1)
    y_true_gender = np.argmax(y_true_gender, axis=1)

    digit_report = classification_report(y_true_digit, y_pred_digit)
    gender_report = classification_report(y_true_gender, y_pred_gender)

    print("Digit Classification Report:")
    print(digit_report)

    print("Gender Classification Report:")
    print(gender_report)

    # ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Save text reports
    with open(os.path.join("plots", "classification_reports.txt"), "w") as f:
        f.write("Digit Classification Report:\n")
        f.write(digit_report + "\n\n")
        f.write("Gender Classification Report:\n")
        f.write(gender_report + "\n")

    # Confusion matrix for digits
    cm_digits = confusion_matrix(y_true_digit, y_pred_digit)
    plt.figure(figsize=(8,6))
    plt.imshow(cm_digits, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Digits")
    plt.colorbar()
    ticks = np.arange(cm_digits.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # annotate
    thresh = cm_digits.max() / 2.
    for i in range(cm_digits.shape[0]):
        for j in range(cm_digits.shape[1]):
            plt.text(j, i, format(cm_digits[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm_digits[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "confusion_digits.png"))
    plt.close()

    # Confusion matrix for gender
    cm_gender = confusion_matrix(y_true_gender, y_pred_gender)
    plt.figure(figsize=(4,4))
    plt.imshow(cm_gender, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Gender")
    plt.colorbar()
    plt.xticks([0,1], [0,1])
    plt.yticks([0,1], [0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = cm_gender.max() / 2.
    for i in range(cm_gender.shape[0]):
        for j in range(cm_gender.shape[1]):
            plt.text(j, i, format(cm_gender[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm_gender[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "confusion_gender.png"))
    plt.close()
