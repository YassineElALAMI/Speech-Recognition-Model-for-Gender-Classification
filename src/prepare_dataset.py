import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from extract_features import extract_mfcc

DIGITS = [f"d{i}" for i in range(10)]
GENDERS = ["male", "female"]

def load_dataset(dataset_path="dataset/"):
    X, y_digit, y_gender = [], [], []

    for digit_idx, digit in enumerate(DIGITS):
        for gender_idx, gender in enumerate(GENDERS):
            folder = os.path.join(dataset_path, digit, gender)
            if not os.path.isdir(folder):
                continue

            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder, file)
                    mfcc = extract_mfcc(file_path)
                    if mfcc is not None:
                        X.append(mfcc)
                        y_digit.append(digit_idx)
                        y_gender.append(gender_idx)

    if len(X) == 0:
        raise ValueError(f"No audio files found under {dataset_path}. Check folders and names (expected {DIGITS} subfolders with {GENDERS}).")

    X = np.array(X)[..., np.newaxis].astype(np.float32)  # shape (samples, n_mfcc, max_len, 1)

    # Keep integer labels for stratify, then make one-hot copies for training
    labels_digit = np.array(y_digit, dtype=np.int32)
    labels_gender = np.array(y_gender, dtype=np.int32)

    y_digit_cat = to_categorical(labels_digit, num_classes=len(DIGITS))
    y_gender_cat = to_categorical(labels_gender, num_classes=len(GENDERS))

    # Split explicitly using integer labels for stratification
    X_train, X_test, y_digit_train, y_digit_test, y_gender_train, y_gender_test = train_test_split(
        X, y_digit_cat, y_gender_cat, test_size=0.3, random_state=42, stratify=labels_digit
    )

    y_train = [y_digit_train, y_gender_train]
    y_test = [y_digit_test, y_gender_test]

    return X_train, X_test, y_train, y_test
