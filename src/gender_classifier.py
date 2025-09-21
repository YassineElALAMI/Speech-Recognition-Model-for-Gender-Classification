import tensorflow as tf
import numpy as np
from extract_features import extract_mfcc

DIGITS = [str(i) for i in range(10)]
GENDERS = ["male", "female"]

def predict(file_path, model_path="models/gender_digit_classifier.h5"):
    model = tf.keras.models.load_model(model_path)
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return None, None

    # reshape to (1, n_mfcc, max_len, 1)
    X = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1).astype(np.float32)

    preds = model.predict(X)
    # model.predict returns a list for multi-output models
    if isinstance(preds, list) and len(preds) == 2:
        digit_pred, gender_pred = preds
    else:
        # unexpected shape
        return None, None

    digit = DIGITS[int(np.argmax(digit_pred, axis=1)[0])]
    gender = GENDERS[int(np.argmax(gender_pred, axis=1)[0])]

    return gender, digit

if __name__ == "__main__":
    file_path = "test/male_5.wav"
    result = predict(file_path)
    print(f"Prediction → Gender: {result[0]}, Digit: {result[1]}")
