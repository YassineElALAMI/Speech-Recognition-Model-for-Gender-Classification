import tensorflow as tf
from tensorflow.keras import layers, models
from prepare_dataset import load_dataset
import os
import matplotlib.pyplot as plt

def build_model(input_shape, num_digits=10, num_genders=2):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    digit_output = layers.Dense(num_digits, activation='softmax', name="digit_output")(x)
    gender_output = layers.Dense(num_genders, activation='softmax', name="gender_output")(x)

    model = models.Model(inputs=inputs, outputs=[digit_output, gender_output])

    model.compile(optimizer='adam',
                  loss={'digit_output': 'categorical_crossentropy',
                        'gender_output': 'categorical_crossentropy'},
                  metrics={'digit_output': 'accuracy', 'gender_output': 'accuracy'})
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset("dataset/")
    model = build_model(X_train.shape[1:])
    model.summary()

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=20, batch_size=32)

    # ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Plot accuracies (if keys exist)
    h = history.history
    plt.figure(figsize=(8,5))
    if 'digit_output_accuracy' in h:
        plt.plot(h['digit_output_accuracy'], label='train_digit')
    if 'val_digit_output_accuracy' in h:
        plt.plot(h['val_digit_output_accuracy'], label='val_digit')
    if 'gender_output_accuracy' in h:
        plt.plot(h['gender_output_accuracy'], label='train_gender', linestyle='--')
    if 'val_gender_output_accuracy' in h:
        plt.plot(h['val_gender_output_accuracy'], label='val_gender', linestyle='--')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "training_accuracy.png"))
    plt.close()

    # Plot loss
    plt.figure(figsize=(8,5))
    if 'loss' in h:
        plt.plot(h['loss'], label='train_loss')
    if 'val_loss' in h:
        plt.plot(h['val_loss'], label='val_loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "training_loss.png"))
    plt.close()

    model.save("models/gender_digit_classifier.h5")
