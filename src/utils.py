import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12,4))

    # Plot digit accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['digit_output_accuracy'], label="train_digit")
    plt.plot(history.history['val_digit_output_accuracy'], label="val_digit")
    plt.legend()
    plt.title("Digit Accuracy")

    # Plot gender accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['gender_output_accuracy'], label="train_gender")
    plt.plot(history.history['val_gender_output_accuracy'], label="val_gender")
    plt.legend()
    plt.title("Gender Accuracy")

    plt.show()
