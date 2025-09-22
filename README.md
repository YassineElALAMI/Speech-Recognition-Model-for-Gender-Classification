# Speech Gender & Digit Classifier

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A complete ML pipeline to classify spoken digits (0–9) and speaker gender (male/female) from audio. The pipeline includes MFCC feature extraction, dataset preparation, model training, evaluation, plotting, and a local GUI for inference.

## 🚀 Features

- MFCC-based feature extraction with normalization and fixed-length frames
- Multi-output CNN to predict digit and gender simultaneously
- Training plots (accuracy, loss) saved to `plots/`
- Evaluation reports and confusion matrices saved to `plots/`
- Simple GUI to record/select audio and run inference
- Reproducible dataset splitting with stratification

## 📂 Project Structure

```
SpeechGenderDigit/
├── dataset/             # Your audio dataset (d0..d9 / male & female)
├── models/              # Saved model files (generated after training)
├── plots/               # Training/evaluation plots and reports
├── recordings/          # Temporary recordings used by GUI
├── src/
│   ├── extract_features.py   # MFCC extraction (normalization, padding)
│   ├── prepare_dataset.py    # Load audio, prepare X and labels, train/test split
│   ├── train_model.py        # Build and train model; save plots & model
│   ├── evaluate_model.py     # Evaluate model; save reports & confusion matrices
│   ├── gender_classifier.py  # Single-file predictor (returns gender, digit)
│   ├── interface.py          # Tkinter GUI for recording/selecting files & plotting
│   └── utils.py              # Optional plotting helper functions
├── README.md
└── requirements.txt
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone repository and change directory:
    ```bash
    git clone https://github.com/YassineElALAMI/Speech-Recognition-Model-for-Gender-Classification.git
    cd "C:\Users\hp\Desktop\for master\S1\RAP\Speech-Recognition-Model-for-Gender-Classification"

    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    # If you don't have requirements.txt:
    pip install numpy librosa matplotlib scikit-learn tensorflow sounddevice scipy
    ```

## 📁 Dataset layout

Place audio files under `dataset/` with this exact structure:

```
dataset/
  d0/
    male/
      file1.wav
      ...
    female/
      ...
  d1/
  ...
  d9/
```

- Folder names must match `d0`..`d9` (prepare_dataset.py expects these).
- Each gender folder contains WAV files for that digit/gender.
- Files will be resampled to 16 kHz and converted to mono by extract_features.py.

## 🚦 Usage

1. Prepare dataset in the structure above.

2. Train model:
    ```bash
    python src/train_model.py
    ```
    - Trained model saved to `models/gender_digit_classifier.h5`.
    - Training plots saved to `plots/training_accuracy.png` and `plots/training_loss.png`.

3. Evaluate model:
    ```bash
    python src/evaluate_model.py
    ```
    - Generates `plots/classification_reports.txt`, `plots/confusion_digits.png`, `plots/confusion_gender.png`.

4. Run GUI for inference:
    ```bash
    python src/interface.py
    ```
    - Record or load audio and run prediction (shows waveform and MFCC plots).

5. Programmatic prediction:
    ```python
    from src.gender_classifier import predict
    gender, digit = predict("path/to/file.wav")
    ```
    - Returns `(None, None)` on failure.

## 📊 Model & Plots

- Training plots and evaluation outputs are saved to `plots/`.
- Model is saved in HDF5 format (`.h5`); consider saving with `.keras` if you prefer native Keras format.

## ✅ What to check after training

- Training vs validation curves for overfitting/underfitting.
- Confusion matrices to find commonly confused digits.
- Class balance in dataset (imbalanced data can inflate accuracy).

## 🔧 Tips & Improvements

- Add ModelCheckpoint and EarlyStopping to save the best model.
- Apply data augmentation: noise injection, time stretch, pitch shift.
- Use batch normalization, dropout, or reduce model size if overfitting.
- Convert model.save(...) to `.keras` to avoid HDF5 legacy warning.
- Run stratified k-fold cross-validation for robust estimates.

## 🧰 Requirements

Suggested packages (put in requirements.txt):
- numpy
- librosa
- matplotlib
- scikit-learn
- tensorflow
- sounddevice
- scipy

## 📝 License

This project is licensed under the MIT License — see the LICENSE file.

## 🤝 Contributing

Contributions welcome. Suggested flow:
1. Fork repository
2. Create feature branch
3. Add tests and update README
4. Open a Pull Request

## Contact

For any questions or suggestions, please contact:

- Project Maintainer: [Yassine EL ALAMI]
- Email: [yassine.elalami5@usmba.ac.ma]
- GitHub: [https://github.com/YassineElALAMI]
