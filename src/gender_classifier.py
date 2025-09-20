"""
Simplified Gender Classification Model for Speech Recognition
Focuses specifically on gender classification from audio recordings using MFCC features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
from scipy.stats import skew, kurtosis

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Neural Network
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Neural network training will be disabled.")


class GenderClassifier:
    """
    Gender classification model for speech recognition.
    Extracts MFCC features and trains a neural network for gender classification.
    """
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        """
        Initialize the gender classifier.
        
        Args:
            sr: Sample rate for audio loading
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
    def extract_mfcc_features(self, file_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Array of MFCC features
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sr)
            y = librosa.util.normalize(y)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Calculate statistical features for each MFCC coefficient
            features = []
            for i in range(self.n_mfcc):
                features.extend([
                    np.mean(mfcc[i]),      # Mean
                    np.std(mfcc[i]),       # Standard deviation
                    np.min(mfcc[i]),       # Minimum
                    np.max(mfcc[i]),       # Maximum
                    skew(mfcc[i]),         # Skewness
                    kurtosis(mfcc[i])      # Kurtosis
                ])
            
            # Additional audio features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            features.extend([
                np.mean(spectral_centroids),    # Spectral centroid mean
                np.std(spectral_centroids),     # Spectral centroid std
                np.mean(spectral_rolloff),      # Spectral rolloff mean
                np.std(spectral_rolloff),       # Spectral rolloff std
                np.mean(zero_crossing_rate),    # Zero crossing rate mean
                np.std(zero_crossing_rate),     # Zero crossing rate std
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return np.zeros(self.n_mfcc * 6 + 6)  # Return zeros if error
    
    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and organize audio files by gender.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Tuple of (features, labels)
        """
        print("Loading and organizing audio files by gender...")
        
        features_list = []
        labels_list = []
        file_paths = []
        
        # Process all digit directories (d0 to d9)
        for digit in range(10):
            digit_dir = os.path.join(dataset_path, f'd{digit}')
            
            if not os.path.exists(digit_dir):
                continue
            
            # Process male and female directories
            for gender in ['male', 'female']:
                gender_dir = os.path.join(digit_dir, gender)
                
                if not os.path.exists(gender_dir):
                    continue
                
                # Get all .wav files in the directory
                for filename in os.listdir(gender_dir):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(gender_dir, filename)
                        
                        # Extract features
                        features = self.extract_mfcc_features(file_path)
                        features_list.append(features)
                        
                        # Encode gender: male=0, female=1
                        gender_label = 0 if gender == 'male' else 1
                        labels_list.append(gender_label)
                        file_paths.append(file_path)
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        print(f"Loaded {len(features_array)} audio files")
        print(f"Male samples: {np.sum(labels_array == 0)}")
        print(f"Female samples: {np.sum(labels_array == 1)}")
        print(f"Feature vector dimension: {features_array.shape[1]}")
        
        return features_array, labels_array
    
    def prepare_train_test_split(self, features: np.ndarray, labels: np.ndarray, 
                                test_size: float = 0.3, random_state: int = 42) -> Tuple:
        """
        Prepare 70/30 train-test split as specified.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion for test set (0.3 for 70/30 split)
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Preparing {int((1-test_size)*100)}/{int(test_size*100)} train-test split...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        print(f"Training - Male: {np.sum(y_train == 0)}, Female: {np.sum(y_train == 1)}")
        print(f"Test - Male: {np.sum(y_test == 0)}, Female: {np.sum(y_test == 1)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_neural_network(self, input_dim: int) -> Sequential:
        """
        Build neural network for gender classification.
        
        Args:
            input_dim: Input feature dimension
            
        Returns:
            Compiled neural network model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network training")
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history dictionary
        """
        print("Training neural network for gender classification...")
        
        # Build model
        self.model = self.build_neural_network(X_train.shape[1])
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance using accuracy and precision metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba.flatten()
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return metrics
    
    def plot_training_history(self, history: Dict) -> None:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(history['accuracy'], label='Training', alpha=0.8)
        axes[0].plot(history['val_accuracy'], label='Validation', alpha=0.8)
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history['loss'], label='Training', alpha=0.8)
        axes[1].plot(history['val_loss'], label='Validation', alpha=0.8)
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Male', 'Female'],
                   yticklabels=['Male', 'Female'])
        plt.title('Gender Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        if TENSORFLOW_AVAILABLE:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("TensorFlow not available. Cannot load model.")


def run_gender_classification_pipeline(dataset_path: str, output_dir: str = "output") -> GenderClassifier:
    """
    Complete pipeline for gender classification.
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save results
        
    Returns:
        Trained GenderClassifier object
    """
    print("=== GENDER CLASSIFICATION PIPELINE ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize classifier
    classifier = GenderClassifier()
    
    # Load dataset
    features, labels = classifier.load_dataset(dataset_path)
    
    # Prepare 70/30 train-test split
    X_train, X_test, y_train, y_test = classifier.prepare_train_test_split(
        features, labels, test_size=0.3
    )
    
    # Train neural network
    history = classifier.train_model(X_train, y_train, X_test, y_test)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Evaluate model
    metrics = classifier.evaluate_model(X_test, y_test)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(y_test, metrics['predictions'])
    
    # Save model
    classifier.save_model(os.path.join(output_dir, 'gender_classifier_model.h5'))
    
    # Print detailed classification report
    print("\n=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(y_test, metrics['predictions'], 
                              target_names=['Male', 'Female']))
    
    print("\n✅ Gender classification pipeline completed successfully!")
    return classifier


if __name__ == "__main__":
    # Example usage
    dataset_path = "Dataset"
    
    # Run complete pipeline
    classifier = run_gender_classification_pipeline(dataset_path)
    
    print(f"\nFinal Results:")
    print(f"Model trained successfully for gender classification")
    print(f"MFCC features extracted and processed")
    print(f"70/30 train-test split applied")
    print(f"Neural network developed and trained")
    print(f"Performance evaluated using accuracy and precision metrics")
