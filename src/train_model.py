"""
Multi-Output CNN training module for speech recognition and gender classification.
Implements a CNN with shared convolutional layers and two output branches for digit and gender prediction.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Deep Learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Deep learning models will be disabled.")

from utils import save_model, load_model, plot_training_history, calculate_multi_output_metrics


class MultiOutputCNNTrainer:
    """
    Multi-Output CNN trainer for speech recognition and gender classification.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the trainer.
        
        Args:
            output_dir: Directory to save models and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.cnn_model = None
        self.training_history = {}
        self.best_accuracy = 0
        
    def build_multi_output_cnn(self, input_shape: Tuple[int, int, int]) -> Model:
        """
        Build Multi-Output CNN with shared convolutional layers and two output branches.
        
        Args:
            input_shape: Input shape (n_mfcc, max_frames, 1)
            
        Returns:
            Compiled Multi-Output CNN model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN training")
        
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, 
            Flatten, Dense, GlobalAveragePooling2D
        )
        from tensorflow.keras.optimizers import Adam
        
        # Input layer
        inputs = Input(shape=input_shape, name='mfcc_input')
        
        # Shared Convolutional Layers
        # First Conv Block
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Second Conv Block
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Third Conv Block
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Global Average Pooling instead of Flatten to reduce parameters
        x = GlobalAveragePooling2D()(x)
        
        # Shared Dense Layers
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Two Output Branches
        
        # Gender classifier branch (softmax over 2 classes)
        gender_branch = Dense(128, activation='relu')(x)
        gender_branch = Dropout(0.2)(gender_branch)
        gender_output = Dense(2, activation='softmax', name='gender_output')(gender_branch)
        
        # Digit classifier branch (softmax over 10 classes)
        digit_branch = Dense(128, activation='relu')(x)
        digit_branch = Dropout(0.2)(digit_branch)
        digit_output = Dense(10, activation='softmax', name='digit_output')(digit_branch)
        
        # Create model
        model = Model(inputs=inputs, outputs=[gender_output, digit_output], name='MultiOutputCNN')
        
        # Compile model with categorical crossentropy for both tasks
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'gender_output': 'categorical_crossentropy',
                'digit_output': 'categorical_crossentropy'
            },
            loss_weights={'gender_output': 1.0, 'digit_output': 1.0},
            metrics=['accuracy']
        )
        
        return model
    
    def train_multi_output_cnn(self, X_train: np.ndarray, y_gender_train: np.ndarray, y_digit_train: np.ndarray,
                              X_test: np.ndarray, y_gender_test: np.ndarray, y_digit_test: np.ndarray,
                              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train Multi-Output CNN for speech recognition and gender classification.
        
        Args:
            X_train: Training MFCC features (n_samples, n_mfcc, max_frames, 1)
            y_gender_train: Training gender labels (n_samples, 2) - one-hot encoded
            y_digit_train: Training digit labels (n_samples, 10) - one-hot encoded
            X_test: Test MFCC features
            y_gender_test: Test gender labels - one-hot encoded
            y_digit_test: Test digit labels - one-hot encoded
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing trained model and results
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping CNN training.")
            return {}
        
        print("Training Multi-Output CNN...")
        
        # Build CNN model
        input_shape = X_train.shape[1:]  # (n_mfcc, max_frames, 1)
        model = self.build_multi_output_cnn(input_shape)
        
        # Print model summary
        model.summary()
        
        # Callbacks
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_cnn_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train,
            [y_gender_train, y_digit_train],
            validation_data=(X_test, [y_gender_test, y_digit_test]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        gender_pred_proba, digit_pred_proba = model.predict(X_test)
        gender_pred = np.argmax(gender_pred_proba, axis=1)
        digit_pred = np.argmax(digit_pred_proba, axis=1)
        
        # Convert one-hot back to class indices for accuracy calculation
        y_gender_test_indices = np.argmax(y_gender_test, axis=1)
        y_digit_test_indices = np.argmax(y_digit_test, axis=1)
        
        # Calculate accuracies
        gender_accuracy = accuracy_score(y_gender_test_indices, gender_pred)
        digit_accuracy = accuracy_score(y_digit_test_indices, digit_pred)
        both_correct = (y_gender_test_indices == gender_pred) & (y_digit_test_indices == digit_pred)
        overall_accuracy = np.mean(both_correct)
        
        # Store results
        results = {
            'multi_output_cnn': {
                'model': model,
                'gender_accuracy': gender_accuracy,
                'digit_accuracy': digit_accuracy,
                'overall_accuracy': overall_accuracy,
                'gender_predictions': gender_pred,
                'digit_predictions': digit_pred,
                'history': history.history
            }
        }
        
        # Save model
        model.save(os.path.join(self.output_dir, 'multi_output_cnn_model.h5'))
        self.cnn_model = model
        
        # Store training history
        self.training_history['multi_output_cnn'] = history.history
        
        print(f"\n=== Multi-Output CNN Training Results ===")
        print(f"Gender accuracy: {gender_accuracy:.4f}")
        print(f"Digit accuracy: {digit_accuracy:.4f}")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        
        # Update best accuracy
        if overall_accuracy > self.best_accuracy:
            self.best_accuracy = overall_accuracy
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history (accuracy/loss curves) for Multi-Output CNN.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.training_history or 'multi_output_cnn' not in self.training_history:
            print("No training history available to plot.")
            return
        
        history = self.training_history['multi_output_cnn']
        
        import matplotlib.pyplot as plt
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gender Accuracy
        axes[0, 1].plot(history['gender_output_accuracy'], label='Training Gender Acc', color='blue')
        axes[0, 1].plot(history['val_gender_output_accuracy'], label='Validation Gender Acc', color='red')
        axes[0, 1].set_title('Gender Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Digit Accuracy
        axes[1, 0].plot(history['digit_output_accuracy'], label='Training Digit Acc', color='blue')
        axes[1, 0].plot(history['val_digit_output_accuracy'], label='Validation Digit Acc', color='red')
        axes[1, 0].set_title('Digit Classification Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gender Loss
        axes[1, 1].plot(history['gender_output_loss'], label='Training Gender Loss', color='blue')
        axes[1, 1].plot(history['val_gender_output_loss'], label='Validation Gender Loss', color='red')
        axes[1, 1].set_title('Gender Classification Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Output CNN Training History', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray],
                            X_test: np.ndarray, y_test: Dict[str, np.ndarray],
                            model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name of model to tune
            
        Returns:
            Dictionary containing best parameters and results
        """
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return {}
        
        # Base models
        base_models = {
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        # Tune digit classifier
        print("Tuning digit classifier...")
        digit_grid = GridSearchCV(
            base_models[model_name],
            param_grids[model_name],
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        digit_grid.fit(X_train, y_train['digit'])
        
        # Tune gender classifier
        print("Tuning gender classifier...")
        gender_grid = GridSearchCV(
            base_models[model_name],
            param_grids[model_name],
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        gender_grid.fit(X_train, y_train['gender'])
        
        # Evaluate best models
        digit_pred = digit_grid.predict(X_test)
        gender_pred = gender_grid.predict(X_test)
        
        digit_accuracy = accuracy_score(y_test['digit'], digit_pred)
        gender_accuracy = accuracy_score(y_test['gender'], gender_pred)
        both_correct = (y_test['digit'] == digit_pred) & (y_test['gender'] == gender_pred)
        overall_accuracy = np.mean(both_correct)
        
        results = {
            'best_digit_params': digit_grid.best_params_,
            'best_gender_params': gender_grid.best_params_,
            'digit_accuracy': digit_accuracy,
            'gender_accuracy': gender_accuracy,
            'overall_accuracy': overall_accuracy,
            'digit_predictions': digit_pred,
            'gender_predictions': gender_pred,
            'digit_model': digit_grid.best_estimator_,
            'gender_model': gender_grid.best_estimator_
        }
        
        print(f"Best digit parameters: {digit_grid.best_params_}")
        print(f"Best gender parameters: {gender_grid.best_params_}")
        print(f"Digit accuracy: {digit_accuracy:.4f}")
        print(f"Gender accuracy: {gender_accuracy:.4f}")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        
        return results
    
    def cross_validate_models(self, X: np.ndarray, y: Dict[str, np.ndarray],
                            cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Perform cross-validation for all models.
        
        Args:
            X: Features
            y: Labels (dict with 'digit' and 'gender' keys)
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary containing CV results
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        cv_results = {}
        
        for model_name, model in models.items():
            print(f"\nCross-validating {model_name}...")
            
            # CV for digit classification
            digit_scores = cross_val_score(
                model, X, y['digit'], 
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # CV for gender classification
            gender_scores = cross_val_score(
                model, X, y['gender'],
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            cv_results[model_name] = {
                'digit_scores': digit_scores,
                'gender_scores': gender_scores,
                'digit_mean': digit_scores.mean(),
                'digit_std': digit_scores.std(),
                'gender_mean': gender_scores.mean(),
                'gender_std': gender_scores.std()
            }
            
            print(f"Digit CV: {digit_scores.mean():.4f} (+/- {digit_scores.std() * 2:.4f})")
            print(f"Gender CV: {gender_scores.mean():.4f} (+/- {gender_scores.std() * 2:.4f})")
        
        return cv_results
    
    def ensemble_predict(self, X_test: np.ndarray, models_to_use: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions using multiple models.
        
        Args:
            X_test: Test features
            models_to_use: List of model names to use (None for all)
            
        Returns:
            Tuple of (digit_predictions, gender_predictions)
        """
        if not self.models:
            raise ValueError("No models available. Train models first.")
        
        if models_to_use:
            available_models = [name for name in models_to_use if name in self.models]
        else:
            available_models = list(self.models.keys())
        
        if not available_models:
            raise ValueError("No available models found.")
        
        print(f"Making ensemble predictions using {available_models}")
        
        digit_predictions = []
        gender_predictions = []
        
        for model_name in available_models:
            if model_name == 'neural_network':
                # Handle neural network predictions
                digit_pred_proba, gender_pred_proba = self.models[model_name]['model'].predict(X_test)
                digit_pred = np.argmax(digit_pred_proba, axis=1)
                gender_pred = np.argmax(gender_pred_proba, axis=1)
            else:
                # Handle traditional models
                digit_pred = self.models[model_name]['digit_model'].predict(X_test)
                gender_pred = self.models[model_name]['gender_model'].predict(X_test)
            
            digit_predictions.append(digit_pred)
            gender_predictions.append(gender_pred)
        
        # Average predictions (voting)
        digit_ensemble = np.round(np.mean(digit_predictions, axis=0)).astype(int)
        gender_ensemble = np.round(np.mean(gender_predictions, axis=0)).astype(int)
        
        return digit_ensemble, gender_ensemble
    
    def save_training_results(self, results: Dict[str, Any], filename: str = "training_results.pkl") -> None:
        """
        Save training results to disk.
        
        Args:
            results: Training results dictionary
            filename: Filename to save results
        """
        # Remove models from results to avoid serialization issues
        results_to_save = {}
        for model_name, model_results in results.items():
            results_to_save[model_name] = {}
            for key, value in model_results.items():
                if key not in ['digit_model', 'gender_model', 'model']:
                    results_to_save[model_name][key] = value
        
        filepath = os.path.join(self.output_dir, filename)
        joblib.dump(results_to_save, filepath)
        print(f"Training results saved to {filepath}")
    
    def load_training_results(self, filename: str = "training_results.pkl") -> Dict[str, Any]:
        """
        Load training results from disk.
        
        Args:
            filename: Filename to load results from
            
        Returns:
            Dictionary containing training results
        """
        filepath = os.path.join(self.output_dir, filename)
        results = joblib.load(filepath)
        print(f"Training results loaded from {filepath}")
        return results


def train_multi_output_cnn_pipeline(dataset_path: str, output_dir: str = "output",
                                   sample_size: Optional[int] = None,
                                   epochs: int = 100, batch_size: int = 32) -> MultiOutputCNNTrainer:
    """
    Complete Multi-Output CNN training pipeline with 70/30 train-test split.
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save models and results
        sample_size: Optional limit on number of files to process
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained MultiOutputCNNTrainer object
    """
    print("=== Multi-Output CNN Training Pipeline ===")
    
    # Import here to avoid circular imports
    from extract_features import create_cnn_dataset
    
    # Create CNN dataset with MFCC features
    print("Creating CNN dataset with MFCC features...")
    mfcc_features, digit_labels, gender_labels = create_cnn_dataset(
        dataset_path, os.path.join(output_dir, 'cnn_data'), sample_size
    )
    
    # 70/30 Train-test split
    from sklearn.model_selection import train_test_split
    
    print("Applying 70/30 train-test split...")
    X_train, X_test, y_gender_train, y_gender_test, y_digit_train, y_digit_test = train_test_split(
        mfcc_features, gender_labels, digit_labels,
        test_size=0.3,  # 30% for testing, 70% for training
        random_state=42,
        stratify=np.argmax(gender_labels, axis=1)  # Stratify by gender
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(mfcc_features)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(mfcc_features)*100:.1f}%)")
    print(f"MFCC features shape: {X_train.shape[1:]}")
    
    # Initialize trainer
    trainer = MultiOutputCNNTrainer(output_dir)
    
    # Train Multi-Output CNN
    results = trainer.train_multi_output_cnn(
        X_train, y_gender_train, y_digit_train,
        X_test, y_gender_test, y_digit_test,
        epochs=epochs, batch_size=batch_size
    )
    
    # Plot training history
    trainer.plot_training_history(os.path.join(output_dir, 'training_history.png'))
    
    # Save training results
    joblib.dump(results, os.path.join(output_dir, 'cnn_training_results.pkl'))
    
    print(f"\n=== Training Pipeline Complete ===")
    print(f"✅ MFCC features extracted and processed")
    print(f"✅ 70/30 train-test split applied")
    print(f"✅ Multi-Output CNN trained with categorical crossentropy")
    print(f"✅ Training history visualized")
    print(f"✅ Best overall accuracy: {trainer.best_accuracy:.4f}")
    
    return trainer


if __name__ == "__main__":
    # Example usage for Multi-Output CNN
    dataset_path = "Dataset"
    
    # Run complete CNN training pipeline
    trainer = train_multi_output_cnn_pipeline(
        dataset_path=dataset_path,
        output_dir="output",
        sample_size=100,  # Use small sample for testing
        epochs=50,
        batch_size=32
    )
    
    print("Multi-Output CNN training completed successfully!")
