"""
Utility functions for the speech recognition and gender classification project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')


def create_output_directory(output_dir: str = "output") -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory name to create
        
    Returns:
        Path to the created directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def save_model(model, model_path: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_path: Path to save the model
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path: str):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def plot_feature_distribution(features_df: pd.DataFrame, feature_cols: list, 
                            target_col: str, title: str = "Feature Distribution") -> None:
    """
    Plot distribution of features grouped by target variable.
    
    Args:
        features_df: DataFrame containing features and target
        feature_cols: List of feature column names to plot
        target_col: Name of target column
        title: Plot title
    """
    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        if i < len(axes):
            for target_value in features_df[target_col].unique():
                subset = features_df[features_df[target_col] == target_value]
                axes[i].hist(subset[feature], alpha=0.7, label=target_value, bins=30)
            
            axes[i].set_title(f'{feature}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(features_df: pd.DataFrame, feature_cols: list, 
                          title: str = "Feature Correlation Matrix") -> None:
    """
    Plot correlation matrix of features.
    
    Args:
        features_df: DataFrame containing features
        feature_cols: List of feature column names
        title: Plot title
    """
    # Calculate correlation matrix
    corr_matrix = features_df[feature_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels: np.ndarray, title: str = "Class Distribution") -> None:
    """
    Plot distribution of class labels.
    
    Args:
        labels: Array of class labels
        title: Plot title
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(unique_labels[i], count + max(counts) * 0.01, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: list, title: str = "Confusion Matrix") -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, title: str = "Training History") -> None:
    """
    Plot training history for neural networks.
    
    Args:
        history: Training history dictionary
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy plots
    axes[0, 0].plot(history['accuracy'], label='Training')
    if 'val_accuracy' in history:
        axes[0, 0].plot(history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plots
    axes[0, 1].plot(history['loss'], label='Training')
    if 'val_loss' in history:
        axes[0, 1].plot(history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Digit accuracy (if multi-output)
    if 'digit_accuracy' in history:
        axes[1, 0].plot(history['digit_accuracy'], label='Training')
        if 'val_digit_accuracy' in history:
            axes[1, 0].plot(history['val_digit_accuracy'], label='Validation')
        axes[1, 0].set_title('Digit Classification Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Gender accuracy (if multi-output)
    if 'gender_accuracy' in history:
        axes[1, 1].plot(history['gender_accuracy'], label='Training')
        if 'val_gender_accuracy' in history:
            axes[1, 1].plot(history['val_gender_accuracy'], label='Validation')
        axes[1, 1].set_title('Gender Classification Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def prepare_features_for_training(features_df: pd.DataFrame, 
                                 feature_cols: list,
                                 target_cols: list,
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> tuple:
    """
    Prepare features for training by scaling and splitting data.
    
    Args:
        features_df: DataFrame containing features and targets
        feature_cols: List of feature column names
        target_cols: List of target column names
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Extract features and targets
    X = features_df[feature_cols].values
    y = features_df[target_cols].values if len(target_cols) > 1 else features_df[target_cols[0]].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(target_cols) == 1 else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_multi_output_metrics(y_true_digit: np.ndarray, y_pred_digit: np.ndarray,
                                 y_true_gender: np.ndarray, y_pred_gender: np.ndarray) -> dict:
    """
    Calculate comprehensive metrics for multi-output classification.
    
    Args:
        y_true_digit: True digit labels
        y_pred_digit: Predicted digit labels
        y_true_gender: True gender labels
        y_pred_gender: Predicted gender labels
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Digit classification metrics
    metrics['digit_accuracy'] = accuracy_score(y_true_digit, y_pred_digit)
    metrics['digit_report'] = classification_report(y_true_digit, y_pred_digit, output_dict=True)
    
    # Gender classification metrics
    metrics['gender_accuracy'] = accuracy_score(y_true_gender, y_pred_gender)
    metrics['gender_report'] = classification_report(y_true_gender, y_pred_gender, output_dict=True)
    
    # Overall accuracy (both predictions correct)
    both_correct = (y_true_digit == y_pred_digit) & (y_true_gender == y_pred_gender)
    metrics['overall_accuracy'] = np.mean(both_correct)
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: list, title: str = "Classification Report") -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Report title
    """
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")


def save_results(results: dict, output_path: str) -> None:
    """
    Save results to a file.
    
    Args:
        results: Dictionary containing results
        output_path: Path to save results
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Convert results
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = convert_numpy(value)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {output_path}")


def load_results(results_path: str) -> dict:
    """
    Load results from a file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Dictionary containing results
    """
    import json
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded from {results_path}")
    return results


def get_feature_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    else:
        print("Model does not have feature_importances_ attribute")
        return pd.DataFrame()


def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance") -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def create_sample_submission(predictions_digit: np.ndarray, predictions_gender: np.ndarray,
                           output_path: str = "sample_submission.csv") -> None:
    """
    Create sample submission file for predictions.
    
    Args:
        predictions_digit: Digit predictions
        predictions_gender: Gender predictions
        output_path: Path to save submission file
    """
    submission_df = pd.DataFrame({
        'digit_prediction': predictions_digit,
        'gender_prediction': predictions_gender
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Sample submission saved to {output_path}")


def validate_dataset_structure(dataset_path: str) -> bool:
    """
    Validate that the dataset has the expected structure.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    expected_dirs = [f'd{i}' for i in range(10)]
    expected_genders = ['male', 'female']
    
    for digit_dir in expected_dirs:
        digit_path = os.path.join(dataset_path, digit_dir)
        if not os.path.exists(digit_path):
            print(f"Missing directory: {digit_dir}")
            return False
        
        for gender in expected_genders:
            gender_path = os.path.join(digit_path, gender)
            if not os.path.exists(gender_path):
                print(f"Missing directory: {digit_dir}/{gender}")
                return False
            
            # Check for .wav files
            wav_files = [f for f in os.listdir(gender_path) if f.endswith('.wav')]
            if len(wav_files) == 0:
                print(f"No .wav files found in {digit_dir}/{gender}")
                return False
    
    print("Dataset structure validation passed!")
    return True


def get_dataset_statistics(dataset_path: str) -> dict:
    """
    Get statistics about the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'total_files': 0,
        'digit_counts': {},
        'gender_counts': {'male': 0, 'female': 0},
        'files_per_digit_gender': {}
    }
    
    for digit in range(10):
        digit_dir = os.path.join(dataset_path, f'd{digit}')
        if os.path.exists(digit_dir):
            digit_count = 0
            
            for gender in ['male', 'female']:
                gender_dir = os.path.join(digit_dir, gender)
                if os.path.exists(gender_dir):
                    wav_files = [f for f in os.listdir(gender_dir) if f.endswith('.wav')]
                    file_count = len(wav_files)
                    
                    digit_count += file_count
                    stats['gender_counts'][gender] += file_count
                    stats['files_per_digit_gender'][f'd{digit}_{gender}'] = file_count
            
            stats['digit_counts'][f'd{digit}'] = digit_count
            stats['total_files'] += digit_count
    
    return stats


def print_dataset_statistics(stats: dict) -> None:
    """
    Print dataset statistics in a formatted way.
    
    Args:
        stats: Dictionary containing dataset statistics
    """
    print("\n=== Dataset Statistics ===")
    print(f"Total audio files: {stats['total_files']}")
    
    print("\nFiles per digit:")
    for digit, count in stats['digit_counts'].items():
        print(f"  {digit}: {count} files")
    
    print("\nFiles per gender:")
    for gender, count in stats['gender_counts'].items():
        print(f"  {gender}: {count} files")
    
    print("\nFiles per digit-gender combination:")
    for combination, count in stats['files_per_digit_gender'].items():
        print(f"  {combination}: {count} files")
