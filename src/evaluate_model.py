"""
Model evaluation module for Multi-Output CNN.
Provides comprehensive evaluation metrics and visualization tools for speech recognition and gender classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils import (
    plot_confusion_matrix, plot_feature_importance, print_classification_report,
    calculate_multi_output_metrics, save_results, create_sample_submission
)


class MultiOutputCNNEvaluator:
    """
    Comprehensive evaluator for Multi-Output CNN with accuracy, precision, recall, F1-score metrics.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.evaluation_results = {}
        self.best_accuracy = 0
        
    def evaluate_multi_output_cnn(self, model_results: Dict[str, Any],
                                 y_gender_test: np.ndarray, y_digit_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate Multi-Output CNN comprehensively with accuracy, precision, recall, F1-score.
        
        Args:
            model_results: Model results dictionary containing predictions
            y_gender_test: Test gender labels (n_samples, 2) - one-hot encoded
            y_digit_test: Test digit labels (n_samples, 10) - one-hot encoded
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        print("\nEvaluating Multi-Output CNN...")
        
        # Extract predictions
        gender_pred = model_results['gender_predictions']
        digit_pred = model_results['digit_predictions']
        
        # Convert one-hot back to class indices for metrics calculation
        y_gender_test_indices = np.argmax(y_gender_test, axis=1)
        y_digit_test_indices = np.argmax(y_digit_test, axis=1)
        
        # Calculate comprehensive metrics
        evaluation = {}
        
        # Gender classification metrics
        evaluation['gender'] = self._calculate_classification_metrics(
            y_gender_test_indices, gender_pred, class_names=['Male', 'Female']
        )
        
        # Digit classification metrics
        evaluation['digit'] = self._calculate_classification_metrics(
            y_digit_test_indices, digit_pred, class_names=[str(i) for i in range(10)]
        )
        
        # Overall metrics
        evaluation['overall'] = self._calculate_overall_metrics(
            y_digit_test_indices, digit_pred, y_gender_test_indices, gender_pred
        )
        
        # Store results
        self.evaluation_results['multi_output_cnn'] = evaluation
        
        # Print summary
        self._print_evaluation_summary('Multi-Output CNN', evaluation)
        
        return evaluation
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        class_names: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        
        return metrics
    
    def _calculate_overall_metrics(self, y_true_digit: np.ndarray, y_pred_digit: np.ndarray,
                                 y_true_gender: np.ndarray, y_pred_gender: np.ndarray) -> Dict[str, Any]:
        """
        Calculate overall metrics for multi-output classification.
        
        Args:
            y_true_digit: True digit labels
            y_pred_digit: Predicted digit labels
            y_true_gender: True gender labels
            y_pred_gender: Predicted gender labels
            
        Returns:
            Dictionary containing overall metrics
        """
        metrics = {}
        
        # Both predictions correct
        both_correct = (y_true_digit == y_pred_digit) & (y_true_gender == y_pred_gender)
        metrics['overall_accuracy'] = np.mean(both_correct)
        
        # Digit correct, gender incorrect
        digit_correct_gender_incorrect = (y_true_digit == y_pred_digit) & (y_true_gender != y_pred_gender)
        metrics['digit_correct_gender_incorrect'] = np.mean(digit_correct_gender_incorrect)
        
        # Gender correct, digit incorrect
        gender_correct_digit_incorrect = (y_true_gender == y_pred_gender) & (y_true_digit != y_pred_digit)
        metrics['gender_correct_digit_incorrect'] = np.mean(gender_correct_digit_incorrect)
        
        # Both incorrect
        both_incorrect = (y_true_digit != y_pred_digit) & (y_true_gender != y_pred_gender)
        metrics['both_incorrect'] = np.mean(both_incorrect)
        
        # Correlation between digit and gender accuracy
        digit_correct = (y_true_digit == y_pred_digit).astype(int)
        gender_correct = (y_true_gender == y_pred_gender).astype(int)
        metrics['digit_gender_correlation'] = np.corrcoef(digit_correct, gender_correct)[0, 1]
        
        return metrics
    
    def _print_evaluation_summary(self, model_name: str, evaluation: Dict[str, Any]) -> None:
        """
        Print evaluation summary for a model.
        
        Args:
            model_name: Name of the model
            evaluation: Evaluation results dictionary
        """
        print(f"\n=== {model_name.upper()} EVALUATION SUMMARY ===")
        
        # Digit metrics
        print(f"\nDigit Classification:")
        print(f"  Accuracy: {evaluation['digit']['accuracy']:.4f}")
        print(f"  Precision: {evaluation['digit']['precision']:.4f}")
        print(f"  Recall: {evaluation['digit']['recall']:.4f}")
        print(f"  F1-Score: {evaluation['digit']['f1_score']:.4f}")
        
        # Gender metrics
        print(f"\nGender Classification:")
        print(f"  Accuracy: {evaluation['gender']['accuracy']:.4f}")
        print(f"  Precision: {evaluation['gender']['precision']:.4f}")
        print(f"  Recall: {evaluation['gender']['recall']:.4f}")
        print(f"  F1-Score: {evaluation['gender']['f1_score']:.4f}")
        
        # Overall metrics
        print(f"\nOverall Multi-output:")
        print(f"  Overall Accuracy: {evaluation['overall']['overall_accuracy']:.4f}")
        print(f"  Digit Correct, Gender Incorrect: {evaluation['overall']['digit_correct_gender_incorrect']:.4f}")
        print(f"  Gender Correct, Digit Incorrect: {evaluation['overall']['gender_correct_digit_incorrect']:.4f}")
        print(f"  Both Incorrect: {evaluation['overall']['both_incorrect']:.4f}")
        print(f"  Digit-Gender Correlation: {evaluation['overall']['digit_gender_correlation']:.4f}")
    
    def plot_model_comparison(self, metric: str = 'overall_accuracy') -> None:
        """
        Plot comparison of models across different metrics.
        
        Args:
            metric: Metric to compare ('overall_accuracy', 'digit_accuracy', 'gender_accuracy')
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_models() first.")
            return
        
        # Extract data for plotting
        model_names = list(self.evaluation_results.keys())
        metrics_data = {}
        
        for metric_name in ['overall_accuracy', 'digit_accuracy', 'gender_accuracy']:
            metrics_data[metric_name] = []
            for model_name in model_names:
                if metric_name == 'overall_accuracy':
                    value = self.evaluation_results[model_name]['overall']['overall_accuracy']
                elif metric_name == 'digit_accuracy':
                    value = self.evaluation_results[model_name]['digit']['accuracy']
                elif metric_name == 'gender_accuracy':
                    value = self.evaluation_results[model_name]['gender']['accuracy']
                metrics_data[metric_name].append(value)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            bars = axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    
    def plot_confusion_matrices(self, model_name: str) -> None:
        """
        Plot confusion matrices for digit and gender classification.
        
        Args:
            model_name: Name of the model to plot
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        evaluation = self.evaluation_results[model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Digit confusion matrix
        digit_cm = evaluation['digit']['confusion_matrix']
        sns.heatmap(digit_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[str(i) for i in range(10)],
                   yticklabels=[str(i) for i in range(10)],
                   ax=axes[0])
        axes[0].set_title(f'{model_name.title()} - Digit Classification')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Gender confusion matrix
        gender_cm = evaluation['gender']['confusion_matrix']
        sns.heatmap(gender_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Male', 'Female'],
                   yticklabels=['Male', 'Female'],
                   ax=axes[1])
        axes[1].set_title(f'{model_name.title()} - Gender Classification')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'{model_name}_confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
    
    def plot_per_class_metrics(self, model_name: str) -> None:
        """
        Plot per-class metrics for a model.
        
        Args:
            model_name: Name of the model to plot
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        evaluation = self.evaluation_results[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Digit per-class metrics
        digit_classes = [str(i) for i in range(10)]
        digit_precision = evaluation['digit']['precision_per_class']
        digit_recall = evaluation['digit']['recall_per_class']
        digit_f1 = evaluation['digit']['f1_per_class']
        
        x = np.arange(len(digit_classes))
        width = 0.25
        
        axes[0, 0].bar(x - width, digit_precision, width, label='Precision', alpha=0.7)
        axes[0, 0].bar(x, digit_recall, width, label='Recall', alpha=0.7)
        axes[0, 0].bar(x + width, digit_f1, width, label='F1-Score', alpha=0.7)
        axes[0, 0].set_title(f'{model_name.title()} - Digit Per-Class Metrics')
        axes[0, 0].set_xlabel('Digit Class')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(digit_classes)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gender per-class metrics
        gender_classes = ['Male', 'Female']
        gender_precision = evaluation['gender']['precision_per_class']
        gender_recall = evaluation['gender']['recall_per_class']
        gender_f1 = evaluation['gender']['f1_per_class']
        
        x_gender = np.arange(len(gender_classes))
        
        axes[0, 1].bar(x_gender - width, gender_precision, width, label='Precision', alpha=0.7)
        axes[0, 1].bar(x_gender, gender_recall, width, label='Recall', alpha=0.7)
        axes[0, 1].bar(x_gender + width, gender_f1, width, label='F1-Score', alpha=0.7)
        axes[0, 1].set_title(f'{model_name.title()} - Gender Per-Class Metrics')
        axes[0, 1].set_xlabel('Gender Class')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x_gender)
        axes[0, 1].set_xticklabels(gender_classes)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overall performance breakdown
        overall = evaluation['overall']
        categories = ['Both Correct', 'Digit Correct\nGender Incorrect', 
                     'Gender Correct\nDigit Incorrect', 'Both Incorrect']
        values = [overall['overall_accuracy'], 
                 overall['digit_correct_gender_incorrect'],
                 overall['gender_correct_digit_incorrect'],
                 overall['both_incorrect']]
        
        axes[1, 0].pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title(f'{model_name.title()} - Prediction Breakdown')
        
        # Accuracy comparison
        metrics = ['Digit Accuracy', 'Gender Accuracy', 'Overall Accuracy']
        scores = [evaluation['digit']['accuracy'], 
                 evaluation['gender']['accuracy'],
                 evaluation['overall']['overall_accuracy']]
        
        bars = axes[1, 1].bar(metrics, scores, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 1].set_title(f'{model_name.title()} - Accuracy Comparison')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'{model_name}_per_class_metrics.png'), 
                   dpi=300, bbox_inches='tight')
    
    def evaluate_models(self, trained_models: Dict[str, Any], y_test: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Args:
            trained_models: Dictionary containing trained models and their results
            y_test: Test labels (dict with 'digit' and 'gender' keys)
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        print("=== MODEL EVALUATION ===")
        
        for model_name, model_results in trained_models.items():
            evaluation = self.evaluate_single_model(model_name, model_results, y_test)
            
            # Track best model
            overall_accuracy = evaluation['overall']['overall_accuracy']
            if overall_accuracy > self.best_score:
                self.best_score = overall_accuracy
                self.best_model_name = model_name
        
        # Generate comparison plots
        self.plot_model_comparison()
        
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Best model: {self.best_model_name}")
        print(f"Best overall accuracy: {self.best_score:.4f}")
        
        return self.evaluation_results
    
    def generate_detailed_report(self, model_name: str) -> str:
        """
        Generate a detailed evaluation report for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Detailed report string
        """
        if model_name not in self.evaluation_results:
            return f"No evaluation results found for {model_name}"
        
        evaluation = self.evaluation_results[model_name]
        
        report = f"""
=== DETAILED EVALUATION REPORT: {model_name.upper()} ===

DIGIT CLASSIFICATION:
- Accuracy: {evaluation['digit']['accuracy']:.4f}
- Precision: {evaluation['digit']['precision']:.4f}
- Recall: {evaluation['digit']['recall']:.4f}
- F1-Score: {evaluation['digit']['f1_score']:.4f}

GENDER CLASSIFICATION:
- Accuracy: {evaluation['gender']['accuracy']:.4f}
- Precision: {evaluation['gender']['precision']:.4f}
- Recall: {evaluation['gender']['recall']:.4f}
- F1-Score: {evaluation['gender']['f1_score']:.4f}

OVERALL MULTI-OUTPUT PERFORMANCE:
- Overall Accuracy (Both Correct): {evaluation['overall']['overall_accuracy']:.4f}
- Digit Correct, Gender Incorrect: {evaluation['overall']['digit_correct_gender_incorrect']:.4f}
- Gender Correct, Digit Incorrect: {evaluation['overall']['gender_correct_digit_incorrect']:.4f}
- Both Incorrect: {evaluation['overall']['both_incorrect']:.4f}
- Digit-Gender Correlation: {evaluation['overall']['digit_gender_correlation']:.4f}

DIGIT PER-CLASS PERFORMANCE:
"""
        
        # Add digit per-class metrics
        for i, (precision, recall, f1) in enumerate(zip(
            evaluation['digit']['precision_per_class'],
            evaluation['digit']['recall_per_class'],
            evaluation['digit']['f1_per_class']
        )):
            report += f"- Digit {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n"
        
        report += f"""
GENDER PER-CLASS PERFORMANCE:
- Male: Precision={evaluation['gender']['precision_per_class'][0]:.3f}, Recall={evaluation['gender']['recall_per_class'][0]:.3f}, F1={evaluation['gender']['f1_per_class'][0]:.3f}
- Female: Precision={evaluation['gender']['precision_per_class'][1]:.3f}, Recall={evaluation['gender']['recall_per_class'][1]:.3f}, F1={evaluation['gender']['f1_per_class'][1]:.3f}
"""
        
        return report
    
    def save_evaluation_results(self, filename: str = "evaluation_results.pkl") -> None:
        """
        Save evaluation results to disk.
        
        Args:
            filename: Filename to save results
        """
        filepath = os.path.join(self.output_dir, filename)
        joblib.dump(self.evaluation_results, filepath)
        print(f"Evaluation results saved to {filepath}")
        
        # Also save as text report
        report_path = os.path.join(self.output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== COMPREHENSIVE EVALUATION REPORT ===\n\n")
            
            for model_name in self.evaluation_results.keys():
                f.write(self.generate_detailed_report(model_name))
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Detailed report saved to {report_path}")
    
    def load_evaluation_results(self, filename: str = "evaluation_results.pkl") -> Dict[str, Any]:
        """
        Load evaluation results from disk.
        
        Args:
            filename: Filename to load results from
            
        Returns:
            Dictionary containing evaluation results
        """
        filepath = os.path.join(self.output_dir, filename)
        self.evaluation_results = joblib.load(filepath)
        print(f"Evaluation results loaded from {filepath}")
        return self.evaluation_results
    
    def create_model_ranking(self) -> pd.DataFrame:
        """
        Create a ranking of models based on different metrics.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_models() first.")
            return pd.DataFrame()
        
        ranking_data = []
        
        for model_name, evaluation in self.evaluation_results.items():
            ranking_data.append({
                'Model': model_name,
                'Overall_Accuracy': evaluation['overall']['overall_accuracy'],
                'Digit_Accuracy': evaluation['digit']['accuracy'],
                'Gender_Accuracy': evaluation['gender']['accuracy'],
                'Digit_F1': evaluation['digit']['f1_score'],
                'Gender_F1': evaluation['gender']['f1_score'],
                'Digit_Precision': evaluation['digit']['precision'],
                'Gender_Precision': evaluation['gender']['precision'],
                'Digit_Recall': evaluation['digit']['recall'],
                'Gender_Recall': evaluation['gender']['recall']
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Sort by overall accuracy
        ranking_df = ranking_df.sort_values('Overall_Accuracy', ascending=False)
        
        print("\n=== MODEL RANKING ===")
        print(ranking_df.round(4))
        
        return ranking_df


def evaluate_multi_output_cnn_pipeline(model_results: Dict[str, Any], 
                                      y_gender_test: np.ndarray, y_digit_test: np.ndarray,
                                      output_dir: str = "output") -> MultiOutputCNNEvaluator:
    """
    Complete Multi-Output CNN evaluation pipeline with accuracy, precision, recall, F1-score.
    
    Args:
        model_results: Dictionary containing trained CNN model and results
        y_gender_test: Test gender labels (n_samples, 2) - one-hot encoded
        y_digit_test: Test digit labels (n_samples, 10) - one-hot encoded
        output_dir: Directory to save evaluation results
        
    Returns:
        MultiOutputCNNEvaluator object with evaluation results
    """
    print("=== Multi-Output CNN Evaluation Pipeline ===")
    
    # Initialize evaluator
    evaluator = MultiOutputCNNEvaluator(output_dir)
    
    # Evaluate Multi-Output CNN
    evaluation_results = evaluator.evaluate_multi_output_cnn(model_results, y_gender_test, y_digit_test)
    
    # Generate detailed plots
    print("\nGenerating detailed analysis for Multi-Output CNN...")
    evaluator.plot_confusion_matrices('multi_output_cnn')
    evaluator.plot_per_class_metrics('multi_output_cnn')
    
    # Create comprehensive evaluation report
    detailed_report = evaluator.generate_detailed_report('multi_output_cnn')
    
    # Save all results
    evaluator.save_evaluation_results()
    
    # Save detailed report
    report_path = os.path.join(output_dir, "detailed_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(detailed_report)
    print(f"Detailed evaluation report saved to {report_path}")
    
    print("\n=== Evaluation Pipeline Complete ===")
    print("✅ Accuracy, precision, recall, F1-score calculated")
    print("✅ Confusion matrices generated")
    print("✅ Per-class metrics visualized")
    print("✅ Comprehensive evaluation report created")
    
    return evaluator


if __name__ == "__main__":
    # Example usage for Multi-Output CNN evaluation
    print("Multi-Output CNN evaluation module loaded successfully!")
    print("Use evaluate_multi_output_cnn_pipeline() function to run the complete evaluation pipeline.")
