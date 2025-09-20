"""
Dataset preparation module for gender classification.
Handles data loading, preprocessing, and 70/30 train-test split for gender classification.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

from extract_features import MFCCFeatureExtractor, load_dataset_for_gender_classification, create_gender_classification_dataset
from utils import validate_dataset_structure, get_dataset_statistics, print_dataset_statistics


class GenderDatasetPreparer:
    """
    Dataset preparation class for gender classification with 70/30 train-test split.
    """
    
    def __init__(self, dataset_path: str, output_dir: str = "output"):
        """
        Initialize the dataset preparer.
        
        Args:
            dataset_path: Path to the dataset directory
            output_dir: Directory to save processed data
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.feature_extractor = MFCCFeatureExtractor()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage for gender classification
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
    def validate_dataset(self) -> bool:
        """
        Validate the dataset structure.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        print("Validating dataset structure...")
        return validate_dataset_structure(self.dataset_path)
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        print("Gathering dataset information...")
        stats = get_dataset_statistics(self.dataset_path)
        print_dataset_statistics(stats)
        return stats
    
    def extract_features(self, sample_size: Optional[int] = None, 
                        force_extract: bool = False) -> pd.DataFrame:
        """
        Extract MFCC features from audio files for gender classification.
        
        Args:
            sample_size: Optional limit on number of files to process
            force_extract: Force re-extraction even if features file exists
            
        Returns:
            DataFrame containing extracted features for gender classification
        """
        features_path = os.path.join(self.output_dir, "gender_features.csv")
        
        # Check if features already exist
        if os.path.exists(features_path) and not force_extract:
            print(f"Loading existing gender classification features from {features_path}")
            self.features_df = pd.read_csv(features_path)
            return self.features_df
        
        print("Extracting MFCC features for gender classification...")
        self.features_df = create_gender_classification_dataset(
            self.dataset_path, 
            features_path, 
            sample_size
        )
        
        return self.features_df
    
    def preprocess_features(self, features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess extracted features.
        
        Args:
            features_df: DataFrame containing features (uses self.features_df if None)
            
        Returns:
            Preprocessed DataFrame
        """
        if features_df is None:
            features_df = self.features_df
        
        if features_df is None:
            raise ValueError("No features DataFrame available. Run extract_features() first.")
        
        print("Preprocessing features...")
        
        # Get feature columns (exclude metadata and labels)
        exclude_cols = ['file_path', 'gender_label', 'gender_encoded', 
                       'sample_rate', 'audio_length']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Handle missing values
        missing_before = features_df[feature_cols].isnull().sum().sum()
        if missing_before > 0:
            print(f"Found {missing_before} missing values. Filling with median...")
            features_df[feature_cols] = features_df[feature_cols].fillna(
                features_df[feature_cols].median()
            )
        
        # Handle infinite values
        inf_count = np.isinf(features_df[feature_cols]).sum().sum()
        if inf_count > 0:
            print(f"Found {inf_count} infinite values. Replacing with large finite values...")
            features_df[feature_cols] = features_df[feature_cols].replace(
                [np.inf, -np.inf], [1e10, -1e10]
            )
        
        # Remove outliers using IQR method
        for col in feature_cols:
            Q1 = features_df[col].quantile(0.25)
            Q3 = features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (features_df[col] < lower_bound) | (features_df[col] > upper_bound)
            if outliers.sum() > 0:
                print(f"Removing {outliers.sum()} outliers from {col}")
                features_df.loc[outliers, col] = features_df[col].median()
        
        # Log transform highly skewed features
        skewed_features = []
        for col in feature_cols:
            skewness = features_df[col].skew()
            if abs(skewness) > 2:
                skewed_features.append(col)
        
        if skewed_features:
            print(f"Applying log transform to {len(skewed_features)} skewed features")
            for col in skewed_features:
                # Add small constant to avoid log(0)
                features_df[f'{col}_log'] = np.log1p(features_df[col] - features_df[col].min() + 1)
        
        print("Feature preprocessing completed.")
        return features_df
    
    def prepare_70_30_split(self, random_state: int = 42) -> Tuple:
        """
        Prepare 70/30 train-test split for gender classification.
        
        Args:
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names)
        """
        if self.features_df is None:
            raise ValueError("No features DataFrame available. Run extract_features() first.")
        
        print("Preparing 70/30 train-test split for gender classification...")
        
        # Get feature columns
        exclude_cols = ['file_path', 'gender_label', 'gender_encoded', 
                       'sample_rate', 'audio_length']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Prepare features and gender labels
        X = self.features_df[feature_cols].values
        y_gender = self.features_df['gender_encoded'].values
        
        # Split the data with 70/30 split (test_size=0.3)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_gender,
            test_size=0.3,  # 30% for testing, 70% for training
            random_state=random_state,
            stratify=y_gender  # Stratify by gender for balanced split
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"70/30 Split Results:")
        print(f"  Training set: {X_train_scaled.shape[0]} samples ({X_train_scaled.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test set: {X_test_scaled.shape[0]} samples ({X_test_scaled.shape[0]/len(X)*100:.1f}%)")
        print(f"  Features: {X_train_scaled.shape[1]}")
        print(f"  Training - Male: {np.sum(y_train == 0)}, Female: {np.sum(y_train == 1)}")
        print(f"  Test - Male: {np.sum(y_test == 0)}, Female: {np.sum(y_test == 1)}")
        
        return (X_train_scaled, X_test_scaled, y_train, y_test, self.scaler, feature_cols)
    
    def prepare_cross_validation_splits(self, n_splits: int = 5, 
                                      random_state: int = 42,
                                      stratify_by: str = 'digit_label') -> List:
        """
        Prepare cross-validation splits.
        
        Args:
            n_splits: Number of CV folds
            random_state: Random state for reproducibility
            stratify_by: Column to use for stratification
            
        Returns:
            List of CV splits
        """
        if self.features_df is None:
            raise ValueError("No features DataFrame available. Run extract_features() first.")
        
        print(f"Preparing {n_splits}-fold cross-validation splits...")
        
        # Get feature columns
        exclude_cols = ['file_path', 'digit_label', 'gender_label', 'gender_encoded', 
                       'sample_rate', 'audio_length']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Prepare features and targets
        X = self.features_df[feature_cols].values
        y_digit = self.features_df['digit_label'].values
        y_gender = self.features_df['gender_encoded'].values
        
        # Create stratification target
        if stratify_by == 'digit_label':
            stratify_target = y_digit
        elif stratify_by == 'gender_label':
            stratify_target = y_gender
        else:
            stratify_target = y_digit * 2 + y_gender
        
        # Create stratified K-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        cv_splits = []
        for train_idx, val_idx in skf.split(X, stratify_target):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_digit_train_cv, y_digit_val_cv = y_digit[train_idx], y_digit[val_idx]
            y_gender_train_cv, y_gender_val_cv = y_gender[train_idx], y_gender[val_idx]
            
            # Scale features for this fold
            scaler_cv = StandardScaler()
            X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
            X_val_cv_scaled = scaler_cv.transform(X_val_cv)
            
            cv_splits.append({
                'X_train': X_train_cv_scaled,
                'X_val': X_val_cv_scaled,
                'y_train': {'digit': y_digit_train_cv, 'gender': y_gender_train_cv},
                'y_val': {'digit': y_digit_val_cv, 'gender': y_gender_val_cv},
                'scaler': scaler_cv
            })
        
        print(f"Created {len(cv_splits)} cross-validation splits")
        return cv_splits
    
    def save_preprocessed_data(self, filename: str = "preprocessed_data.pkl") -> None:
        """
        Save preprocessed data to disk.
        
        Args:
            filename: Name of the file to save
        """
        import joblib
        
        data_to_save = {
            'features_df': self.features_df,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'scaler': self.scaler
        }
        
        filepath = os.path.join(self.output_dir, filename)
        joblib.dump(data_to_save, filepath)
        print(f"Preprocessed data saved to {filepath}")
    
    def load_preprocessed_data(self, filename: str = "preprocessed_data.pkl") -> None:
        """
        Load preprocessed data from disk.
        
        Args:
            filename: Name of the file to load
        """
        import joblib
        
        filepath = os.path.join(self.output_dir, filename)
        data = joblib.load(filepath)
        
        self.features_df = data['features_df']
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.scaler = data['scaler']
        
        print(f"Preprocessed data loaded from {filepath}")
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of features.
        
        Returns:
            DataFrame with feature summary statistics
        """
        if self.features_df is None:
            raise ValueError("No features DataFrame available. Run extract_features() first.")
        
        exclude_cols = ['file_path', 'digit_label', 'gender_label', 'gender_encoded', 
                       'sample_rate', 'audio_length']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        summary = self.features_df[feature_cols].describe()
        summary.loc['skewness'] = self.features_df[feature_cols].skew()
        summary.loc['kurtosis'] = self.features_df[feature_cols].kurtosis()
        
        return summary
    
    def analyze_feature_correlations(self, threshold: float = 0.9) -> pd.DataFrame:
        """
        Analyze feature correlations and identify highly correlated features.
        
        Args:
            threshold: Correlation threshold for identifying highly correlated features
            
        Returns:
            DataFrame with correlation pairs above threshold
        """
        if self.features_df is None:
            raise ValueError("No features DataFrame available. Run extract_features() first.")
        
        exclude_cols = ['file_path', 'digit_label', 'gender_label', 'gender_encoded', 
                       'sample_rate', 'audio_length']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = self.features_df[feature_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        high_corr_df = pd.DataFrame(high_corr_pairs)
        if len(high_corr_df) > 0:
            high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
        
        print(f"Found {len(high_corr_df)} feature pairs with correlation >= {threshold}")
        return high_corr_df
    
    def remove_highly_correlated_features(self, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features to reduce dimensionality.
        
        Args:
            threshold: Correlation threshold for removing features
            
        Returns:
            List of features to remove
        """
        if self.features_df is None:
            raise ValueError("No features DataFrame available. Run extract_features() first.")
        
        exclude_cols = ['file_path', 'digit_label', 'gender_label', 'gender_encoded', 
                       'sample_rate', 'audio_length']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = self.features_df[feature_cols].corr()
        
        # Find features to remove
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    # Remove the feature with lower variance
                    var_i = self.features_df[corr_matrix.columns[i]].var()
                    var_j = self.features_df[corr_matrix.columns[j]].var()
                    
                    if var_i < var_j:
                        to_remove.add(corr_matrix.columns[i])
                    else:
                        to_remove.add(corr_matrix.columns[j])
        
        to_remove = list(to_remove)
        print(f"Removing {len(to_remove)} highly correlated features")
        
        # Remove features from DataFrame
        if to_remove:
            self.features_df = self.features_df.drop(columns=to_remove)
        
        return to_remove


def prepare_gender_classification_pipeline(dataset_path: str, output_dir: str = "output",
                                         sample_size: Optional[int] = None) -> GenderDatasetPreparer:
    """
    Complete dataset preparation pipeline for gender classification with 70/30 split.
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save processed data
        sample_size: Optional limit on number of files to process
        
    Returns:
        Prepared GenderDatasetPreparer object
    """
    print("=== Gender Classification Dataset Preparation Pipeline ===")
    
    # Initialize preparer
    preparer = GenderDatasetPreparer(dataset_path, output_dir)
    
    # Validate dataset
    if not preparer.validate_dataset():
        raise ValueError("Dataset validation failed")
    
    # Get dataset info
    preparer.get_dataset_info()
    
    # Extract MFCC features for gender classification
    preparer.extract_features(sample_size=sample_size)
    
    # Preprocess features
    preparer.preprocess_features()
    
    # Prepare 70/30 train-test split
    preparer.prepare_70_30_split()
    
    # Save preprocessed data
    preparer.save_preprocessed_data()
    
    print("Gender classification dataset preparation completed successfully!")
    print("✅ MFCC features extracted and organized by gender")
    print("✅ 70/30 train-test split applied")
    return preparer


if __name__ == "__main__":
    # Example usage for gender classification
    dataset_path = "Dataset"
    
    # Run complete preparation pipeline for gender classification
    preparer = prepare_gender_classification_pipeline(
        dataset_path=dataset_path,
        output_dir="output",
        sample_size=100  # Use small sample for testing
    )
    
    # Print feature summary
    print("\nFeature Summary:")
    print(preparer.get_feature_summary())
    
    # Analyze correlations
    print("\nHigh Correlation Analysis:")
    high_corr = preparer.analyze_feature_correlations(threshold=0.9)
    print(high_corr.head(10))
