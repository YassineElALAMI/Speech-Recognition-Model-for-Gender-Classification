"""
MFCC Feature extraction module for Multi-Output CNN.
Extracts Mel-Frequency Cepstral Coefficients (MFCCs) with fixed sequence length for CNN training.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MFCCCNNFeatureExtractor:
    """
    MFCC feature extractor for Multi-Output CNN.
    
    Extracts MFCC features with fixed sequence length for CNN training:
    - MFCC features (Mel-frequency cepstral coefficients)
    - Standardized input shape by padding/truncating sequences to fixed length
    - Returns numpy arrays of shape (n_mfcc, time, 1)
    """
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 13, n_fft: int = 2048, 
                 hop_length: int = 512, max_frames: int = 100):
        """
        Initialize the CNN feature extractor.
        
        Args:
            sr: Sample rate for audio loading
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            max_frames: Maximum number of time frames (for padding/truncating)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with preprocessing.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sr)
            # Normalize audio
            y = librosa.util.normalize(y)
            return y, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.array([]), self.sr
    
    def extract_mfcc_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of MFCC features
        """
        features = {}
        
        # Extract MFCC coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, 
                                   n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Statistical features for each MFCC coefficient
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            features[f'mfcc_{i}_skew'] = skew(mfcc[i])
            features[f'mfcc_{i}_kurtosis'] = kurtosis(mfcc[i])
            features[f'mfcc_{i}_min'] = np.min(mfcc[i])
            features[f'mfcc_{i}_max'] = np.max(mfcc[i])
        
        # Delta and delta-delta MFCC features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        for i in range(self.n_mfcc):
            features[f'delta_mfcc_{i}_mean'] = np.mean(delta_mfcc[i])
            features[f'delta_mfcc_{i}_std'] = np.std(delta_mfcc[i])
            features[f'delta2_mfcc_{i}_mean'] = np.mean(delta2_mfcc[i])
            features[f'delta2_mfcc_{i}_std'] = np.std(delta2_mfcc[i])
        
        return features
    
    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Statistical measures for spectral features
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_skew'] = skew(spectral_centroids)
        
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        features['spectral_bandwidth_skew'] = skew(spectral_bandwidth)
        
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        features['spectral_rolloff_skew'] = skew(spectral_rolloff)
        
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)
        features['zero_crossing_rate_skew'] = skew(zero_crossing_rate)
        
        # Spectral contrast features
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])
        
        return features
    
    def extract_rhythm_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract rhythm and tempo features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of rhythm features
        """
        features = {}
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        features['tempo'] = tempo
        
        # Rhythm patterns
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            features['beat_interval_mean'] = np.mean(beat_intervals)
            features['beat_interval_std'] = np.std(beat_intervals)
            features['beat_interval_cv'] = np.std(beat_intervals) / np.mean(beat_intervals) if np.mean(beat_intervals) > 0 else 0
        else:
            features['beat_interval_mean'] = 0
            features['beat_interval_std'] = 0
            features['beat_interval_cv'] = 0
        
        # Onset strength
        onset_frames = librosa.onset.onset_detect(y=y, sr=self.sr, hop_length=self.hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        if len(onset_times) > 1:
            onset_intervals = np.diff(onset_times)
            features['onset_rate'] = len(onset_times) / (len(y) / self.sr)
            features['onset_interval_mean'] = np.mean(onset_intervals)
            features['onset_interval_std'] = np.std(onset_intervals)
        else:
            features['onset_rate'] = 0
            features['onset_interval_mean'] = 0
            features['onset_interval_std'] = 0
        
        return features
    
    def extract_energy_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract energy-related features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of energy features
        """
        features = {}
        
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_skew'] = skew(rms)
        
        # Energy
        energy = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = np.mean(energy)
        features['energy_std'] = np.std(energy)
        features['energy_max'] = np.max(energy)
        features['energy_min'] = np.min(energy)
        
        # Energy entropy
        energy_entropy = -np.sum(energy * np.log(energy + 1e-10))
        features['energy_entropy'] = energy_entropy
        
        # Short-time energy
        frame_length = self.n_fft
        hop_length = self.hop_length
        st_energy = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            st_energy.append(np.sum(frame ** 2))
        
        st_energy = np.array(st_energy)
        features['short_time_energy_mean'] = np.mean(st_energy)
        features['short_time_energy_std'] = np.std(st_energy)
        features['short_time_energy_max'] = np.max(st_energy)
        features['short_time_energy_min'] = np.min(st_energy)
        
        return features
    
    def extract_pitch_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features (important for gender classification).
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of pitch features
        """
        features = {}
        
        # Pitch estimation using YIN algorithm
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr, hop_length=self.hop_length)
        
        # Extract pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            pitch_values = np.array(pitch_values)
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
            features['pitch_median'] = np.median(pitch_values)
            features['pitch_skew'] = skew(pitch_values)
            
            # Fundamental frequency features
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=self.sr)
            f0_clean = f0[f0 > 0]
            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['f0_median'] = np.median(f0_clean)
                features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
            else:
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['f0_median'] = 0
                features['f0_range'] = 0
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_median'] = 0
            features['pitch_skew'] = 0
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_median'] = 0
            features['f0_range'] = 0
        
        return features
    
    def extract_formant_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract formant features (important for gender classification).
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of formant features
        """
        features = {}
        
        # Pre-emphasize the signal
        y_preemph = librosa.effects.preemphasis(y)
        
        # Apply windowing and compute LPC coefficients
        frame_length = self.n_fft
        hop_length = self.hop_length
        
        formant_frequencies = []
        
        for i in range(0, len(y_preemph) - frame_length, hop_length):
            frame = y_preemph[i:i + frame_length]
            
            # Compute LPC coefficients
            lpc_coeffs = librosa.lpc(frame, order=12)
            
            # Find roots of the LPC polynomial
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            
            # Convert to frequencies
            angles = np.angle(roots)
            freqs = angles * self.sr / (2 * np.pi)
            freqs = np.sort(freqs)
            
            # Extract first few formants
            formant_frequencies.append(freqs[:4])  # F1, F2, F3, F4
        
        if formant_frequencies:
            formant_frequencies = np.array(formant_frequencies)
            
            for i in range(4):
                if i < formant_frequencies.shape[1]:
                    formant_data = formant_frequencies[:, i]
                    formant_data = formant_data[formant_data > 0]  # Remove zeros
                    
                    if len(formant_data) > 0:
                        features[f'formant_{i+1}_mean'] = np.mean(formant_data)
                        features[f'formant_{i+1}_std'] = np.std(formant_data)
                        features[f'formant_{i+1}_median'] = np.median(formant_data)
                    else:
                        features[f'formant_{i+1}_mean'] = 0
                        features[f'formant_{i+1}_std'] = 0
                        features[f'formant_{i+1}_median'] = 0
                else:
                    features[f'formant_{i+1}_mean'] = 0
                    features[f'formant_{i+1}_std'] = 0
                    features[f'formant_{i+1}_median'] = 0
        
        return features
    
    def extract_chroma_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract chroma features (harmonic content).
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of chroma features
        """
        features = {}
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, hop_length=self.hop_length)
        
        # Statistical features for each chroma bin
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_std'] = np.std(chroma[i])
            features[f'chroma_{i}_max'] = np.max(chroma[i])
        
        # Chroma energy normalized
        chroma_norm = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        features['chroma_energy_mean'] = np.mean(chroma_norm)
        features['chroma_energy_std'] = np.std(chroma_norm)
        
        return features
    
    def extract_temporal_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Duration
        features['duration'] = len(y) / self.sr
        
        # Silence ratio
        silence_threshold = 0.01
        silence_frames = np.sum(np.abs(y) < silence_threshold)
        features['silence_ratio'] = silence_frames / len(y)
        
        # Voice activity detection
        intervals = librosa.effects.split(y, top_db=20)
        features['num_speech_segments'] = len(intervals)
        
        if len(intervals) > 0:
            segment_lengths = [end - start for start, end in intervals]
            features['avg_segment_length'] = np.mean(segment_lengths)
            features['max_segment_length'] = np.max(segment_lengths)
            features['min_segment_length'] = np.min(segment_lengths)
        else:
            features['avg_segment_length'] = 0
            features['max_segment_length'] = 0
            features['min_segment_length'] = 0
        
        return features
    
    def extract_mfcc_for_cnn(self, file_path: str) -> np.ndarray:
        """
        Extract MFCC features for CNN with fixed sequence length.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            MFCC array of shape (n_mfcc, max_frames) for CNN input
        """
        y, sr = self.load_audio(file_path)
        
        if len(y) == 0:
            print(f"Warning: Could not load audio from {file_path}")
            return np.zeros((self.n_mfcc, self.max_frames))
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, 
                                   n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < self.max_frames:
            # Pad with zeros
            pad_width = self.max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif mfcc.shape[1] > self.max_frames:
            # Truncate
            mfcc = mfcc[:, :self.max_frames]
        
        return mfcc
    
    def extract_features_batch(self, file_paths: List[str], 
                              progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Extract features from multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            DataFrame containing features for all files
        """
        features_list = []
        
        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i, len(file_paths))
            
            features = self.extract_all_features(file_path)
            if features:
                features_list.append(features)
        
        if progress_callback:
            progress_callback(len(file_paths), len(file_paths))
        
        return pd.DataFrame(features_list)


def load_dataset_for_multi_output_cnn(dataset_path: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load dataset structure and create file paths with both digit and gender labels.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (file_paths, digit_labels, gender_labels)
    """
    file_paths = []
    digit_labels = []
    gender_labels = []
    
    # Iterate through digit directories (d0 to d9)
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
                    file_paths.append(file_path)
                    digit_labels.append(digit)
                    gender_labels.append(gender)
    
    return file_paths, digit_labels, gender_labels


def create_cnn_dataset(dataset_path: str, output_dir: str = 'cnn_data',
                      sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset for Multi-Output CNN with MFCC features.
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save the dataset
        sample_size: Optional limit on number of files to process
        
    Returns:
        Tuple of (mfcc_features, digit_labels, gender_labels)
        - mfcc_features: numpy array of shape (n_samples, n_mfcc, max_frames, 1)
        - digit_labels: numpy array of shape (n_samples, 10) - one-hot encoded
        - gender_labels: numpy array of shape (n_samples, 2) - one-hot encoded
    """
    print("Loading dataset structure for Multi-Output CNN...")
    file_paths, digit_labels, gender_labels = load_dataset_for_multi_output_cnn(dataset_path)
    
    print(f"Found {len(file_paths)} audio files")
    
    if sample_size and sample_size < len(file_paths):
        # Sample randomly if sample_size is specified
        indices = np.random.choice(len(file_paths), sample_size, replace=False)
        file_paths = [file_paths[i] for i in indices]
        digit_labels = [digit_labels[i] for i in indices]
        gender_labels = [gender_labels[i] for i in indices]
        print(f"Sampling {sample_size} files for processing")
    
    # Initialize MFCC feature extractor
    extractor = MFCCCNNFeatureExtractor()
    
    print("Extracting MFCC features for CNN...")
    mfcc_features = []
    
    for i, file_path in enumerate(file_paths):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing file {i+1}/{len(file_paths)}")
        
        mfcc = extractor.extract_mfcc_for_cnn(file_path)
        # Reshape for CNN input: (n_mfcc, max_frames) -> (n_mfcc, max_frames, 1)
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc_features.append(mfcc)
    
    # Convert to numpy arrays
    mfcc_features = np.array(mfcc_features)
    
    # One-hot encode labels
    from tensorflow.keras.utils import to_categorical
    
    # Gender labels: [1,0] = male, [0,1] = female
    gender_mapping = {'male': 0, 'female': 1}
    gender_encoded = [gender_mapping[g] for g in gender_labels]
    gender_onehot = to_categorical(gender_encoded, num_classes=2)
    
    # Digit labels: one-hot vector of length 10
    digit_onehot = to_categorical(digit_labels, num_classes=10)
    
    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'mfcc_features.npy'), mfcc_features)
    np.save(os.path.join(output_dir, 'digit_labels.npy'), digit_onehot)
    np.save(os.path.join(output_dir, 'gender_labels.npy'), gender_onehot)
    
    print(f"CNN dataset saved to {output_dir}")
    print(f"MFCC features shape: {mfcc_features.shape}")
    print(f"Digit labels shape: {digit_onehot.shape}")
    print(f"Gender labels shape: {gender_onehot.shape}")
    
    return mfcc_features, digit_onehot, gender_onehot


def analyze_features(features_df: pd.DataFrame) -> None:
    """
    Analyze and visualize the extracted features.
    
    Args:
        features_df: DataFrame containing extracted features
    """
    print("\n=== Feature Analysis ===")
    print(f"Total samples: {len(features_df)}")
    print(f"Total features: {len(features_df.columns) - 4}")  # Exclude metadata columns
    
    # Class distribution
    print("\nDigit class distribution:")
    print(features_df['digit_label'].value_counts().sort_index())
    
    print("\nGender class distribution:")
    print(features_df['gender_label'].value_counts())
    
    # Feature statistics
    feature_cols = [col for col in features_df.columns 
                   if col not in ['file_path', 'digit_label', 'gender_label', 'gender_encoded']]
    
    print(f"\nFeature statistics:")
    print(features_df[feature_cols].describe())
    
    # Check for missing values
    missing_values = features_df[feature_cols].isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found")


if __name__ == "__main__":
    # Example usage for Multi-Output CNN
    dataset_path = "Dataset"
    
    # Create CNN dataset
    print("Creating Multi-Output CNN dataset...")
    mfcc_features, digit_labels, gender_labels = create_cnn_dataset(
        dataset_path, "cnn_data", sample_size=100  # Use small sample for testing
    )
    
    print(f"\nDataset created successfully!")
    print(f"MFCC features shape: {mfcc_features.shape}")
    print(f"Digit labels shape: {digit_labels.shape}")
    print(f"Gender labels shape: {gender_labels.shape}")
    print(f"Sample digit label: {digit_labels[0]} (digit: {np.argmax(digit_labels[0])})")
    print(f"Sample gender label: {gender_labels[0]} (gender: {'male' if np.argmax(gender_labels[0]) == 0 else 'female'})")
