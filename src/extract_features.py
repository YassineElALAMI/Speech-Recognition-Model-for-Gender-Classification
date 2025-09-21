import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, max_len=174, sr=16000):
    """
    Extract MFCC features from an audio file.
    Args:
        file_path (str): Path to .wav file
        n_mfcc (int): Number of MFCCs
        max_len (int): Fixed length for padding/truncating (frames)
        sr (int): target sampling rate for loading audio
    Returns:
        np.ndarray or None: MFCC feature array with shape (n_mfcc, max_len) dtype float32, or None on error
    """
    try:
        # load with a fixed sample rate and mono to ensure consistent features
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Pad or truncate in the time (frame) dimension to ensure consistent shape
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        # Normalize per-feature (zero mean, unit variance)
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)

        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
