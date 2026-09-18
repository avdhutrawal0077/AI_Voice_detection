import librosa
import numpy as np
import io
from core.config import settings

def load_audio_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decodes uploaded audio bytes to a mono, 16kHz float32 numpy array.
    """
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=settings.sample_rate, mono=True)
    return audio

def is_speech_present(audio_window: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Simple RMS energy-based VAD.
    """
    if len(audio_window) == 0:
        return False
    rms = librosa.feature.rms(y=audio_window)
    mean_energy = float(np.mean(rms))
    return mean_energy > threshold

def extract_features(audio_window: np.ndarray) -> np.ndarray:
    """
    Extracts MFCCs and Mel Spectrogram, returning a flattened 1D summary vector.
    """
    # 13 MFCCs
    mfccs = librosa.feature.mfcc(y=audio_window, sr=settings.sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # 128-band Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_window, sr=settings.sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_mean = np.mean(mel_spec_db, axis=1)
    
    # Concatenate into a single 141-dimensional vector
    return np.concatenate((mfccs_mean, mel_spec_mean))

def generate_windows(audio: np.ndarray):
    """
    Generator that yields overlapping audio windows.
    """
    window_length = int(settings.sample_rate * settings.window_duration_sec)
    hop_length = int(settings.sample_rate * settings.hop_duration_sec)
    
    for start in range(0, len(audio) - window_length + 1, hop_length):
        yield audio[start : start + window_length]