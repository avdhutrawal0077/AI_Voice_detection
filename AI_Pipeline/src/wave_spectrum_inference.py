"""
Wave Spectrum � AI Voice Detection Inference Module
====================================================
Usage:
    from wave_spectrum_inference import WaveSpectrumDetector, TemporalSmoother

    detector = WaveSpectrumDetector()   # call once at server startup
    smoother = TemporalSmoother()       # one per active call session

    # For each 2-second audio window:
    result   = detector.predict(pcm_list)
    smoothed = smoother.update(result["p_fake"])
"""
import parselmouth
from parselmouth.praat import call
import os, json, time, pickle
import numpy as np
import torch
import torchaudio.transforms as T
import soundfile as sf
import xgboost as xgb
from transformers import Wav2Vec2Model

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR      = 16000
TARGET_SAMPLES = 64000
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))

def _load_config(name):
    path = os.path.join(BASE_DIR, "..", "config", name)
    with open(path) as f:
        return json.load(f)


class WaveSpectrumDetector:

    def __init__(self):
        pre_cfg = _load_config("preprocessing_config.json")
        # Thresholds are now managed externally via the core configuration.

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base", use_safetensors=True).to(DEVICE)
        self.wav2vec2.eval()
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

        self.mel_transform = T.MelSpectrogram(
            sample_rate=pre_cfg["TARGET_SR"],
            n_fft=pre_cfg["MEL_N_FFT"],
            hop_length=pre_cfg["MEL_HOP_LENGTH"],
            n_mels=pre_cfg["MEL_N_MELS"],
            f_min=pre_cfg["MEL_FMIN"],
            f_max=pre_cfg["MEL_FMAX"],
            power=2.0).to(DEVICE)

        self.amplitude_to_db = T.AmplitudeToDB(top_db=80).to(DEVICE)

        self.mfcc_transform = T.MFCC(
            sample_rate=pre_cfg["TARGET_SR"], n_mfcc=13,
            melkwargs={"n_fft": pre_cfg["MEL_N_FFT"],
                       "hop_length": pre_cfg["MEL_HOP_LENGTH"],
                       "n_mels": pre_cfg["MEL_N_MELS"],
                       "f_min": pre_cfg["MEL_FMIN"],
                       "f_max": pre_cfg["MEL_FMAX"]}).to(DEVICE)

        self.freq_bins = torch.linspace(
            pre_cfg["MEL_FMIN"], pre_cfg["MEL_FMAX"],
            pre_cfg["MEL_N_MELS"], device=DEVICE)

        self.model_a = xgb.XGBClassifier()
        self.model_a.load_model(
            os.path.join(BASE_DIR, "..", "models", "model_a_branch_a.json"))

        self.model_b = xgb.XGBClassifier()
        self.model_b.load_model(
            os.path.join(BASE_DIR, "..", "models", "model_b_branch_b.json"))

        with open(os.path.join(BASE_DIR, "..", "models",
                               "model_c_fusion.pkl"), "rb") as f:
            self.model_c = pickle.load(f)

        with open(os.path.join(BASE_DIR, "..", "models",
                               "branch_b_scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        self._pre_cfg = pre_cfg

    def preprocess(self, pcm_input):
        audio = torch.tensor(
            np.array(pcm_input, dtype=np.float32)).unsqueeze(0)
        peak  = audio.abs().max()
        if peak > 1e-6:
            audio = audio / peak
        if audio.shape[1] < TARGET_SAMPLES:
            audio = torch.nn.functional.pad(
                audio, (0, TARGET_SAMPLES - audio.shape[1]))
        else:
            audio = audio[:, :TARGET_SAMPLES]
        return audio.to(DEVICE)

    def extract_branch_a(self, audio):
        with torch.no_grad():
            out = self.wav2vec2(audio).last_hidden_state
            emb = out.mean(dim=1)
        return emb.cpu().numpy()

    def extract_branch_b(self, audio):
        cfg       = self._pre_cfg
        frame_len = cfg["MEL_HOP_LENGTH"]
        n_frames  = TARGET_SAMPLES // frame_len
        with torch.no_grad():
            mfcc      = self.mfcc_transform(audio)
            mfcc_mean = mfcc.mean(dim=2)
            mfcc_std  = mfcc.std(dim=2)
            mel       = self.mel_transform(audio)
            mel_db    = self.amplitude_to_db(mel)
            freq_w    = self.freq_bins.view(1, -1, 1)
            mel_sum   = mel.sum(dim=1, keepdim=True).clamp(min=1e-8)
            centroid  = (mel * freq_w / mel_sum).sum(dim=1)
            sc_mean   = centroid.mean(dim=1, keepdim=True)
            sc_std    = centroid.std(dim=1, keepdim=True)
            diff      = (freq_w - centroid.unsqueeze(1)) ** 2
            bandwidth = (mel * diff / mel_sum).sum(dim=1).sqrt()
            sb_mean   = bandwidth.mean(dim=1, keepdim=True)
            sb_std    = bandwidth.std(dim=1, keepdim=True)
            cumsum    = mel.cumsum(dim=1)
            threshold = 0.85 * mel.sum(dim=1, keepdim=True)
            rolloff   = (cumsum < threshold).sum(dim=1).float() \
                        / cfg["MEL_N_MELS"] * cfg["MEL_FMAX"]
            sr_mean   = rolloff.mean(dim=1, keepdim=True)
            sr_std    = rolloff.std(dim=1, keepdim=True)
            flux      = (mel_db[:,:,1:] -
                         mel_db[:,:,:-1]).abs().mean(dim=1)
            flux_mean = flux.mean(dim=1, keepdim=True)
            flux_std  = flux.std(dim=1, keepdim=True)
            frames    = audio[:, :n_frames*frame_len]\
                        .reshape(1, n_frames, frame_len)
            rms       = frames.pow(2).mean(dim=2).sqrt()
            rms_mean  = rms.mean(dim=1, keepdim=True)
            rms_std   = rms.std(dim=1, keepdim=True)
            signs     = audio.sign()
            zcr_all   = (signs[:,1:] != signs[:,:-1]).float()
            zcr_nf    = zcr_all.shape[1] // frame_len
            zcr       = zcr_all[:, :zcr_nf*frame_len]\
                        .reshape(1, zcr_nf, frame_len).mean(dim=2)
            zcr_mean  = zcr.mean(dim=1, keepdim=True)
            zcr_std   = zcr.std(dim=1, keepdim=True)
            features  = torch.cat([
                mfcc_mean, mfcc_std, sc_mean, sc_std,
                sb_mean, sb_std, sr_mean, sr_std,
                flux_mean, flux_std, rms_mean, rms_std,
                zcr_mean, zcr_std], dim=1)
        raw = features.cpu().numpy().astype(np.float32)
        return self.scaler.transform(raw)

    def extract_acoustic_features(self, audio_tensor, audio_np):
        """
        Extract 8 acoustic biomarkers for frontend display.
        audio_tensor: (1, 64000) GPU tensor
        audio_np:     (64000,)   numpy float32 array
        """
        features = {}

        # ── 1. Spectral Centroid (Hz) ──────────────────────────────────────
        with torch.no_grad():
            mel      = self.mel_transform(audio_tensor)
            mel_sum  = mel.sum(dim=1, keepdim=True).clamp(min=1e-8)
            freq_w   = self.freq_bins.view(1, -1, 1)
            centroid = (mel * freq_w / mel_sum).sum(dim=1)
            sc_mean  = float(centroid.mean().cpu())
        features['spectral_centroid_hz'] = round(sc_mean, 1)

        # ── 2. RMS Energy (dB) ─────────────────────────────────────────────
        rms_linear = float(audio_tensor.pow(2).mean().sqrt().cpu())
        rms_db     = 20 * np.log10(rms_linear + 1e-9)
        features['rms_energy_db'] = round(rms_db, 1)

        # ── Find actual audio end (exclude zero padding) ───────────────────
        last_nonzero = np.nonzero(audio_np)[0]
        if len(last_nonzero) > 0:
            actual_samples = last_nonzero[-1] + 1
            actual_dur     = actual_samples / TARGET_SR
        else:
            actual_samples = len(audio_np)
            actual_dur     = actual_samples / TARGET_SR

        # ── 3. Pause Ratio (%) ─────────────────────────────────────────────
        frame_len     = self._pre_cfg['MEL_HOP_LENGTH']
        n_frames      = TARGET_SAMPLES // frame_len
        frames        = audio_tensor[
            :, :n_frames * frame_len
        ].reshape(n_frames, frame_len)
        frame_rms     = frames.pow(2).mean(dim=1).sqrt()
        actual_frames = min(n_frames, actual_samples // frame_len)

        if actual_frames > 0:
            silence_threshold = 0.02
            pause_frames      = (
                frame_rms[:actual_frames] < silence_threshold
            ).sum().item()
            pause_ratio = pause_frames / actual_frames * 100
        else:
            pause_ratio = 0.0
        features['pause_ratio_percent'] = round(pause_ratio, 1)

        # ── 4. Speech Rate (syllables/second) ──────────────────────────────
        frame_len_sr = int(TARGET_SR * 0.025)
        hop_len_sr   = int(TARGET_SR * 0.010)
        audio_voiced = audio_np[:actual_samples]   # only real audio
        n_frames_sr  = (len(audio_voiced) - frame_len_sr) // hop_len_sr

        if n_frames_sr > 0:
            rms_env = np.array([
                np.sqrt(np.mean(
                    audio_voiced[
                        i*hop_len_sr : i*hop_len_sr+frame_len_sr
                    ]**2
                ))
                for i in range(n_frames_sr)
            ])
            kernel     = np.ones(5) / 5
            rms_smooth = np.convolve(rms_env, kernel, mode='same')
            threshold  = rms_smooth.max() * 0.15   # lower threshold
            above      = rms_smooth > threshold
            syllables  = int((np.diff(above.astype(int)) > 0).sum())
            features['speech_rate_syll_per_sec'] = round(
                syllables / actual_dur if actual_dur > 0 else 0.0, 1)
        else:
            features['speech_rate_syll_per_sec'] = 0.0

        # ── Parselmouth section ────────────────────────────────────────────
        try:
            sound = parselmouth.Sound(
                audio_np.astype(np.float64),
                sampling_frequency=TARGET_SR
            )

            # Trimmed sound excludes zero padding for HNR accuracy
            sound_trimmed = sound.extract_part(
                from_time        = 0,
                to_time          = actual_dur,
                window_shape     = parselmouth.WindowShape.RECTANGULAR,
                relative_width   = 1,
                preserve_times   = False
            )

            # ── 5. Pitch F0 (Hz) ───────────────────────────────────────────
            pitch        = call(sound, "To Pitch", 0.0, 75, 500)
            pitch_values = [
                pitch.get_value_at_time(pitch.xs()[i])
                for i in range(len(pitch.xs()))
            ]
            voiced = [
                v for v in pitch_values
                if v is not None and not np.isnan(v) and v > 0
            ]
            features['pitch_f0_hz'] = round(
                float(np.mean(voiced)) if voiced else 0.0, 1)

            # ── 6. Jitter (%) ──────────────────────────────────────────────
            point_process = call(
                sound, "To PointProcess (periodic, cc)", 75, 500)
            n_points = call(point_process, "Get number of points")

            jitter = call(
                point_process,
                "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_percent'] = round(
                float(jitter) * 100
                if jitter and n_points >= 3 else 0.0, 2)

            # ── 7. Shimmer (%) ─────────────────────────────────────────────
            # parselmouth returns local shimmer as ratio (0.23 = 23%)
            # Only reliable with enough voiced periods
            shimmer = call(
                [sound, point_process],
                "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_percent'] = round(
                float(shimmer) * 100
                if shimmer and n_points >= 10
                and not np.isnan(float(shimmer))
                else 0.0, 2)

            # ── 8. HNR (dB) — computed on trimmed sound ────────────────────
            harmonicity = call(
                sound_trimmed,
                "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            features['hnr_db'] = round(
                float(hnr)
                if hnr and not np.isnan(float(hnr))
                else 0.0, 1)

        except Exception as e:
            features.setdefault('pitch_f0_hz',              0.0)
            features.setdefault('jitter_percent',            0.0)
            features.setdefault('shimmer_percent',           0.0)
            features.setdefault('hnr_db',                    0.0)
            features.setdefault('speech_rate_syll_per_sec',  0.0)

        return features
    
    def predict(self, pcm_input):
        t_start = time.time()
        audio   = self.preprocess(pcm_input)
        audio_np = audio.squeeze(0).cpu().numpy()  
        # Branch A → Model A
        t_a    = time.time()
        emb_a  = self.extract_branch_a(audio)
        p_a    = self.model_a.predict_proba(emb_a)[0, 1]
        time_a = (time.time() - t_a) * 1000

        # Branch B → Model B
        t_b    = time.time()
        feat_b = self.extract_branch_b(audio)
        p_b    = self.model_b.predict_proba(feat_b)[0, 1]
        time_b = (time.time() - t_b) * 1000

        # Model C fusion
        t_c     = time.time()
        mean_s  = (p_a + p_b) / 2
        fusion  = np.array([[p_a, p_b, mean_s, abs(p_a-p_b), p_a*p_b]])
        p_fake  = self.model_c.predict_proba(fusion)[0, 1]
        time_c  = (time.time() - t_c) * 1000

        # Extract acoustic biomarkers for frontend display
        acoustic  = self.extract_acoustic_features(audio, audio_np)
        total_ms  = (time.time() - t_start) * 1000

        return {
            'p_fake'             : round(float(p_fake),  4),
            'p_a'                : round(float(p_a),     4),
            'p_b'                : round(float(p_b),     4),
            'processing_time_ms' : round(total_ms,       1),
            'branch_timing'      : {
                'model_a_ms'     : round(time_a, 1),
                'model_b_ms'     : round(time_b, 1),
                'model_c_ms'     : round(time_c, 1)
            },
            'acoustic_features'  : {
                'pitch_f0_hz'           : acoustic.get('pitch_f0_hz',             0.0),
                'jitter_percent'        : acoustic.get('jitter_percent',           0.0),
                'shimmer_percent'       : acoustic.get('shimmer_percent',          0.0),
                'hnr_db'                : acoustic.get('hnr_db',                   0.0),
                'spectral_centroid_hz'  : acoustic.get('spectral_centroid_hz',     0.0),
                'rms_energy_db'         : acoustic.get('rms_energy_db',            0.0),
                'speech_rate_syll_per_sec': acoustic.get('speech_rate_syll_per_sec', 0.0),
                'pause_ratio_percent'   : acoustic.get('pause_ratio_percent',      0.0)
            }
        }


class TemporalSmoother:
    """
    Exponential Moving Average (EMA) smoother for per-window p_fake scores.

    Computes:  S(t) = alpha * P(t) + (1 - alpha) * S(t-1)

    The verdict thresholds (high / low) must be supplied by the caller from
    the canonical settings object.  This class intentionally has no built-in
    defaults so that only one source of truth can exist in the system.

    alpha: smoothing factor in (0, 1].
           - Higher alpha = more reactive to the latest window.
           - Lower  alpha = smoother, slower to change.
           Default 0.3 is roughly equivalent to a 5-window SMA in responsiveness.
    """

    def __init__(self, alpha: float = 0.3, high: float = None, low: float = None):
        if high is None or low is None:
            raise ValueError(
                "TemporalSmoother requires explicit 'high' and 'low' thresholds. "
                "Pass them from the canonical settings object."
            )
        self.alpha         = alpha
        self.high          = high
        self.low           = low
        self._ema: float | None = None   # None until first window seen
        self._windows_seen = 0

    def update(self, p_fake: float) -> dict:
        p_fake = float(p_fake)

        if self._ema is None:
            # Bootstrap: first observation initialises the EMA directly
            self._ema = p_fake
        else:
            self._ema = self.alpha * p_fake + (1 - self.alpha) * self._ema

        self._windows_seen += 1
        smoothed = self._ema

        if smoothed >= self.high:
            verdict, uncertain = "AI", False
            confidence = float(smoothed)
        elif smoothed <= self.low:
            verdict, uncertain = "HUMAN", False
            confidence = float(1 - smoothed)
        else:
            verdict, uncertain = "UNCERTAIN", True
            confidence = float(1 - abs(smoothed - 0.5) * 2)

        return {
            "verdict"              : verdict,
            "confidence"           : round(confidence, 4),
            "smoothed_p_fake"      : round(smoothed, 4),
            "uncertain"            : uncertain,
            "windows_seen"         : self._windows_seen,
        }

    def reset(self):
        self._ema          = None
        self._windows_seen = 0

