import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from torchaudio.transforms import MelSpectrogram
from scipy.signal import butter, sosfilt

class MasteringEffects:
    """
    Professional Audio Mastering Node for ComfyUI
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "lufs_target": ("FLOAT", {"default": -14.0, "min": -30.0, "max": -5.0, "step": 0.5}),
                "bass_boost": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.5}),
                "treble_boost": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.5}),
                "compression": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "vocal_clarity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "genre": (["none", "trap", "gospel", "pop", "rock", "edm", "jazz"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mastered_audio",)
    FUNCTION = "master"
    CATEGORY = "audio/mastering"

    def master(self, audio, lufs_target, bass_boost, treble_boost, compression, 
               stereo_width, vocal_clarity, genre):
        # Ensure audio is 2D tensor [channels, samples]
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        
        # Convert to numpy for processing
        sample_rate = 48000  # ComfyUI standard
        audio_np = audio.numpy()
        
        # Apply genre presets
        if genre != "none":
            audio_np = self.apply_genre_preset(audio_np, genre, compression, 
                                              bass_boost, treble_boost, vocal_clarity)
        
        # Apply EQ
        audio_np = self.apply_eq(audio_np, sample_rate, bass_boost, treble_boost)
        
        # Apply compression
        if compression > 0:
            audio_np = self.apply_compression(audio_np, sample_rate, compression)
        
        # Apply stereo width
        if audio_np.shape[0] == 2 and stereo_width != 1.0:
            audio_np = self.apply_stereo_width(audio_np, stereo_width)
        
        # Vocal clarity enhancement
        if vocal_clarity > 0:
            audio_np = self.enhance_vocals(audio_np, sample_rate, vocal_clarity)
        
        # Loudness normalization
        audio_np = self.normalize_loudness(audio_np, sample_rate, lufs_target)
        
        # Final limiting
        audio_np = self.apply_limiter(audio_np, sample_rate)
        
        # Convert back to tensor
        mastered_audio = torch.from_numpy(audio_np).unsqueeze(0)
        return (mastered_audio,)

    def apply_genre_preset(self, audio, genre, compression, bass_boost, 
                          treble_boost, vocal_clarity):
        """Apply genre-specific processing presets"""
        if genre == "trap":
            bass_boost = max(bass_boost, 3.0)
            treble_boost = max(treble_boost, 2.0)
            compression = min(compression + 0.2, 0.9)
            # Apply high-pass filter to clean lows
            audio = self.butterworth_filter(audio, 50, 'highpass')
            
        elif genre == "gospel":
            vocal_clarity = min(vocal_clarity + 0.4, 1.0)
            compression = max(compression - 0.1, 0.3)
            # Gentle low-shelf boost for warmth
            audio = self.apply_lowshelf(audio, 150, 0.7, 2.0)
            
        elif genre == "pop":
            vocal_clarity = min(vocal_clarity + 0.3, 1.0)
            treble_boost = max(treble_boost, 1.5)
            compression = min(compression + 0.1, 0.8)
            
        elif genre == "edm":
            bass_boost = max(bass_boost, 4.0)
            compression = min(compression + 0.3, 1.0)
            
        return audio

    def apply_eq(self, audio, sample_rate, bass_gain, treble_gain):
        """Apply basic EQ adjustments"""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Apply bass boost (low shelf)
        if bass_gain != 0:
            audio_tensor = F.lowshelf_biquad(
                audio_tensor, sample_rate, 150, 0.7, bass_gain
            )
        
        # Apply treble boost (high shelf)
        if treble_gain != 0:
            audio_tensor = F.highshelf_biquad(
                audio_tensor, sample_rate, 5000, 0.7, treble_gain
            )
            
        return audio_tensor.numpy()

    def apply_lowshelf(self, audio, center_freq, Q, gain):
        """Apply low shelf filter"""
        audio_tensor = torch.from_numpy(audio).float()
        return F.lowshelf_biquad(
            audio_tensor, 48000, center_freq, Q, gain
        ).numpy()

    def butterworth_filter(self, audio, cutoff, filter_type, order=4):
        """Apply butterworth filter using scipy"""
        sos = butter(order, cutoff, btype=filter_type, fs=48000, output='sos')
        return sosfilt(sos, audio)

    def apply_compression(self, audio, sample_rate, amount):
        """Simple digital compression"""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2, axis=-1, keepdims=True))
        threshold = 0.1 + (0.4 * (1 - amount))  # Adaptive threshold
        
        # Compression curve
        gain_reduction = np.where(rms > threshold, 
                                 threshold / (rms + 1e-7), 
                                 np.ones_like(rms))
        
        # Apply compression with smoothing
        compressed = audio * gain_reduction
        return 0.7 * compressed + 0.3 * audio  # Mix dry/wet

    def apply_stereo_width(self, audio, width):
        """Mid/side stereo width adjustment"""
        mid = (audio[0] + audio[1]) * 0.5
        side = (audio[1] - audio[0]) * 0.5
        side *= width
        return np.stack([mid - side, mid + side])

    def enhance_vocals(self, audio, sample_rate, amount):
        """Vocal presence enhancement"""
        if audio.shape[0] != 2:  # Only works on stereo
            return audio
            
        # Extract mid channel (vocals typically centered)
        mid = (audio[0] + audio[1]) * 0.5
        
        # EQ boost in vocal range (1-4kHz)
        mid_tensor = torch.from_numpy(mid).float()
        mid_tensor = F.equalizer_biquad(
            mid_tensor, sample_rate, 2000, 1.0, 4.0 * amount
        )
        
        # Blend enhanced mids
        enhanced_mid = mid_tensor.numpy()
        mid = (1 - amount) * mid + amount * enhanced_mid
        
        # Reconstruct stereo
        side = (audio[1] - audio[0]) * 0.5
        return np.stack([mid - side, mid + side])

    def normalize_loudness(self, audio, sample_rate, target_lufs):
        """LUFS-based loudness normalization"""
        # Calculate RMS as simple loudness approximation
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(target_lufs / 20)
        gain = target_rms / (rms + 1e-7)
        return audio * gain

    def apply_limiter(self, audio, sample_rate, threshold=0.95):
        """Peak limiter to prevent clipping"""
        peak = np.max(np.abs(audio))
        if peak > threshold:
            gain = threshold / peak
            return audio * gain
        return audio

# For ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "MasteringEffects": MasteringEffects
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MasteringEffects": "🔊 Mastering Effects"
}