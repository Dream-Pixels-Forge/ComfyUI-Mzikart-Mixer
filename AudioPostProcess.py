import torch
import numpy as np
from scipy.signal import butter, sosfiltfilt

class AudioPostProcessor:
    """
    Advanced Audio Post-Processing Node for ComfyUI
    Applies professional-grade effects chain to audio
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "effect": ([
                    "gospel_choir", 
                    "trap_808", 
                    "vocal_doubler",
                    "vinyl_emulation",
                    "radio_effect",
                    "hall_reverb",
                    "tape_saturation"
                ], {"default": "gospel_choir"}),
                "intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "wet_dry": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("processed_audio",)
    FUNCTION = "process"
    CATEGORY = "audio/effects"

    def process(self, audio, effect, intensity, wet_dry):
        # Convert to numpy array
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        audio_np = audio.numpy()
        sample_rate = 48000
        
        # Process each channel
        processed = np.zeros_like(audio_np)
        for c in range(audio_np.shape[0]):
            dry_signal = audio_np[c]
            wet_signal = self.apply_effect(dry_signal, sample_rate, effect, intensity)
            
            # Apply wet/dry mix
            processed[c] = (1 - wet_dry) * dry_signal + wet_dry * wet_signal
            
        return (processed.unsqueeze(0),)

    def apply_effect(self, audio, sample_rate, effect, intensity):
        """Apply selected effect to audio"""
        if effect == "gospel_choir":
            return self.gospel_choir(audio, sample_rate, intensity)
        elif effect == "trap_808":
            return self.trap_808(audio, sample_rate, intensity)
        elif effect == "vocal_doubler":
            return self.vocal_doubler(audio, sample_rate, intensity)
        elif effect == "vinyl_emulation":
            return self.vinyl_emulation(audio, sample_rate, intensity)
        elif effect == "radio_effect":
            return self.radio_effect(audio, sample_rate, intensity)
        elif effect == "hall_reverb":
            return self.hall_reverb(audio, sample_rate, intensity)
        elif effect == "tape_saturation":
            return self.tape_saturation(audio, sample_rate, intensity)
        return audio

    def gospel_choir(self, audio, sample_rate, intensity):
        """Add gospel choir harmonies"""
        # Create harmonies
        harmony1 = self.pitch_shift(audio, sample_rate, 3)  # Minor third
        harmony2 = self.pitch_shift(audio, sample_rate, 7)  # Fifth
        harmony3 = self.pitch_shift(audio, sample_rate, 10)  # Minor seventh
        
        # Blend harmonies
        choir = 0.5 * harmony1 + 0.3 * harmony2 + 0.2 * harmony3
        
        # Apply church reverb
        choir = self.apply_reverb(choir, sample_rate, 2.5, 0.6)
        
        return (1 - intensity) * audio + intensity * choir

    def trap_808(self, audio, sample_rate, intensity):
        """Add trap-style 808 bass"""
        # Detect fundamental frequency
        fundamental = self.detect_fundamental(audio, sample_rate)
        
        # Generate 808 sine wave
        t = np.arange(len(audio)) / sample_rate
        sine_wave = np.sin(2 * np.pi * fundamental * t)
        
        # Add pitch slide
        slide = np.exp(-t * 2) * fundamental * 0.5
        sine_wave = np.sin(2 * np.pi * (fundamental + slide) * t)
        
        # Apply distortion
        sine_wave = np.tanh(sine_wave * 3)
        
        # Mix with original
        return audio + intensity * sine_wave * 0.3

    def vocal_doubler(self, audio, sample_rate, intensity):
        """Create vocal doubling effect"""
        # Short delay
        delay_samples = int(0.03 * sample_rate)  # 30ms delay
        delayed = np.roll(audio, delay_samples)
        delayed[:delay_samples] = 0
        
        # Pitch modulation
        modulated = self.pitch_shift(delayed, sample_rate, 0.5 * intensity)
        
        # Stereo spread
        left = 0.7 * audio + 0.3 * modulated
        right = 0.3 * audio + 0.7 * modulated
        
        return np.stack([left, right]) if audio.ndim > 1 else left

    def vinyl_emulation(self, audio, sample_rate, intensity):
        """Add vinyl record characteristics"""
        # Apply low-pass filter
        sos = butter(2, 10000, 'lowpass', fs=sample_rate, output='sos')
        audio = sosfiltfilt(sos, audio)
        
        # Add crackle
        if intensity > 0.3:
            crackle = np.random.uniform(-0.02, 0.02, len(audio))
            audio += intensity * crackle * 0.5
        
        # Add wow & flutter
        phase = np.cumsum(1 + 0.005 * np.sin(2 * np.pi * 0.5 * np.arange(len(audio)) / sample_rate))
        audio = np.interp(np.arange(len(audio)), phase, audio)
        
        return audio

    def pitch_shift(self, audio, sample_rate, semitones):
        """Simple pitch shifting"""
        # Using Fourier method for simplicity (real implementations use phase vocoding)
        n = len(audio)
        freqs = np.fft.rfftfreq(n, d=1/sample_rate)
        fft = np.fft.rfft(audio)
        shifted_freqs = freqs * (2 ** (semitones / 12))
        
        # Interpolate complex spectrum
        new_fft = np.interp(freqs, shifted_freqs, fft, left=0, right=0)
        return np.fft.irfft(new_fft, n=n).real

    def detect_fundamental(self, audio, sample_rate):
        """Detect fundamental frequency using autocorrelation"""
        # Normalize audio
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        
        # Autocorrelation
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first peak after zero
        peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
        if len(peaks) > 0:
            return sample_rate / peaks[0]
        return 100  # Default to 100Hz

# For ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "AudioPostProcessor": AudioPostProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioPostProcessor": "🎚️ Audio Post-Processor"
}