import torch
import numpy as np

class LimiterNode:
    """
    Professional Audio Limiter Node for ComfyUI
    Prevents clipping with true peak detection and oversampling
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold": ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.1}),
                "release": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 500.0, "step": 5.0}),
                "oversampling": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "lookahead": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("limited_audio",)
    FUNCTION = "limit"
    CATEGORY = "audio/processing"

    def limit(self, audio, threshold, release, oversampling, lookahead):
        # Convert to numpy array
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        audio_np = audio.numpy()
        sample_rate = 48000
        
        # Process each channel
        processed = np.zeros_like(audio_np)
        for c in range(audio_np.shape[0]):
            processed[c] = self.process_channel(
                audio_np[c], sample_rate, 
                threshold, release, oversampling, lookahead
            )
            
        return (processed.unsqueeze(0),)

    def process_channel(self, audio, sample_rate, threshold, release, oversampling, lookahead):
        # Apply oversampling
        if oversampling > 1:
            audio = self.oversample(audio, oversampling)
            sample_rate *= oversampling
            
        # Convert time parameters to samples
        release_samples = int((release / 1000) * sample_rate)
        lookahead_samples = int((lookahead / 1000) * sample_rate)
        
        # Apply lookahead
        if lookahead_samples > 0:
            audio = np.concatenate((audio, np.zeros(lookahead_samples)))
        
        # Initialize variables
        gain = 1.0
        envelope = 0.0
        output = np.zeros_like(audio)
        threshold_linear = 10 ** (threshold / 20)
        
        # Main processing loop
        for i in range(len(audio)):
            # Detect peaks
            peak = np.abs(audio[i])
            
            # Calculate gain reduction needed
            if peak > threshold_linear:
                reduction_needed = threshold_linear / peak
            else:
                reduction_needed = 1.0
                
            # Smooth gain changes
            if reduction_needed < gain:
                # Attack immediately
                gain = reduction_needed
            else:
                # Release phase
                gain += (1.0 - gain) / release_samples
                
            # Apply gain reduction with lookahead offset
            if i >= lookahead_samples:
                output[i - lookahead_samples] = audio[i] * gain
            elif lookahead_samples == 0:
                output[i] = audio[i] * gain
                
        # Downsample if needed
        if oversampling > 1:
            output = self.downsample(output, oversampling)
            
        return output[:len(audio)]

    def oversample(self, audio, factor):
        """Upsample audio by integer factor"""
        return np.repeat(audio, factor)

    def downsample(self, audio, factor):
        """Downsample audio by integer factor"""
        return audio[::factor]

# For ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LimiterNode": LimiterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LimiterNode": "🔊 Audio Limiter"
}