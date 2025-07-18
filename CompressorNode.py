import torch
import numpy as np
from scipy.signal import lfilter

class CompressorNode:
    """
    Professional Audio Compressor Node for ComfyUI
    Applies dynamic range compression with lookahead capability
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0, "step": 1.0}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "attack": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "release": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                "makeup_gain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.5}),
                "lookahead": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "knee": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("compressed_audio",)
    FUNCTION = "compress"
    CATEGORY = "audio/processing"

    def compress(self, audio, threshold, ratio, attack, release, makeup_gain, lookahead, knee):
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
                threshold, ratio, attack, release, 
                makeup_gain, lookahead, knee
            )
            
        return (processed.unsqueeze(0),)

    def process_channel(self, audio, sample_rate, threshold, ratio, 
                       attack, release, makeup_gain, lookahead, knee):
        # Convert time parameters to samples
        attack_samples = int((attack / 1000) * sample_rate)
        release_samples = int((release / 1000) * sample_rate)
        lookahead_samples = int((lookahead / 1000) * sample_rate)
        
        # Apply lookahead
        if lookahead_samples > 0:
            audio = np.concatenate((audio, np.zeros(lookahead_samples)))
        
        # Initialize variables
        gain_reduction = 0.0
        envelope = 0.0
        output = np.zeros_like(audio)
        
        # Convert to dB
        linear_to_db = lambda x: 20 * np.log10(np.maximum(np.abs(x), 1e-7))
        
        # Main processing loop
        for i in range(len(audio)):
            # Calculate current level in dB
            current_level = linear_to_db(np.abs(audio[i]))
            
            # Calculate overshoot above threshold
            overshoot = current_level - threshold
            if overshoot < 0:
                overshoot = 0
                
            # Soft knee
            if knee > 0 and overshoot > 0 and overshoot < knee:
                overshoot = overshoot**2 / (2 * knee)
            
            # Calculate desired gain reduction
            desired_reduction = overshoot * (1 - 1/ratio)
            
            # Attack/release envelope
            if desired_reduction > gain_reduction:
                # Attack phase
                gain_reduction += (desired_reduction - gain_reduction) / attack_samples
            else:
                # Release phase
                gain_reduction -= (gain_reduction - desired_reduction) / release_samples
                
            # Convert gain reduction to linear scale
            gain_linear = 10 ** (makeup_gain / 20) / (10 ** (gain_reduction / 20))
            
            # Apply gain reduction with lookahead offset
            if i >= lookahead_samples:
                output[i - lookahead_samples] = audio[i] * gain_linear
            elif lookahead_samples == 0:
                output[i] = audio[i] * gain_linear
                
        return output[:len(audio)]

# For ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "CompressorNode": CompressorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompressorNode": "🔊 Audio Compressor"
}