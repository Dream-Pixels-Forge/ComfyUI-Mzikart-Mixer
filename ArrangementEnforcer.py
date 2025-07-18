import torch
import numpy as np
import librosa

class ArrangementEnforcer:
    """
    Song Structure Enforcer Node for ComfyUI
    Rearranges audio to match specified song structure
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "structure": ("STRING", {
                    "multiline": True, 
                    "default": "Intro(8 bars)|Verse(16 bars)|Chorus(16 bars)|Verse(16 bars)|Chorus(16 bars)|Bridge(8 bars)|Outro(8 bars)"
                }),
                "bpm": ("FLOAT", {"default": 120.0, "min": 60.0, "max": 200.0, "step": 1.0}),
                "time_signature": (["4/4", "3/4", "6/8"], {"default": "4/4"}),
                "quantization": (["none", "bar", "beat"], {"default": "bar"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("arranged_audio",)
    FUNCTION = "arrange"
    CATEGORY = "audio/arrangement"

    def arrange(self, audio, structure, bpm, time_signature, quantization):
        # Convert to numpy array
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        audio_np = audio.numpy()
        sample_rate = 48000
        
        # Parse time signature
        beats_per_bar = int(time_signature.split('/')[0])
        
        # Calculate bar duration in samples
        bar_duration = (60 / bpm) * beats_per_bar  # in seconds
        bar_samples = int(bar_duration * sample_rate)
        
        # Split audio into segments
        segments = self.split_audio(audio_np, bar_samples, beats_per_bar, quantization)
        
        # Parse structure
        section_names, section_lengths = self.parse_structure(structure)
        
        # Arrange segments according to structure
        arranged_audio = self.build_arrangement(segments, section_names, section_lengths, bar_samples)
        
        return (arranged_audio.unsqueeze(0),)

    def split_audio(self, audio, bar_samples, beats_per_bar, quantization):
        """Split audio into bars/beats"""
        segments = []
        
        if quantization == "none":
            # Split into individual bars
            for i in range(0, audio.shape[1], bar_samples):
                segments.append(audio[:, i:i+bar_samples])
        elif quantization == "bar":
            # Split into bars
            num_bars = audio.shape[1] // bar_samples
            for i in range(num_bars):
                start = i * bar_samples
                end = start + bar_samples
                segments.append(audio[:, start:end])
        elif quantization == "beat":
            # Split into beats
            beat_samples = bar_samples // beats_per_bar
            num_beats = audio.shape[1] // beat_samples
            for i in range(num_beats):
                start = i * beat_samples
                end = start + beat_samples
                segments.append(audio[:, start:end])
                
        return segments

    def parse_structure(self, structure):
        """Parse structure string into sections"""
        section_names = []
        section_lengths = []
        
        # Split structure string
        sections = structure.split('|')
        for section in sections:
            if '(' in section and ')' in section:
                name = section.split('(')[0].strip()
                length_str = section.split('(')[1].split(')')[0].strip()
                
                # Parse length (support bars or seconds)
                if 'bar' in length_str:
                    length = int(length_str.split()[0])
                elif 'sec' in length_str:
                    seconds = float(length_str.split()[0])
                    length = seconds / (60 / bpm)  # Convert to bars
                else:
                    length = 4  # Default to 4 bars
                    
                section_names.append(name)
                section_lengths.append(length)
                
        return section_names, section_lengths

    def build_arrangement(self, segments, section_names, section_lengths, bar_samples):
        """Assemble segments according to structure"""
        arranged = []
        segment_idx = 0
        
        for name, length in zip(section_names, section_lengths):
            # Get segments for this section
            section_segments = []
            for _ in range(int(length)):
                if segment_idx < len(segments):
                    section_segments.append(segments[segment_idx])
                    segment_idx += 1
                else:
                    # Pad with silence if needed
                    silence = np.zeros((segments[0].shape[0], bar_samples))
                    section_segments.append(silence)
            
            # Apply section-specific processing
            if name.lower() == "chorus":
                section_segments = self.enhance_chorus(section_segments)
            elif name.lower() == "bridge":
                section_segments = self.process_bridge(section_segments)
                
            # Concatenate segments
            arranged.append(np.concatenate(section_segments, axis=1))
        
        # Combine all sections
        return np.concatenate(arranged, axis=1)

    def enhance_chorus(self, segments):
        """Apply chorus-specific enhancements"""
        enhanced = []
        for seg in segments:
            # Add subtle reverb
            seg = self.apply_reverb(seg, 48000, 1.5, 0.4)
            
            # Increase volume
            seg = seg * 1.2
            
            # Add stereo width
            if seg.shape[0] == 2:
                mid = (seg[0] + seg[1]) * 0.5
                side = (seg[1] - seg[0]) * 0.5
                side *= 1.5  # Wider stereo
                seg = np.stack([mid - side, mid + side])
                
            enhanced.append(seg)
        return enhanced

    def apply_reverb(self, audio, sample_rate, decay, wet_dry):
        """Simple reverb effect"""
        # This is placeholder logic - real implementation uses convolution
        impulse = np.exp(-np.arange(0, decay * sample_rate) / (decay * sample_rate))
        reverb = np.convolve(audio.flatten(), impulse, mode='same')
        reverb = reverb.reshape(audio.shape)
        return (1 - wet_dry) * audio + wet_dry * reverb

# For ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ArrangementEnforcer": ArrangementEnforcer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArrangementEnforcer": "🎶 Arrangement Enforcer"
}