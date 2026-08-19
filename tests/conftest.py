"""Pytest configuration and fixtures for WavTTS tests."""

import torchaudio
import soundfile as sf


class AudioInfo:
    """Simple audio info container matching torchaudio.info() expected interface."""

    def __init__(self, num_frames, sample_rate, num_channels=1):
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.num_channels = num_channels


def _torchaudio_info(path: str) -> AudioInfo:
    """Wrapper around soundfile.info to provide AudioInfo compatible interface."""
    sf_info = sf.info(path)
    return AudioInfo(
        num_frames=sf_info.frames,
        sample_rate=sf_info.samplerate,
        num_channels=sf_info.channels,
    )


# Add torchaudio.info if not present (compatibility for older versions)
if not hasattr(torchaudio, "info"):
    torchaudio.info = _torchaudio_info
