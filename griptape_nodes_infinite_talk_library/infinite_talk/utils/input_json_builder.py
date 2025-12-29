"""Build input JSON configuration for InfiniteTalk inference."""

from pathlib import Path
from typing import Any


def build_input_json(
    prompt: str,
    cond_media_path: str | Path,
    audio_paths: dict[str, str | Path],
) -> dict[str, Any]:
    """Build input_json for InfiniteTalk inference.

    Args:
        prompt: Text description of the video content
        cond_media_path: Path to conditioning image or video
        audio_paths: Dictionary mapping person IDs to audio file paths
            Example: {"person1": "/path/to/audio.wav"}

    Returns:
        Dictionary with InfiniteTalk input configuration
    """
    return {
        "prompt": prompt,
        "cond_video": str(cond_media_path),
        "cond_audio": {k: str(v) for k, v in audio_paths.items()},
    }
