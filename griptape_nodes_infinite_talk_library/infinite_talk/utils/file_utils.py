"""File utilities for InfiniteTalk nodes."""

import logging
from pathlib import Path
from typing import Any

import httpx
from griptape.artifacts import AudioUrlArtifact, ImageUrlArtifact, VideoUrlArtifact

from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("griptape_nodes_infinite_talk_library")

DEFAULT_TIMEOUT = 60


def save_video_to_static(video_path: Path, filename: str) -> VideoUrlArtifact:
    """Save video file to static storage and return VideoUrlArtifact.

    Args:
        video_path: Path to the video file on disk
        filename: Desired filename for the saved video

    Returns:
        VideoUrlArtifact pointing to the saved video
    """
    video_bytes = video_path.read_bytes()
    static_files_manager = GriptapeNodes.StaticFilesManager()
    saved_url = static_files_manager.save_static_file(video_bytes, filename)
    return VideoUrlArtifact(value=saved_url, name=filename)


def _is_local_path(url: str) -> bool:
    """Check if URL is a local file path."""
    return not url.startswith(("http://", "https://"))


def _download_from_url(url: str) -> bytes:
    """Download bytes from a URL."""
    response = httpx.get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.content


def download_artifact_to_temp(
    artifact: ImageUrlArtifact | VideoUrlArtifact | AudioUrlArtifact | Any,
    temp_dir: Path,
    filename: str,
) -> Path:
    """Download artifact to temp directory for InfiniteTalk input.

    Args:
        artifact: URL artifact (ImageUrlArtifact, VideoUrlArtifact, or AudioUrlArtifact)
        temp_dir: Temporary directory to save the file
        filename: Desired filename

    Returns:
        Path to the downloaded file
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / filename

    # Get URL from artifact
    if hasattr(artifact, "value"):
        url = artifact.value
    elif isinstance(artifact, str):
        url = artifact
    else:
        msg = f"Unsupported artifact type: {type(artifact)}"
        raise TypeError(msg)

    # Handle local paths
    if _is_local_path(url):
        local_path = Path(url)
        if local_path.exists():
            # Copy file to temp directory
            output_path.write_bytes(local_path.read_bytes())
            return output_path
        msg = f"Local file not found: {url}"
        raise FileNotFoundError(msg)

    # Download from URL
    logger.info("Downloading %s to %s", url, output_path)
    content = _download_from_url(url)
    output_path.write_bytes(content)
    return output_path
