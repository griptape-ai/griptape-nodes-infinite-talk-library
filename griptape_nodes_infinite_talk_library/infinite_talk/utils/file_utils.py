"""File utilities for InfiniteTalk nodes."""

import logging
from pathlib import Path
from typing import Any

from griptape.artifacts import AudioUrlArtifact, ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination

logger = logging.getLogger("griptape_nodes_infinite_talk_library")


def save_video_to_static(video_path: Path, filename: str) -> VideoUrlArtifact:
    """Save video file to static storage and return VideoUrlArtifact.

    Args:
        video_path: Path to the video file on disk
        filename: Desired filename for the saved video

    Returns:
        VideoUrlArtifact pointing to the saved video
    """
    video_bytes = video_path.read_bytes()
    dest = ProjectFileDestination(filename=filename, situation="save_node_output")
    saved = dest.write_bytes(video_bytes)
    return VideoUrlArtifact(value=saved.location, name=filename)


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

    output_path.write_bytes(File(url).read_bytes())
    return output_path
