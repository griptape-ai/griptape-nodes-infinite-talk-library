"""InfiniteTalk Video to Video node."""

import tempfile
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo

from infinite_talk.nodes.base_infinite_talk_node import BaseInfiniteTalkNode
from infinite_talk.utils.file_utils import download_artifact_to_temp


class InfiniteTalkVideo2Video(BaseInfiniteTalkNode):
    """Generate talking video from video and audio using InfiniteTalk (dubbing).

    This node takes a reference video and new driving audio to generate a dubbed
    video where the person's lips are synchronized to the new audio.

    Inputs:
        - video: Reference video to dub
        - audio: New driving audio for lip sync
        - prompt: Text description of the scene
        - mode: Generation mode ('clip' or 'streaming')
        - base_model: Wan-AI model for video generation (determines resolution)
        - audio_encoder: Wav2Vec model for audio encoding
        - infinitetalk_weights: InfiniteTalk model weights

    Outputs:
        - video: Generated dubbed video
        - was_successful: Whether generation succeeded
        - result_details: Details about the result or error
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "Video Generation"
        self.description = "Generate talking video from video and audio using InfiniteTalk (dubbing)"

        # Add video input parameter (node-specific, after audio from base)
        self.add_parameter(
            ParameterVideo(
                name="input_video",
                tooltip="Reference video to dub with new audio",
                clickable_file_browser=True,
                allow_output=False,
            )
        )

        # Add video output and status parameters (must be last)
        self._add_final_parameters()

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self) -> None:
        """Process the video to video dubbing."""
        # Get input values
        input_video = self.get_parameter_value("input_video")
        audio = self.get_parameter_value("audio")
        prompt = self.get_parameter_value("prompt") or ""
        mode = self.get_parameter_value("mode") or "clip"

        if input_video is None:
            self._set_status_results(was_successful=False, result_details="FAILURE: No video provided")
            self.parameter_output_values["video"] = None
            return

        if audio is None:
            self._set_status_results(was_successful=False, result_details="FAILURE: No audio provided")
            self.parameter_output_values["video"] = None
            return

        # Download artifacts to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download video
            video_path = download_artifact_to_temp(input_video, temp_path, "input_video.mp4")

            # Download audio
            audio_path = download_artifact_to_temp(audio, temp_path, "input_audio.wav")

            # Run inference
            self._prepare_and_run_inference(
                prompt=prompt,
                cond_media_path=video_path,
                audio_path=audio_path,
                mode=mode,
            )
