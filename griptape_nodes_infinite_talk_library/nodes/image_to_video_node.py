"""InfiniteTalk Image to Video node."""

import tempfile
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage

from griptape_nodes_infinite_talk_library.nodes.base_infinite_talk_node import BaseInfiniteTalkNode
from griptape_nodes_infinite_talk_library.utils.file_utils import download_artifact_to_temp


class InfiniteTalkImage2Video(BaseInfiniteTalkNode):
    """Generate talking video from image and audio using InfiniteTalk.

    This node takes a reference image and driving audio to generate a talking
    video where the person in the image appears to speak the audio.

    Inputs:
        - image: Reference image of the person
        - audio: Driving audio for lip sync and expressions
        - prompt: Text description of the scene
        - mode: Generation mode ('clip' or 'streaming')
        - base_model: Wan-AI model for video generation (determines resolution)
        - audio_encoder: Wav2Vec model for audio encoding
        - infinitetalk_weights: InfiniteTalk model weights

    Outputs:
        - video: Generated talking video
        - was_successful: Whether generation succeeded
        - result_details: Details about the result or error
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "Video Generation"
        self.description = "Generate talking video from image and audio using InfiniteTalk"

        # Add image input parameter (before common parameters)
        self._add_image_parameter()

        # Add audio input parameter
        self._add_audio_parameter()

    def _add_image_parameter(self) -> None:
        """Add image input parameter."""
        self.add_parameter(
            ParameterImage(
                name="image",
                tooltip="Reference image of the person to animate",
                clickable_file_browser=True,
                allow_output=False,
            )
        )

    def _add_audio_parameter(self) -> None:
        """Add audio input parameter."""
        self.add_parameter(
            ParameterAudio(
                name="audio",
                tooltip="Driving audio for lip sync and expressions",
                clickable_file_browser=True,
                allow_output=False,
            )
        )

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self) -> None:
        """Process the image to video generation."""
        # Get input values
        image = self.get_parameter_value("image")
        audio = self.get_parameter_value("audio")
        prompt = self.get_parameter_value("prompt") or ""
        mode = self.get_parameter_value("mode") or "clip"

        if image is None:
            self._set_status_results(was_successful=False, result_details="FAILURE: No image provided")
            self.parameter_output_values["video"] = None
            return

        if audio is None:
            self._set_status_results(was_successful=False, result_details="FAILURE: No audio provided")
            self.parameter_output_values["video"] = None
            return

        # Download artifacts to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download image
            image_path = download_artifact_to_temp(image, temp_path, "input_image.png")

            # Download audio
            audio_path = download_artifact_to_temp(audio, temp_path, "input_audio.wav")

            # Run inference
            self._prepare_and_run_inference(
                prompt=prompt,
                cond_media_path=image_path,
                audio_path=audio_path,
                mode=mode,
            )
