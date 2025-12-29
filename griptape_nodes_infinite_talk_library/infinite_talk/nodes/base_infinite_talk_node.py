"""Base class for InfiniteTalk nodes with common subprocess execution logic."""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from griptape.artifacts import VideoUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from infinite_talk.utils.file_utils import save_video_to_static
from infinite_talk.utils.input_json_builder import build_input_json

logger = logging.getLogger("griptape_nodes_infinite_talk_library")


class BaseInfiniteTalkNode(SuccessFailureNode):
    """Base class for InfiniteTalk nodes with common subprocess execution."""

    # Model constants
    MODEL_480P: ClassVar[str] = "Wan-AI/Wan2.1-I2V-14B-480P"
    MODEL_720P: ClassVar[str] = "Wan-AI/Wan2.1-I2V-14B-720P"
    CKPT_REPOS: ClassVar[list[str]] = [MODEL_480P, MODEL_720P]
    WAV2VEC_REPOS: ClassVar[list[str]] = ["TencentGameMate/chinese-wav2vec2-base"]
    INFINITETALK_REPOS: ClassVar[list[str]] = ["MeiGen-AI/InfiniteTalk"]

    # Model to size mapping
    MODEL_TO_SIZE: ClassVar[dict[str, str]] = {
        MODEL_480P: "infinitetalk-480",
        MODEL_720P: "infinitetalk-720",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Create HuggingFace model parameters
        self.ckpt_model_param = HuggingFaceRepoParameter(
            self, repo_ids=self.CKPT_REPOS, parameter_name="base_model"
        )
        self.wav2vec_model_param = HuggingFaceRepoParameter(
            self, repo_ids=self.WAV2VEC_REPOS, parameter_name="audio_encoder"
        )
        self.infinitetalk_model_param = HuggingFaceRepoParameter(
            self, repo_ids=self.INFINITETALK_REPOS, parameter_name="infinitetalk_weights"
        )

        # Add HuggingFace model parameters (creates dropdowns or warning messages)
        self.ckpt_model_param.add_input_parameters()
        self.wav2vec_model_param.add_input_parameters()
        self.infinitetalk_model_param.add_input_parameters()

        # Add prompt and mode parameters
        self._add_common_parameters()

        # Add audio input (common to both I2V and V2V)
        self._add_audio_parameter()

    def _add_audio_parameter(self) -> None:
        """Add audio input parameter."""
        from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio

        self.add_parameter(
            ParameterAudio(
                name="audio",
                tooltip="Driving audio for lip sync and expressions",
                clickable_file_browser=True,
                allow_output=False,
            )
        )

    def _add_final_parameters(self) -> None:
        """Add video output and status parameters.

        Child classes should call this after adding their specific input parameters.
        """
        # Add video output
        self._add_video_output_parameter()

        # Create status parameters for success/failure tracking (last)
        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or errors",
            result_details_placeholder="Generation status will appear here.",
        )

    def _add_common_parameters(self) -> None:
        """Add parameters common to all InfiniteTalk nodes."""
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the video content",
                multiline=True,
                placeholder_text="Describe the scene...",
                allow_output=False,
            )
        )

        self.add_parameter(
            Parameter(
                name="mode",
                input_types=["str"],
                type="str",
                default_value="clip",
                tooltip="Generation mode: 'clip' for single segment, 'streaming' for longer videos",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["clip", "streaming"])},
            )
        )

    def _add_video_output_parameter(self) -> None:
        """Add video output parameter."""
        self.add_parameter(
            Parameter(
                name="video",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                tooltip="Generated video output",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"is_full_width": True, "pulse_on_run": True},
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that all required models are downloaded."""
        errors = []
        for param in [self.ckpt_model_param, self.wav2vec_model_param, self.infinitetalk_model_param]:
            result = param.validate_before_node_run()
            if result:
                errors.extend(result)
        return errors if errors else None

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent.parent

    def _get_infinitetalk_dir(self) -> Path:
        """Get the InfiniteTalk submodule directory.

        The submodule is initialized by the advanced library during library load.
        """
        return self._get_library_root() / "InfiniteTalk"

    def _get_library_env_python(self) -> Path:
        """Get path to library venv Python executable."""
        venv_path = Path(__file__).parent.parent.parent / ".venv"
        if GriptapeNodes.OSManager().is_windows():
            venv_python_path = venv_path / "Scripts" / "python.exe"
        else:
            venv_python_path = venv_path / "bin" / "python"

        if venv_python_path.exists():
            logger.debug("Python executable found at: %s", venv_python_path)
            return venv_python_path

        msg = f"Python executable not found in expected location: {venv_python_path}"
        raise FileNotFoundError(msg)

    def _get_model_path(self, repo_id: str, revision: str | None = None) -> Path:
        """Get HuggingFace cache path for model."""
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, revision=revision))

    def _get_size_from_model(self, model_name: str) -> str:
        """Derive size parameter from model name using constant mapping."""
        return self.MODEL_TO_SIZE.get(model_name, "infinitetalk-480")

    async def _run_inference(
        self,
        input_json_path: Path,
        output_dir: Path,
        mode: str,
    ) -> Path | None:
        """Run InfiniteTalk inference via subprocess.

        Args:
            input_json_path: Path to the input JSON configuration
            output_dir: Directory for output files
            mode: Generation mode ('clip' or 'streaming')

        Returns:
            Path to the generated video, or None if generation failed
        """
        try:
            library_env_python = self._get_library_env_python()
        except FileNotFoundError as e:
            error_msg = f"Failed to find python executable: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            self._handle_failure_exception(e)
            return None

        # Get InfiniteTalk directory (submodule initialized by advanced library)
        infinitetalk_dir = self._get_infinitetalk_dir()
        if not infinitetalk_dir.exists() or not any(infinitetalk_dir.iterdir()):
            error_msg = f"InfiniteTalk submodule not initialized: {infinitetalk_dir}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            return None

        script_path = infinitetalk_dir / "generate_infinitetalk.py"

        # Get model paths from HuggingFace parameters
        ckpt_repo, ckpt_revision = self.ckpt_model_param.get_repo_revision()
        wav2vec_repo, wav2vec_revision = self.wav2vec_model_param.get_repo_revision()
        infinitetalk_repo, infinitetalk_revision = self.infinitetalk_model_param.get_repo_revision()

        # Derive size from selected base model
        size = self._get_size_from_model(ckpt_repo)

        # Get model paths
        ckpt_path = self._get_model_path(ckpt_repo, ckpt_revision)
        wav2vec_path = self._get_model_path(wav2vec_repo, wav2vec_revision)
        infinitetalk_path = self._get_model_path(infinitetalk_repo, infinitetalk_revision)

        # InfiniteTalk weights are in a subfolder
        infinitetalk_weights = infinitetalk_path / "single" / "infinitetalk.safetensors"

        command = [
            str(library_env_python),
            "-u",
            str(script_path),
            "--ckpt_dir", str(ckpt_path),
            "--wav2vec_dir", str(wav2vec_path),
            "--infinitetalk_dir", str(infinitetalk_weights),
            "--input_json", str(input_json_path),
            "--size", size,
            "--mode", mode,
            "--save_dir", str(output_dir),
        ]

        logger.info("Running InfiniteTalk inference: %s", " ".join(command))

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_output = stderr.decode() if stderr else "Unknown error"
                error_msg = f"InfiniteTalk inference failed with code {process.returncode}: {error_output}"
                logger.error(error_msg)
                self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
                return None

            # Find the output video file
            output_videos = list(output_dir.glob("*.mp4"))
            if not output_videos:
                error_msg = "No output video found after inference"
                self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
                return None

            # Return the most recently modified video
            return max(output_videos, key=lambda p: p.stat().st_mtime)

        except Exception as e:
            error_msg = f"Failed to execute InfiniteTalk inference: {e}"
            logger.exception(error_msg)
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            self._handle_failure_exception(e)
            return None

    def _prepare_and_run_inference(
        self,
        prompt: str,
        cond_media_path: Path,
        audio_path: Path,
        mode: str,
    ) -> None:
        """Prepare input JSON and run inference.

        This is the main processing logic used by both Image2Video and Video2Video nodes.
        """
        self._clear_execution_status()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Build input JSON
            input_config = build_input_json(
                prompt=prompt,
                cond_media_path=cond_media_path,
                audio_paths={"person1": audio_path},
            )

            input_json_path = temp_path / "input.json"
            input_json_path.write_text(json.dumps(input_config, indent=2))

            output_dir = temp_path / "output"
            output_dir.mkdir(exist_ok=True)

            # Run inference synchronously in the async wrapper
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                output_video_path = loop.run_until_complete(
                    self._run_inference(input_json_path, output_dir, mode)
                )
            finally:
                loop.close()

            if output_video_path is None:
                self.parameter_output_values["video"] = None
                return

            # Save to static storage
            filename = f"infinitetalk_{int(time.time())}.mp4"
            video_artifact = save_video_to_static(output_video_path, filename)
            self.parameter_output_values["video"] = video_artifact
            self._set_status_results(
                was_successful=True,
                result_details=f"Video generated successfully and saved as {filename}.",
            )
