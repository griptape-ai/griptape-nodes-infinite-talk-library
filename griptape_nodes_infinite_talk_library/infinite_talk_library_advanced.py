"""Advanced library hooks for InfiniteTalk library initialization."""

from __future__ import annotations

import logging
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("griptape_nodes_infinite_talk_library")


class InfiniteTalkLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library hooks for InfiniteTalk library.

    Handles git submodule initialization and dependency installation before nodes are loaded.
    """

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Initialize the InfiniteTalk git submodule and install dependencies."""
        logger.info("Loading InfiniteTalk library: %s", library_data.name)

        # Apply Python 3.11+ compatibility patches before any InfiniteTalk code is imported
        self._apply_python_compatibility_patches()

        # Check if dependencies are already installed
        if self._check_dependencies_installed():
            logger.info("InfiniteTalk dependencies already installed, skipping installation")
            # Still need to ensure submodule is initialized
            self._init_infinitetalk_submodule()
            return

        logger.info("InfiniteTalk dependencies not found, beginning installation...")
        self._install_infinitetalk_dependencies()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Log completion of library loading and configure PyTorch."""
        logger.info(
            "InfiniteTalk library loaded: %d nodes registered",
            len(library.get_registered_nodes()),
        )
        self._configure_pytorch_settings()

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent

    def _apply_python_compatibility_patches(self) -> None:
        """Apply compatibility patches for Python 3.11+.

        InfiniteTalk uses inspect.ArgSpec which was removed in Python 3.11.
        This patches the inspect module to restore compatibility.
        """
        import inspect

        if not hasattr(inspect, "ArgSpec"):
            inspect.ArgSpec = inspect.FullArgSpec
            logger.debug("Applied Python 3.11+ compatibility patch for inspect.ArgSpec")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
            logger.debug("Applied Python 3.11+ compatibility patch for inspect.getargspec")

    def _get_venv_python_path(self) -> Path:
        """Get the Python executable path from the library's venv."""
        venv_path = self._get_library_root() / ".venv"

        if GriptapeNodes.OSManager().is_windows():
            venv_python_path = venv_path / "Scripts" / "python.exe"
        else:
            venv_python_path = venv_path / "bin" / "python"

        if not venv_python_path.exists():
            raise RuntimeError(
                f"Library venv Python not found at {venv_python_path}. "
                "The library venv must be initialized before loading."
            )

        logger.debug("Python executable found at: %s", venv_python_path)
        return venv_python_path

    def _configure_pytorch_settings(self) -> None:
        """Configure PyTorch TF32 settings for Ampere+ GPUs."""
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.debug("PyTorch TF32 settings enabled for GPU acceleration")
        except ImportError:
            logger.warning("PyTorch not available, skipping TF32 configuration")

    def _check_dependencies_installed(self) -> bool:
        """Check if InfiniteTalk dependencies are installed."""
        try:
            # Check for a key dependency from InfiniteTalk's requirements
            easydict_version = version("easydict")
            logger.debug("Found easydict %s", easydict_version)

            # Log torch for debugging
            try:
                import torch
                logger.debug(
                    "Found torch %s, CUDA: %s",
                    torch.__version__,
                    torch.version.cuda if torch.cuda.is_available() else "N/A"
                )
            except ImportError:
                logger.debug("torch not found")

            return True

        except PackageNotFoundError:
            logger.debug("easydict not found")
            return False

    def _ensure_pip_installed(self) -> None:
        """Ensure pip is installed in the library's venv."""
        python_path = self._get_venv_python_path()

        result = subprocess.run(
            [str(python_path), "-m", "pip", "--version"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.debug("pip already installed: %s", result.stdout.strip())
            return

        logger.info("pip not found in venv, installing with ensurepip...")
        subprocess.run(
            [str(python_path), "-m", "ensurepip", "--upgrade"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("pip installed successfully")

    def _run_pip_install(self, args: list[str]) -> None:
        """Run pip install with the given arguments."""
        python_path = self._get_venv_python_path()
        cmd = [str(python_path), "-m", "pip", "install", *args]
        logger.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                logger.debug(result.stdout)
            if result.stderr:
                logger.debug(result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error("pip install failed with exit code %d", e.returncode)
            if e.stdout:
                logger.error("stdout: %s", e.stdout)
            if e.stderr:
                logger.error("stderr: %s", e.stderr)
            raise

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        """Recursively update and initialize all submodules using pygit2.

        Equivalent to: git submodule update --init --recursive
        """
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)

        # Recursively update nested submodules
        for submodule in repo.submodules:
            submodule_path = repo_path / submodule.path
            if submodule_path.exists() and (submodule_path / ".git").exists():
                self._update_submodules_recursive(submodule_path)

    def _init_infinitetalk_submodule(self) -> Path:
        """Initialize the InfiniteTalk git submodule."""
        library_root = self._get_library_root()
        infinitetalk_dir = library_root / "InfiniteTalk"

        # Check if submodule is already initialized
        if infinitetalk_dir.exists() and any(infinitetalk_dir.iterdir()):
            logger.info("InfiniteTalk submodule already initialized")
            return infinitetalk_dir

        logger.info("Initializing InfiniteTalk submodule...")
        git_repo_root = library_root.parent
        self._update_submodules_recursive(git_repo_root)

        if not infinitetalk_dir.exists() or not any(infinitetalk_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {infinitetalk_dir} is empty or does not exist"
            )

        logger.info("InfiniteTalk submodule initialized successfully")
        return infinitetalk_dir

    def _install_infinitetalk_dependencies(self) -> None:
        """Install InfiniteTalk and required dependencies."""
        try:
            logger.info("=" * 80)
            logger.info("Installing InfiniteTalk Library Dependencies...")
            logger.info("=" * 80)

            # Ensure pip is available
            self._ensure_pip_installed()

            # Step 1/2: Initialize InfiniteTalk submodule
            logger.info("Step 1/2: Initializing InfiniteTalk submodule...")
            infinitetalk_dir = self._init_infinitetalk_submodule()

            # Step 2/3: Install requirements
            logger.info("Step 2/3: Installing InfiniteTalk requirements...")
            requirements_file = infinitetalk_dir / "requirements.txt"

            if not requirements_file.exists():
                raise RuntimeError(f"requirements.txt not found: {requirements_file}")

            self._run_pip_install(["--force-reinstall", "-r", str(requirements_file)])

            # Step 3/3: Install additional dependencies from README (not in requirements.txt)
            # Note: flash_attn is Linux-only, so we skip it on Windows
            logger.info("Step 3/3: Installing additional dependencies...")
            additional_deps = [
                "misaki[en]",
                "ninja",
                "psutil",
                "packaging",
                "wheel",
                "sentencepiece",
                "transformers>=4.49.0,<5.0.0",
            ]
            self._run_pip_install(["--force-reinstall", *additional_deps])

            logger.info("InfiniteTalk installation completed successfully!")
            logger.info("=" * 80)

        except Exception as e:
            error_msg = f"Failed to install InfiniteTalk dependencies: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
