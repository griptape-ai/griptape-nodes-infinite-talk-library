"""Advanced library hooks for InfiniteTalk library initialization."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("griptape_nodes_infinite_talk_library")


class InfiniteTalkLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library hooks for InfiniteTalk library.

    Ensures the InfiniteTalk git submodule is present before nodes are loaded. The submodule
    carries the model implementation as a plain source tree rather than a package on an index,
    so it cannot be expressed as a dependency and has to be fetched with git.
    """

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Initialize the InfiniteTalk git submodule."""
        logger.info("Loading InfiniteTalk library: %s", library_data.name)
        # This hook runs on both the orchestrator and the worker. Only the worker runs the inference
        # subprocess that puts the submodule's source tree on `sys.path`, so the orchestrator has no
        # use for the checkout.
        if not GriptapeNodes.LibraryManager().is_worker:
            return
        self._init_infinitetalk_submodule()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Log completion of library loading."""
        logger.info(
            "InfiniteTalk library loaded: %d nodes registered",
            len(library.get_registered_nodes()),
        )

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent

    def _init_infinitetalk_submodule(self) -> None:
        """Initialize the InfiniteTalk git submodule."""
        library_root = self._get_library_root()
        infinitetalk_dir = library_root / "InfiniteTalk"

        # Check if submodule is already initialized
        if infinitetalk_dir.exists() and any(infinitetalk_dir.iterdir()):
            logger.info("InfiniteTalk submodule already initialized")
            return

        logger.info("Initializing InfiniteTalk submodule...")
        # The git CLI rather than pygit2: the engine dropped pygit2 (its bundled TLS trust
        # store breaks on some platforms) and requires git on PATH, so it is the one tool
        # guaranteed to be here.
        git_repo_root = library_root.parent
        subprocess.check_call(["git", "-C", str(git_repo_root), "submodule", "update", "--init", "--recursive"])

        if not infinitetalk_dir.exists() or not any(infinitetalk_dir.iterdir()):
            raise RuntimeError(f"Submodule initialization failed: {infinitetalk_dir} is empty or does not exist")

        logger.info("InfiniteTalk submodule initialized successfully")
