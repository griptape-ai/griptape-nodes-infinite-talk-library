"""Advanced library hooks for InfiniteTalk library initialization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary

if TYPE_CHECKING:
    from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logger = logging.getLogger("griptape_nodes_infinite_talk_library")


class InfiniteTalkLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library hooks for InfiniteTalk library.

    Handles git submodule initialization before nodes are loaded.
    """

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Initialize the InfiniteTalk git submodule before loading nodes."""
        logger.info("Loading InfiniteTalk library: %s", library_data.name)
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

        if infinitetalk_dir.exists() and any(infinitetalk_dir.iterdir()):
            logger.info("InfiniteTalk submodule already initialized")
            return infinitetalk_dir

        logger.info("Initializing InfiniteTalk submodule...")
        git_repo_root = library_root.parent
        self._update_submodules_recursive(git_repo_root)

        if not infinitetalk_dir.exists() or not any(infinitetalk_dir.iterdir()):
            msg = f"Submodule initialization failed: {infinitetalk_dir} is empty or does not exist"
            raise RuntimeError(msg)

        logger.info("InfiniteTalk submodule initialized successfully")
        return infinitetalk_dir
