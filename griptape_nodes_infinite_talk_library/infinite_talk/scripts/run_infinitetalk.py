"""Wrapper script to run InfiniteTalk with Python 3.11+ compatibility patches."""

import inspect
import sys

# Apply Python 3.11+ compatibility patches before importing InfiniteTalk
if not hasattr(inspect, "ArgSpec"):
    inspect.ArgSpec = inspect.FullArgSpec

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# Add InfiniteTalk directory to path
from pathlib import Path

infinitetalk_dir = Path(__file__).parent.parent.parent / "InfiniteTalk"
sys.path.insert(0, str(infinitetalk_dir))

# Now import and run the actual generate script
if __name__ == "__main__":
    # Import after patches are applied
    from generate_infinitetalk import _parse_args, generate

    args = _parse_args()
    generate(args)
