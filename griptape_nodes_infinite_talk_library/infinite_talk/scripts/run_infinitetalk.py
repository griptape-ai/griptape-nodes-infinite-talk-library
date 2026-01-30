"""Wrapper script to run InfiniteTalk with Python 3.11+ compatibility patches."""

import inspect
import sys

# Apply Python 3.11+ compatibility patches before importing InfiniteTalk
if not hasattr(inspect, "ArgSpec"):
    inspect.ArgSpec = inspect.FullArgSpec

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# Force eager attention for wav2vec2 - SDPA doesn't support output_attentions=True
# which InfiniteTalk's wav2vec2.py requires
from transformers import Wav2Vec2Config
_original_wav2vec2_init = Wav2Vec2Config.__init__
def _patched_wav2vec2_init(self, *args, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    _original_wav2vec2_init(self, *args, **kwargs)
Wav2Vec2Config.__init__ = _patched_wav2vec2_init

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
