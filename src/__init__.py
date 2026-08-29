"""DreamScript — converting rough hand-drawn diagrams into executable, runnable code.

Subpackages follow the pipeline order: ingest -> preprocess -> features / detect / ocr ->
classify -> parse -> rl -> synth, with eval, serve and utils cutting across all of them.
See docs/layout.md for the full tree and the rules that govern it.
"""

from __future__ import annotations

__version__ = "0.1.0"
