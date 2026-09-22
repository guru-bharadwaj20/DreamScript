"""Phase 15 - MLOps: tracking, data versioning, drift, serving and CI for the pipeline.

Phases 1-14 built and measured the pipeline. This package is about keeping it honest once it
runs more than once: where a run's numbers are recorded (15.1), which data is versioned and
where it lives (15.2), what is allowed to be promoted to production (15.3), how the stages
compose into a DAG (15.4), and what should fire an alarm when the input distribution moves
(15.5) or the model's confidence does (15.6).
"""

PHASE = "15"
__all__: list[str] = []
