"""NeuroScan Phase 2 — shared code for the three modelling tracks.

Every model trained in Phase 2 goes through `neuroscan.data` for its inputs and
`neuroscan.evaluate` for its metrics, so that results from the three tracks are
directly comparable. See PHASE2.md for the agreed contract.
"""

__all__ = ["data", "evaluate"]
__version__ = "0.1.0"
