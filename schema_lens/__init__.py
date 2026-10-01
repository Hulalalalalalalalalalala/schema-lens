"""schema-lens: infer and validate schemas over JSONL record streams."""

from .lens import Lens, SchemaConflict

__all__ = ["Lens", "SchemaConflict"]
