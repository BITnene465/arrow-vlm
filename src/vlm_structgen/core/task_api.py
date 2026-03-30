from __future__ import annotations

from .registry import (
    SUPPORTED_DOMAIN_TYPES,
    SUPPORTED_TASK_TYPES,
    TaskAdapter,
    get_adapter,
    normalize_domain_type,
    normalize_task_type,
)

__all__ = [
    "SUPPORTED_DOMAIN_TYPES",
    "SUPPORTED_TASK_TYPES",
    "TaskAdapter",
    "get_adapter",
    "normalize_domain_type",
    "normalize_task_type",
]
