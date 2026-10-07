# from blind-counsel 8b1bf8f config.py
"""Local paths. Milestone 3 and the rich-DB prototype read UK legislation and the
router index from a checkout of https://github.com/azterizm/legal-rag-router.
Set LEGAL_RAG_ROUTER to its path, or clone it next to this repository."""
from __future__ import annotations

import os

LRR = os.environ.get(
    "LEGAL_RAG_ROUTER",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legal-rag-router")),
)
