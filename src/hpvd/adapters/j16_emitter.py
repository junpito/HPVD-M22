"""
J16 Emitter
============

Wraps ``FamilyAssignment`` list into a ``J16_AnalogFamilyAssignment`` envelope.
"""

from typing import Any, Dict, List

from .j_file_schemas import J16_AnalogFamilyAssignment
from .retrieval_strategy import FamilyAssignment


class J16Emitter:
    """Emit ``J16_AnalogFamilyAssignment`` from computed families."""

    @staticmethod
    def emit(
        query_id: str,
        families: List[FamilyAssignment],
        metadata: Dict[str, Any] | None = None,
    ) -> J16_AnalogFamilyAssignment:
        total_members = sum(f.coherence.size for f in families)
        return J16_AnalogFamilyAssignment(
            query_id=query_id,
            families=[f.to_dict() for f in families],
            total_members=total_members,
            total_families=len(families),
            metadata=metadata or {},
        )
