from typing import TypeAlias
from bnp.column_types import ColumnInfo
from dataclasses import dataclass

# Configuration for an IRM-style relation
# mapping a list of entities to some output value
EntityType: TypeAlias = int

@dataclass
class RelationInfo:
    entity_types: list[EntityType]
    output: ColumnInfo
