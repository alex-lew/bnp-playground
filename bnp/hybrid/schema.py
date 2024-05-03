from dataclasses import dataclass
from ..irm.relation import RelationInfo
from ..pclean.schema import Schema
from ..column_types import ColumnInfo
from ..components import MakeGensym


@dataclass
class HybridSchema:
    dependency_graph: dict
    obs_class: int
    attribute_columns: list[list[ColumnInfo]]
    alphas: list[float]

def hybrid_schema_to_irm_relations(schema):
    relation_id = {}
    relations = []
    for class_id, fks in schema.dependency_graph.items():
        for attr_id, attr in enumerate(schema.attribute_columns[class_id]):
            relation_id[(class_id, attr_id)] = len(relations)
            relations.append(RelationInfo([class_id, *fks], attr))
    return relations, relation_id


def make_likelihood(schema, gensym, class_id):
    return lambda _, fks: (
        class_id,
        gensym(),
        tuple(fks[i] for i in schema.dependency_graph[class_id]),
    )


def hybrid_schema_to_pclean_schema(schema):
    num_classes = len(schema.attribute_columns)
    gensym = MakeGensym()
    return Schema(
        dependency_graph=schema.dependency_graph,
        obs_class=schema.obs_class,
        priors=[lambda: () for _ in range(num_classes)],
        likelihoods=[
            make_likelihood(schema, gensym, class_id) for class_id in range(num_classes)
        ],
        alphas=schema.alphas,
    )
