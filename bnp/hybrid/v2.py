from ..components import InfiniteArray
from ..hirm.v1 import HIRM
from ..pclean.v1 import PClean
from .schema import hybrid_schema_to_pclean_schema, hybrid_schema_to_irm_relations


def Hybrid(view_alpha, cluster_alpha, schema):
    # First, use PClean's model to generate entities and their links (but no attributes)
    pclean_schema = hybrid_schema_to_pclean_schema(schema)
    top_level, db = PClean(pclean_schema)

    # HIRM to model the attributes
    relations, relation_ids = hybrid_schema_to_irm_relations(schema)
    hirm = HIRM(view_alpha, cluster_alpha, relations)

    # Create a way to look up objects by their keys
    objects = InfiniteArray(
        lambda class_id, obj_id, parents: {
            "key": (class_id, obj_id, parents),
            "attrs": [
                hirm[
                    relation_ids[(class_id, attr_id)], (obj_id, *[o[1] for o in parents])
                ]
                for attr_id in range(len(schema.attribute_columns[class_id]))
            ],
        }
    )

    data = InfiniteArray(lambda i: objects[top_level[i]])
    return (data, objects)
