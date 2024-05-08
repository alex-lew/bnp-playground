# "HIRM + PClean" model suggested by Emily Fertig.
# This is a different model from the one in v1.py and v2.py.
# It is much closer to a standard HIRM, but we add a layer of clustering
# to model that we may not be observing "true" entity IDs (i.e., entities
# recorded as distinct in the observed data may in fact correspond to the
# same entity).

from ..hirm.v1 import HIRM
from ..components import InfiniteArray, GEM


def Hybrid(view_alpha, cluster_alpha, entity_res_alpha, relations):
    # First, create an HIRM
    hirm = HIRM(view_alpha, cluster_alpha, relations)

    # For each entity type in the relations, create a one-to-many 
    # mapping from observed entity ID to true entity ID
    choose_true_entity = InfiniteArray(lambda entity_type: GEM(entity_res_alpha))
    resolved_entities = InfiniteArray(
        lambda entity_type, entity_id: choose_true_entity[entity_type]()
    )

    # Create a layer of indirection to look up entities by their true IDs
    data = InfiniteArray(
        lambda relation_id, entity_ids: hirm[
            relation_id,
            tuple(
                resolved_entities[relations[relation_id].entity_types[i], entity_id]
                for (i, entity_id) in enumerate(entity_ids)
            ),
        ]
    )
    
    return data
