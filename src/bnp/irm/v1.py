# Infinite Relational Model
from ..components import InfiniteArray, GEM

def IRM(alpha, relations):
    # Cluster entities of each type.
    clusterDistributions = InfiniteArray(lambda entity_type: GEM(alpha))
    clusters = InfiniteArray(
        lambda entity_type, entity_id: clusterDistributions[entity_type]()
    )
    hypers = InfiniteArray(
        lambda relation_id: relations[relation_id].output.hyperprior()
    )
    latents = InfiniteArray(
        lambda relation_id, entity_clusters: relations[relation_id].output.prior(
            hypers[relation_id]
        )
    )
    data = InfiniteArray(
        lambda relation_id, entity_ids: relations[relation_id].output.likelihood(
            latents[
                relation_id,
                tuple(
                    clusters[relations[relation_id].entity_types[i], entity_id]
                    for (i, entity_id) in enumerate(entity_ids)
                ),
            ]
        )
    )
    return data
