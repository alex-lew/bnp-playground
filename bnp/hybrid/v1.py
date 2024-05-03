# PClean, but the attribute models are learned via CrossCat
from dataclasses import dataclass
from ..column_types import ColumnInfo
from toposort import toposort_flatten
from bnp.components import InfiniteArray, DP, GEM

def make_base_measure(schema, class_id, choose_foreign_object, cluster_ids, view_ids):
    hypers = InfiniteArray(lambda attr_id: schema.attribute_columns[class_id][attr_id].hyperprior())
    
    # Generate latents for each attribute for each set of cluster ids
    latents = InfiniteArray(
        lambda attr_id, all_cluster_ids: schema.attribute_columns[class_id][
            attr_id
        ].prior(hypers[attr_id])
    )

    def generate_new_entity(entity_id):
        # Draw for foreign keys
        objects = [(fk, choose_foreign_object[fk]()) for fk in schema.dependency_graph[class_id]]
        objects.append((class_id, entity_id))
        
        # Look up their cluster assignments within each relevant view
        fk_ids = [
            tuple(
                cluster_ids[fk, view_ids[class_id, attr_id], obj_id]
                for (fk, obj_id) in objects
            )
            for attr_id in range(len(schema.attribute_columns[class_id]))
        ]
        
        # Generate the values for each attribute
        return [
            schema.attribute_columns[class_id][attr_id].likelihood(latents[attr_id, fk_ids[attr_id]])
            for attr_id in range(len(schema.attribute_columns[class_id]))
        ]
    
    return generate_new_entity

def Hybrid(view_alpha, cluster_alpha, schema):
    # To each view, we associate an array (indexed by class) of GEMs (cluster distributions).
    choose_view_id = GEM(view_alpha)
    choose_cluster_id = InfiniteArray(lambda view_id: GEM(cluster_alpha))
    view_ids = InfiniteArray(lambda class_id, attr_id: choose_view_id())
    cluster_ids = InfiniteArray(
        lambda class_id, view_id, row_id: choose_cluster_id[view_id]()
    )

    # Topologically sort the latent classes.
    latent_classes = [
        c for c in toposort_flatten(schema.dependency_graph) if c != schema.obs_class
    ]

    choose_foreign_object = {}
    tables = {}
    for class_id in latent_classes:
        choose_foreign_object[class_id] = GEM(schema.alphas[class_id])
        
        # Note that in this model, attributes of latent classes are actually not used for anything (unless we later observe).
        tables[class_id] = InfiniteArray(make_base_measure(schema, class_id, choose_foreign_object, cluster_ids, view_ids))

    # Generate the observed class
    generate_observation = make_base_measure(schema, schema.obs_class, choose_foreign_object, cluster_ids, view_ids)
    observed = InfiniteArray(generate_observation)
    return (observed, tables)
