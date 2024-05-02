from dataclasses import dataclass
from bnp.components import DP, InfiniteArray
from toposort import toposort_flatten


@dataclass
class Schema:
    dependency_graph: dict
    obs_class: int
    priors: list      # Priors on any class-wide parameters
    likelihoods: list # List of functions, from (class-parameters, {foreign_key: entity}) -> entity
    alphas: list[float]

def PClean(schema):
    # Topologically sort the classes.
    classes = toposort_flatten(schema.dependency_graph)

    # Instantiate each class
    tables = []
    for cls in classes:
        if cls == schema.obs_class:
            continue
        
        def generate_table(cls=cls):
            latent = schema.priors[cls]()
            return DP(
                schema.alphas[cls],
                lambda: schema.likelihoods[cls](
                    latent, {fk: tables[fk]() for fk in schema.dependency_graph[cls]}
                ),
            )
        tables.append(generate_table())
    
    # Generate the observed class
    latent = schema.priors[schema.obs_class]()
    observed = InfiniteArray(lambda i: schema.likelihoods[schema.obs_class](
        latent, {fk: tables[fk]() for fk in schema.dependency_graph[schema.obs_class]}
    ))
    
    return (observed, tables)