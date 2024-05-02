from dataclasses import dataclass
from bnp.components import DP, InfiniteArray
from toposort import toposort_flatten


@dataclass
class Schema:
    dependency_graph: dict
    obs_class: int
    priors: list  # Priors on any class-wide parameters
    likelihoods: list  # List of functions, from (class-parameters, {foreign_key: entity}) -> entity
    alphas: list[float]


def make_base_measure(schema, existing_tables, cls):
    latent = schema.priors[cls]()
    return lambda: schema.likelihoods[cls](
        latent, {fk: existing_tables[fk]() for fk in schema.dependency_graph[cls]}
    )


def PClean(schema):
    # Topologically sort the latent classes.
    latent_classes = [
        c for c in toposort_flatten(schema.dependency_graph) if c != schema.obs_class
    ]

    # Instantiate the tables
    tables = {}
    for cls in latent_classes:
        tables[cls] = DP(schema.alphas[cls], make_base_measure(schema, tables, cls))

    # Generate the observed class
    generate_observation = make_base_measure(schema, tables, schema.obs_class)
    observed = InfiniteArray(lambda i: generate_observation())
    return (observed, tables)
