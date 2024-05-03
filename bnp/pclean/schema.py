from dataclasses import dataclass

@dataclass
class Schema:
    dependency_graph: dict
    obs_class: int
    priors: list  # Priors on any class-wide parameters
    likelihoods: list  # List of functions, from (class-parameters, {foreign_key: entity}) -> entity
    alphas: list[float]
