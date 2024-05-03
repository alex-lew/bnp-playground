from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Any, TypeAlias
import numpy as np

ParameterType = TypeVar("ParameterType")
LatentType = TypeVar("LatentType")
DataType = TypeVar("DataType")

# Configuration for a CrossCat-style column
@dataclass
class ColumnInfo(Generic[LatentType, DataType]):
    hyperprior: Callable[[], ParameterType]
    prior: Callable[[ParameterType], LatentType]
    likelihood: Callable[[LatentType], DataType]


# Example configurations
def grid_hyperprior(grid: np.ndarray) -> Callable[[], ParameterType]:
    return lambda: grid[np.random.randint(len(grid))]


# Beta-Bernoulli column
BetaBernoulli = ColumnInfo(
    grid_hyperprior(
        [
            (alpha, beta)
            for alpha in np.linspace(0.1, 10, 20)
            for beta in np.linspace(0.1, 10, 20)
        ]
    ),
    lambda params: np.random.beta(params[0], params[1]),
    lambda latent: np.random.uniform() < latent,
)

# Normal-Inverse-Gamma / Normal column
NIGNormal = ColumnInfo(
    grid_hyperprior(
        [
            (mu, sigma, alpha, beta)
            for mu in np.linspace(-10, 10, 20)
            for sigma in np.linspace(0.1, 10, 20)
            for alpha in np.linspace(0.1, 10, 20)
            for beta in np.linspace(0.1, 10, 20)
        ]
    ),
    # Generate a normal-inverse-gamma latent
    lambda params: (np.random.normal(params[0], params[1]), 1 / np.random.gamma(params[2], params[3])),
    lambda latent: np.random.normal(latent[0], latent[1] ** 0.5),
)



# Configuration for an IRM-style relation
# mapping a list of entities to some output value
EntityType: TypeAlias = int

@dataclass
class RelationInfo:
    entity_types: list[EntityType]
    output: ColumnInfo
