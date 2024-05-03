from bnp.column_types import BetaBernoulli, NIGNormal
from bnp.irm.relation import RelationInfo
from bnp.dpmm.v2 import DPMM
from bnp.crosscat.v1 import CrossCat
from bnp.irm.v1 import IRM
from bnp.hirm.v1 import HIRM

# Tabular data example with five columns
columns = [BetaBernoulli, BetaBernoulli, NIGNormal, NIGNormal, NIGNormal]

print("DPMM data:")
print("----------")
data = DPMM(1, columns)
for row_id in range(10):
    for col_id in range(5):
        print(data[row_id, col_id], end="\t")
    print()

from bnp.components import print_sample
print_sample(data)

print("CrossCat data:")
print("--------------")
data = CrossCat(1, 1, columns)
for row_id in range(10):
    for col_id in range(5):
        print(data[row_id, col_id], end="\t")
    print()
    
print_sample(data)

# IRM data example with one entity types and two relations
country = 1
allies = RelationInfo([country, country], BetaBernoulli)
exports = RelationInfo([country, country], NIGNormal)
relations = [allies, exports]
data = IRM(1, relations)
print("IRM data:")
print("---------")
for relation_id in range(2):
    print("Relation %d:" % relation_id)
    for entity_id_1 in range(10):
        for entity_id_2 in range(10):
            print(f"\t{entity_id_1}", end="\t")
            print(
                f"{entity_id_2}\t{data[relation_id, (entity_id_1, entity_id_2)]}",
                end="\t",
            )
            print()
    print()

print_sample(data)

data = HIRM(1, 1, relations)
print("HIRM data:")
print("---------")
for relation_id in range(2):
    print("Relation %d:" % relation_id)
    for entity_id_1 in range(10):
        for entity_id_2 in range(10):
            print(f"\t{entity_id_1}", end="\t")
            print(
                f"{entity_id_2}\t{data[relation_id, (entity_id_1, entity_id_2)]}",
                end="\t",
            )
            print()
    print()


# PClean
from bnp.pclean.v1 import PClean
from bnp.pclean.schema import Schema
import numpy as np

# We assume two classes.
schema = Schema({1: {0}, 0: {}}, 1, [lambda: np.random.uniform(), lambda: np.random.uniform()], 
                [lambda theta, _: np.random.uniform() < theta, lambda theta, d: (d[0], 4 if d[0] else np.random.normal(theta,1))], [1.0, 1.0])

data, db = PClean(schema)
print("PClean data:")
print("---------")
print([data[i] for i in range(20)])

# Hybrid
from bnp.hybrid.v2 import Hybrid
from bnp.hybrid.schema import HybridSchema
schema = HybridSchema({1: {0}, 0: {}}, 1, [[BetaBernoulli, BetaBernoulli], [BetaBernoulli, NIGNormal, NIGNormal]], [1.0, 1.0])
data, objects = Hybrid(1, 1, schema)
print("Hybrid data:")
print("---------")
print([data[i] for i in range(20)])