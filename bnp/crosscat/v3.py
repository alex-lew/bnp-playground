from ..components import DP, InfiniteArray
from ..hirm.v1 import HIRM
from ..irm.relation import RelationInfo

def CrossCat(view_alpha, cluster_alpha, columns):
    # CrossCat is a version of HIRM using a set of unary relations called
    # columns. There is one entity type which is the row.
    relations = [RelationInfo([1], c) for c in columns]
    hirm = HIRM(view_alpha, cluster_alpha, relations)
    return InfiniteArray(lambda row_id, col_id: hirm[col_id, (row_id,)])
