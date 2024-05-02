from ..components import DP, InfiniteArray
from ..dpmm.v2 import DPMM

def CrossCat(view_alpha, cluster_alpha, columns):
    # Per-view data matrices
    viewDistribution = DP(view_alpha, lambda: DPMM(cluster_alpha, columns))
    views = InfiniteArray(lambda col_id: viewDistribution())

    # Overall data matrix
    data = InfiniteArray(lambda row_id, col_id: views[col_id][row_id, col_id])
    return data
