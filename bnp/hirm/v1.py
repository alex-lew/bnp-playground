from ..components import DP, InfiniteArray
from ..irm.v1 import IRM

def HIRM(view_alpha, cluster_alpha, relations):
    # Per-view data matrices
    viewDistribution = DP(view_alpha, lambda: IRM(cluster_alpha, relations))
    views = InfiniteArray(lambda relation_id: viewDistribution())

    # Overall data matrix
    data = InfiniteArray(
        lambda relation_id, entity_ids: views[relation_id][relation_id, entity_ids]
    )
    return data
