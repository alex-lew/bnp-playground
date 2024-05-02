from ..components import InfiniteArray, GEM

def CrossCat(view_alpha, cluster_alpha, columns):
    # Per-column hyperparameters
    hypers = InfiniteArray(lambda col_id: columns[col_id].hyperprior())

    # Per-view clusterings
    viewDistribution = GEM(view_alpha)
    views = InfiniteArray(lambda col_id: viewDistribution())
    clusterDistributions = InfiniteArray(lambda view_id: GEM(cluster_alpha))
    clusters = InfiniteArray(lambda view_id, row_id: clusterDistributions[view_id]())

    # Per-cluster latents
    latents = InfiniteArray(
        lambda col_id, cluster_id: columns[col_id].prior(hypers[col_id])
    )

    # Data matrix
    data = InfiniteArray(
        lambda row_id, col_id: columns[col_id].likelihood(
            latents[col_id, clusters[views[col_id], row_id]]
        )
    )
    return data