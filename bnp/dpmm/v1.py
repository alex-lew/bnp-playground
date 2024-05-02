from ..components import InfiniteArray, GEM

def DPMM(alpha, columns):
    # Per-column hyperparameters
    hypers = InfiniteArray(lambda col_id: columns[col_id].hyperprior())

    # Per-cluster latents
    clusterDistribution = GEM(alpha)
    clusters = InfiniteArray(lambda row_id: clusterDistribution())
    latents = InfiniteArray(
        lambda col_id, cluster_id: columns[col_id].prior(hypers[col_id])
    )

    # Data matrix
    data = InfiniteArray(
        lambda row_id, col_id: columns[col_id].likelihood(
            latents[col_id, clusters[row_id]]
        )
    )
    return data
