from ..components import DP, InfiniteArray

def DPMM(alpha, columns):
    # Per-column hyperparameters
    hypers = InfiniteArray(lambda col_id: columns[col_id].hyperprior())

    # Per-cluster latents
    latentsDistribution = DP(
        alpha,
        lambda: InfiniteArray(lambda col_id: columns[col_id].prior(hypers[col_id])),
    )
    latents = InfiniteArray(lambda row_id: latentsDistribution())

    # Data matrix
    data = InfiniteArray(
        lambda row_id, col_id: columns[col_id].likelihood(latents[row_id][col_id])
    )
    return data
