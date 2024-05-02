# Bayesian Non-parametrics Playground

A little playground for defining and running Bayesian nonparametric models, in the style of [Roy et al. (2008)](http://danroy.org/papers/RoyManGooTen-ICMLNPB-2008.pdf).

Note that this is currently not a library for *fitting* these models, just for running them to generate synthetic data. It is intended for pedagogical purposes & quick prototyping.

### Getting started

You can install the package and run a test script using `poetry`.

```
poetry install
poetry run python examples/examples.py
```

The package provides several basic components from which more complex models can be built:

1. `InfiniteArray(f)`: given a stochastic function `f` from _indices_ to values, sample an infinite array of values by calling f(i) on every index. (In practice, this works via stochastic memoization: if `a = InifiniteArray(f)`, then the first time `a[i]` is accessed for a new index `i`, `f(i)` is called; subsequent lookups of `a[i]` then always return the same value.)
2. `GEM(alpha)`: generates an infinite vector of probabilities from a GEM (Griffiths, Engen, McCloskey) distribution.
3. `DP(alpha, H)`: generates a random probability measure with base measure `H` (a stochastic function).

It also provides examples of using these components to define various models, including DPMMs, IRMs, CrossCat models, and hierarchical IRMs.