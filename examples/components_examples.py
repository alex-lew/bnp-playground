import numpy as np
from bnp.components import GEM, DP, PitmanYor, HDP

print("GEM samples:")
print("----------")
gem = GEM(1.)
print([gem() for _ in range(10)])

print("GEM discount=0.3 samples:")
print("----------")
gem = GEM(1., discount=0.3)
print([gem() for _ in range(10)])

print("DP samples:")
print("----------")
dp = DP(1., lambda: np.random.normal())
print([dp() for _ in range(10)])

print("HDP samples:")
print("----------")
hdp = HDP(1., 2.5, lambda: np.random.normal())
print([hdp() for _ in range(10)])

print("Pitman-Yor samples:")
print("----------")
py = PitmanYor(1., 0.5, lambda: np.random.normal())
print([py() for _ in range(10)])
