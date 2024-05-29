import numpy as np

x = \
 [0.0000, 0.0105, 0.0184, 0.0261, 0.0332, 0.0398, 0.0466, 0.0526, 0.0595,
  0.0658, 0.0719, 0.0778, 0.0837, 0.0891, 0.0948, 0.1005, 0.1059, 0.1121,
  0.1175, 0.1223, 0.1274, 0.1321, 0.1368, 0.1415, 0.1462, 0.1509, 0.1552,
  0.1596, 0.1637, 0.1676, 0.1714, 0.1754, 0.1799, 0.1846, 0.1895, 0.1938,
  0.1994, 0.2039, 0.2091, 0.2140, 0.2188, 0.2236, 0.2276, 0.2321, 0.2373,
  0.2425, 0.2464, 0.2513, 0.2558, 0.2602, 0.2645, 0.2685, 0.2731, 0.2777,
  0.2816, 0.2854, 0.2897, 0.2935, 0.2983, 0.3032, 0.3083, 0.3134, 0.3184,
  0.3226, 0.3265, 0.3303, 0.3342, 0.3388, 0.3440, 0.3488, 0.3554, 0.3609,
  0.3663, 0.3713, 0.3768, 0.3822, 0.3872, 0.3917, 0.3971, 0.4012, 0.4068,
  0.4109, 0.4157, 0.4188, 0.4228, 0.4275, 0.4318, 0.4360, 0.4407, 0.4456,
  0.4499, 0.4548, 0.4593, 0.4637, 0.4685, 0.4734, 0.4786, 0.4839, 0.4891,
  0.4931, 0.4973, 0.5018, 0.5068, 0.5125, 0.5187, 0.5250, 0.5310, 0.5365,
  0.5420, 0.5475, 0.5521, 0.5566, 0.5606, 0.5645, 0.5694, 0.5738, 0.5783,
  0.5819, 0.5859, 0.5914, 0.5959, 0.6003, 0.6044, 0.6087, 0.6136, 0.6181,
  0.6225, 0.6271, 0.6316, 0.6365, 0.6416, 0.6468, 0.6514, 0.6560, 0.6613,
  0.6665, 0.6709, 0.6761, 0.6824, 0.6879, 0.6939, 0.6998, 0.7050, 0.7097,
  0.7148, 0.7189, 0.7226, 0.7260, 0.7298, 0.7341, 0.7371, 0.7411, 0.7461,
  0.7512, 0.7555, 0.7597, 0.7635, 0.7675, 0.7714, 0.7746, 0.7786, 0.7822,
  0.7863, 0.7900, 0.7938, 0.7980, 0.8026, 0.8075, 0.8119, 0.8172, 0.8231,
  0.8283, 0.8339, 0.8399, 0.8459, 0.8514, 0.8564, 0.8609, 0.8650, 0.8689,
  0.8714, 0.8740, 0.8764, 0.8797, 0.8833, 0.8860, 0.8899, 0.8939, 0.8980,
  0.9023, 0.9069, 0.9118, 0.9169, 0.9220, 0.9278, 0.9324, 0.9385, 0.9453,
  0.9521, 0.9587, 0.9651, 0.9715, 0.9769, 0.9833, 0.9888, 0.9933, 0.9973,
  1.0017, 1.0049, 1.0088, 1.0126, 1.0149, 1.0167, 1.0191, 1.0207, 1.0230,
  1.0252, 1.0281, 1.0325, 1.0360, 1.0413, 1.0462, 1.0511, 1.0570, 1.0631,
  1.0696, 1.0756, 1.0811, 1.0877, 1.0939, 1.1000, 1.1052, 1.1108, 1.1160,
  1.1215, 1.1268, 1.1316, 1.1358, 1.1406, 1.1451, 1.1490, 1.1536, 1.1582,
  1.1627, 1.1666, 1.1706, 1.1752, 1.1806, 1.1849, 1.1890, 1.1934],

x = np.array(x)
print(f'{len(x) = }')
print(x[:50].mean()*0.275)

