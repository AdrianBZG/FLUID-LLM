import numpy as np

x = \
 [0.0000, 0.0072, 0.0140, 0.0211, 0.0273, 0.0335, 0.0402, 0.0459, 0.0514,
  0.0569, 0.0622, 0.0676, 0.0727, 0.0777, 0.0827, 0.0876, 0.0924, 0.0973,
  0.1019, 0.1065, 0.1112, 0.1160, 0.1209, 0.1256, 0.1305, 0.1352, 0.1398,
  0.1443, 0.1491, 0.1541, 0.1594, 0.1648, 0.1703, 0.1757, 0.1809, 0.1862,
  0.1911, 0.1955, 0.1994, 0.2037, 0.2083, 0.2126, 0.2172, 0.2218, 0.2266,
  0.2313, 0.2360, 0.2411, 0.2460, 0.2511, 0.2562, 0.2615, 0.2670, 0.2728,
  0.2784, 0.2843, 0.2903, 0.2963, 0.3023, 0.3083, 0.3144, 0.3206, 0.3270,
  0.3333, 0.3398, 0.3460, 0.3524, 0.3589, 0.3653, 0.3715, 0.3774, 0.3831,
  0.3889, 0.3943, 0.4000, 0.4060, 0.4122, 0.4188, 0.4252, 0.4322, 0.4394,
  0.4469, 0.4548, 0.4625, 0.4705, 0.4783, 0.4865, 0.4948, 0.5032, 0.5116,
  0.5197, 0.5275, 0.5351, 0.5425, 0.5501, 0.5578, 0.5658, 0.5741, 0.5824,
  0.5909, 0.5993, 0.6076, 0.6156, 0.6235, 0.6314, 0.6391, 0.6469, 0.6548,
  0.6628, 0.6707, 0.6787, 0.6871, 0.6954, 0.7043, 0.7129, 0.7217, 0.7308,
  0.7399, 0.7491, 0.7585, 0.7674, 0.7764, 0.7852, 0.7937, 0.8020, 0.8101,
  0.8182, 0.8266, 0.8353, 0.8448, 0.8547, 0.8650, 0.8755, 0.8860, 0.8969,
  0.9078, 0.9187, 0.9295, 0.9400, 0.9506, 0.9607, 0.9706, 0.9799, 0.9895,
  0.9995, 1.0088, 1.0183, 1.0278, 1.0374, 1.0470, 1.0570, 1.0668, 1.0772,
  1.0883, 1.0998, 1.1117, 1.1232, 1.1349, 1.1466, 1.1583, 1.1698, 1.1816,
  1.1934, 1.2051, 1.2167, 1.2284, 1.2401, 1.2516, 1.2631, 1.2743, 1.2852,
  1.2964, 1.3073, 1.3181, 1.3288, 1.3401, 1.3510, 1.3616, 1.3723, 1.3829,
  1.3934, 1.4038, 1.4143, 1.4244, 1.4352, 1.4466, 1.4585, 1.4709, 1.4835,
  1.4960, 1.5084, 1.5210, 1.5329, 1.5446, 1.5560, 1.5673, 1.5783, 1.5893,
  1.6003, 1.6112, 1.6223, 1.6333, 1.6441, 1.6549, 1.6658, 1.6765, 1.6867,
  1.6978, 1.7085, 1.7191, 1.7298, 1.7405, 1.7513, 1.7622, 1.7725, 1.7832,
  1.7941, 1.8053, 1.8166, 1.8280, 1.8398, 1.8518, 1.8639, 1.8763, 1.8886,
  1.9009, 1.9129, 1.9254, 1.9371, 1.9485, 1.9596, 1.9705, 1.9813, 1.9921,
  2.0032, 2.0143, 2.0255, 2.0370, 2.0490, 2.0609, 2.0727, 2.0845, 2.0960,
  2.1077, 2.1192, 2.1298, 2.1409, 2.1519, 2.1628, 2.1741, 2.1853]

x = np.array(x)
print(f'{len(x) = }')
print(x[:51].mean()*0.275)

