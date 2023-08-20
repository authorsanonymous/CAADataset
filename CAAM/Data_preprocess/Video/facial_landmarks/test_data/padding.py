import numpy as np
from keras.preprocessing import sequence

test_set_IDs =  [1004, 1005, 1009, 1012, 1014, 1029, 1037, 1041, 1050, 1055, 1069, 1072, 1083, 1088, 1090, 1091, 1095, 1104, 1112, 1115,
				1117, 1118, 1121, 1127, 1138, 1140, 1159, 1173, 1178, 1185, 1186, 1189, 1195, 1201, 1210, 1216, 1223, 1238, 1248, 1251,
				1253, 1255, 1264, 1270, 1272, 1273, 1274, 1275, 1276, 1277, 1284, 1288, 1289, 1293, 1297, 1303, 1313, 1314, 1317, 1328,
				1332, 1333, 1348, 1350, 1356, 1370, 1379, 1386, 1394, 1399, 1411, 1420, 1429, 1430, 1439, 1441, 1452, 1463, 1465, 1500,
				1502, 1506, 1508, 1510, 1517, 1520, 1527, 1528, 1533, 1540, 1557, 1558, 1566, 1569, 1574, 1576, 1585, 1589, 1604, 1627,
				1640, 1650, 1651, 1653, 1662, 1664, 1666, 1671, 1674, 1675, 1680, 1682, 1683, 1684, 1699, 1706, 1723, 1724, 1726, 1727,
				1735, 1736, 1737, 1741, 1745, 1749, 1750, 1754, 1756, 1761, 1766, 1768, 1769, 1772, 1773, 1781, 1787, 1793, 1796, 1800,
				1814, 1816, 1820, 1825, 1838, 1844, 1851, 1857, 1859, 1865, 1882, 1887, 1892, 1893, 1899, 1912, 1913, 1914, 1915, 1917,
				1920, 1922, 1924, 1925, 1934, 1938, 1947, 1954, 1961, 1965, 1967, 1969, 1970, 1973, 1978, 1984, 1985, 1986, 1992, 2001,
				2007, 2008, 2010, 2011, 2015, 2018, 2019, 2020, 2022, 2028, 2036, 2037, 2040, 2042, 2058, 2059, 2063, 2069, 2074, 2079,
				2081, 2086, 2087, 2088, 2092, 2102, 2105, 2110, 2112, 2114, 2117, 2119, 2128, 2129, 2133, 2142, 2144, 2151, 2157, 2168,
				2169, 2170, 2175, 2177, 2178, 2198, 2200, 2204, 2205, 2214, 2216, 2219, 2220, 2226, 2227, 2236, 2241, 2242, 2249, 2258,
				2262, 2263, 2269, 2270, 2271, 2276, 2278, 2284, 2290, 2294, 2308, 2309, 2319, 2329, 2330, 2331, 2332, 2333, 2337, 2346,
				2349, 2351, 2352, 2364, 2371, 2372, 2376, 2378, 2381, 2383, 2384, 2388, 2394, 2396, 2402, 2404, 2407, 2414, 2417, 2418,
				2422, 2432, 2433, 2434, 2441, 2448, 2449, 2450, 2451, 2453, 2463, 2464, 2472, 2477, 2479, 2481, 2482, 2486, 2488, 2489]


X = []

for ID in test_set_IDs:
    print(ID, end='\r')
    location = './'+str(ID)+'_facial_landmarks.npy'
    a = np.load(location).T
    X.append(sequence.pad_sequences(a, maxlen=10000, dtype='float32', padding='pre').T.tolist())

X = np.array(X)

np.save('./facial_landmarks.npy', X)