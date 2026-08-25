import numpy as np
import time

n = 10000
y_hat = np.random.normal(size=n)
y = np.random.normal(size=n)

st_time = time.time()
mse = np.power(y - y_hat, 2)
elapse_time = time.time() - st_time
st_time = time.time()
print(elapse_time)
mses = []
for i in range(n):
    mses.append(np.power(y[i] - y_hat[i], 2))
elapse_time = time.time() - st_time
print(elapse_time)
