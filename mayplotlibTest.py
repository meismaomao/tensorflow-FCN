import numpy as np
import matplotlib.pyplot as plt


check = np.zeros((9, 9))
check[::2, 1::2] = 1
check[1::2, ::2] = 1
plt.imshow(check, cmap="gray", interpolation='nearest')
plt.show()

images = np.array([check for i in range(3)])
print(check)
print(images)