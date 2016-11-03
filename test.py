import numpy as np
import math
from scipy.spatial import distance

if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    a[[0],:] = [0,0]
    print a
