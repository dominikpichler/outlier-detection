from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
R = np.array(R)
nmf = NMF()
W = nmf.fit_transform(R);
H = nmf.components_;
nR = np.dot(W,H)
print(W)
print("_____________________")
print(H)
print("_____________________")
print(nR)