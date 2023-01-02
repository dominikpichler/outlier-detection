import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.fft import fft, fftfreq
from numpy.fft import fft, ifft
import numpy as np




mu,sigma,n = 0.,1.,1000

def normal(x,mu,sigma):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )

x = np.random.normal(mu,sigma,n)
y = normal(x,mu,sigma)

df_test = pd.DataFrame()

df_test["x"] = x

plt.scatter(x,y)
plt.show()
#x = df_raw.iloc[:,0]
#y = df_raw.iloc[:,1]


mymodel = np.poly1d(np.polyfit(x, y, 4))


myline = np.linspace(-7, -2, 100)
print(r2_score(y, mymodel(x)))



df_test = pd.DataFrame(columns=["X","Y_ori","Y_aprox"])
df_test["X"] = x
df_test["Y_ori"] = y
df_test["Y_aprox"] = mymodel(x)
df_test["Diff"] = df_test["Y_ori"] - df_test["Y_aprox"]

plt.scatter(x,y)
plt.scatter(x, mymodel(x))

plt.show()