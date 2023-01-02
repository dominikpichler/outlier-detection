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
from sklearn.metrics import mean_absolute_percentage_error
from scipy.fft import fft, fftfreq
from numpy.fft import fft, ifft
import statsmodels.api as sm
import math

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


#TODO: Folderstructure for results
#TODO: Automatic Labeling @ Plots
#TODO: Multi-Dim-Cluster!!!
#TODO: Opt: Sampling Frequency in Fourier
#


#====== Update Cafer:
# TODO: Fourier for each Cluster


def load_rawData(path:str) -> pd.DataFrame:

    df_rawData = pd.read_csv(path)

    #TODO: Tests for data quality

    return df_rawData

def applyFeature_extraction_lin(rawData:pd.DataFrame,label:str) -> pd.DataFrame:

    X = rawData.values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_scaled)
    df_principal = pd.DataFrame(data=principalComponents
                               , columns=['PC_1', 'PC_2'])

    # plotting a scatter plot
    print("Scatter Plot:  ")
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(df_principal["PC_1"], df_principal["PC_2"])
    pcaResult = "results_plots/" + "PCA_for_" + label + ".png"
    pcaResultTeX = "results_plots/" + "PCA_for_" + label + ".pgf"
    plt.title("Result PCA")
    plt.savefig(pcaResult)

    fig.set_size_inches(w=6.5, h=3.5)
    plt.savefig(pcaResultTeX)
    #plt.show()

    list_explaind_vd = pca.explained_variance_ratio_.tolist()

    expl_variance = 0
    for pc in list_explaind_vd:
        expl_variance += pc

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print("PC1 and PC2 cumulatively explain " + str(round(pc * 100,2)) + " % of the total variance")
    if pc < 0.8:
        print("Unable to find helpful PC's in 2D. Please use a different Method")
    else:
        print("Found helpful PC's in 2D")


    #Store Data

    label_PCA_Result_data = "results_data/" + "PCA_for_" + label + ".csv"
    df_principal.to_csv(label_PCA_Result_data)


    return df_principal

def applyFeature_extraction_nonLin(rawData:pd.DataFrame,label:str) -> pd.DataFrame:
    #TODO: OPTIONAL: add dimensionality check to it doesn't get to big!

    X = rawData.values

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)


    x_1 = pd.Series(tsne_results[:, 0])
    x_2 = pd.Series(tsne_results[:, 1])

    df_tsne_result = pd.DataFrame(columns=['C1', 'C2'])

    df_tsne_result['C1'] = x_1
    df_tsne_result['C2'] = x_2
    fig = plt.figure(figsize=(10, 7))

    plt.figure(figsize=(16, 10))
    plt.title("Result t-SNE")
    sns.scatterplot(
        x="C1", y="C2",
        palette=sns.color_palette("hls", 10),
        data=df_tsne_result,
        legend="full",
        alpha=0.3
    )

    tsneResult = "results_plots/" + "t-SNE_for" + label + ".png"
    tsneResultTeX = "results_plots/" + "t-SNE_for" + label + ".pgf"
    plt.savefig(tsneResult)
    fig.set_size_inches(w=6.5, h=3.5)
    plt.savefig(tsneResultTeX)
    plt.show()

    label_tsne_Result_data = "results_data/" + "t-SNE_for_" + label + ".csv"
    df_tsne_result.to_csv(label_tsne_Result_data)
    return df_tsne_result


#2D
def applyFourierTransform(rawData:pd.DataFrame,label) -> pd.DataFrame:


    df_rawData = rawData.copy()
    df_rawData.columns = ['x', 'y']
    df_rawData = df_rawData.sort_values('x')


    x = df_rawData.iloc[:, 0]
    y = df_rawData.iloc[:, 1]



    lst = []
    for i in range(len(x)):
        lst.append(i)


    n_polyDeg = 2
    accurateModel_found = False


    while(accurateModel_found == False):

        mymodel = np.poly1d(np.polyfit(x, y,n_polyDeg ))
        r_2_score = r2_score(y, mymodel(x))
        mape = mean_absolute_percentage_error(y,mymodel(x))

        if(r_2_score > 0.95 and mape < 0.1):
            accurateModel_found = True

        if n_polyDeg > 10:

            print(" ***** Caution @ [FOURIER]: No sufficiently approximating model for the observations found ***** ")
            return pd.DataFrame()


            break

        n_polyDeg += 1

    print("----- Finished Modelling -----")
    print("Sufficiently approximating model found")
    print("-- Parameters of Model --")
    print("Mape: " + str(mape))
    print("R_2: " + str(r_2_score))
    print("f(x) is a polynom of order " + str(n_polyDeg))
    print("The coeffs for the polynom are: " + str(mymodel.c))






    print("-------------------------------")

    print("----- Starting the Fourier Transformation -----")


    fig = plt.figure(figsize=(10, 7))

    plt.scatter(x,mymodel(x),label = "Original Observations")
    plt.plot(x,y, "r-", label = "Model")
    plt.savefig("results_plots/Modelling_for_DFT" + label + ".png")
    plt.legend()
    plt.ylabel("Dim 1")
    plt.xlabel("Dim ")
    plt.title("Model Fit")

    fig.set_size_inches(w=6.5, h=3.5)
    plt.savefig("results_plots/Modelling_for_DFT" + label + ".pgf")
    plt.show()


    # ======= FOURIER
    #TODO: Generalize

    sr = 600
    ts = 1.0 / sr
    t = np.arange(min(x), max(x), ts)
    x = mymodel(t)

    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N / sr
    freq = n / T

    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(X)*ts, 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 10)
    plt.title("Frequency Domain")

    plt.subplot(122)
    plt.plot(t, ifft(X), 'r')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    labelFourier_plot = "results_plots/" + "FourierSeries_of_" + label + ".png"
    plt.savefig(labelFourier_plot)
    plt.title("Space Domain")

    plt.show()

    df_coeffs_FourierSeries = pd.DataFrame({'Frequency': freq, 'FFT_Amplitude': np.abs(X)*ts},
                                           columns=['Frequency', 'FFT_Amplitude'])




    #---------- CUT-OFF / Qualitiy Testing:

    #Nyquist frequency cut off
    n_cut = len(df_coeffs_FourierSeries)/2
    # pick relevant ones
    df_coeffs_FourierSeries = df_coeffs_FourierSeries.iloc[:-int(n_cut), :]

    df_coeffs_F_sorted = df_coeffs_FourierSeries.sort_values("FFT_Amplitude",ascending=False)

    i = 0
    n_FComp = 3 #TODO: define automatic cut off! -> R_2*(1-MAPE)?
    y_FourierSeries = 0

    list_FSComponents = []
    #df_rawData["Cluster"] = n_FComp

    plt.figure()
    plt.plot(t,x,label= "Original Observations")
    df_fourierClustered = df_rawData.copy()
    df_fourierClustered["Cluster"] = n_FComp


    while i < n_FComp:
        list_xGroupMemebers = []

        comp_i = df_coeffs_F_sorted.iloc[i, 1] * np.sin(2 * np.pi * df_coeffs_F_sorted.iloc[i, 0] * t)
        if df_coeffs_F_sorted.iloc[i, 0] == 0:
            comp_i += df_coeffs_F_sorted.iloc[i, 1]
            i += 1
            continue

        else:
            #binary one
            y_max = max(comp_i)
            j = 0

            #find local maximas
            while j < len(comp_i):
                if math.isclose(comp_i[j],max(comp_i), rel_tol=y_max*0.05) :
                    #df_rawData.loc[j,"Cluster"] = i
                    list_xGroupMemebers.append(t[j])
                j += 1
                y_FourierSeries += comp_i

            li = np.linspace(5, 5, len(list_xGroupMemebers))
            clusterName = "Cluster_" + str(i)
            plt.scatter(list_xGroupMemebers, li,label=clusterName)
            componentenName = "Component_" + str(i)
            plt.plot(t, 2 * comp_i, label=componentenName)


            # Update in Clustering Version


            m = 0
            while m < len(df_fourierClustered):

                for element in list_xGroupMemebers:

                    if math.isclose(df_fourierClustered.loc[m,"x"],element, rel_tol=abs(element*0.001)):
                        df_fourierClustered.loc[m, "Cluster"] = i
                        #print( str(m) + "Element: " + str(element) + " Is close! to " + str(df_fourierClustered.loc[m,"x"]) )


                m += 1

            i += 1



    plt.ylabel('Dim 1')
    plt.xlabel('Dim 2')
    plt.legend()
    plt.title("Fourier Clustering")
    plt.show()




    # ====== Plot clustered result:
    fourier_clusters  = df_fourierClustered['Cluster'].unique()


    plt.figure()
    for cluster in fourier_clusters:
        df_tmpPlot  = df_fourierClustered[df_fourierClustered["Cluster"] == cluster]
        plt.scatter(df_tmpPlot["x"],df_tmpPlot["y"],label = ("Cluster " + str(cluster)))
    plt.legend()
    plt.ylabel("Dim 1")
    plt.xlabel("Dim 2")
    plt.title("Fourier Clustering")
    plt.show()




    #plt.plot(t,2*y_FourierSeries) #TODO: warum faktor 2?


    label_Fourier_Result_data = "results_data/" + "fourierSeries_for_" + label + ".csv"
    label_FourierClustering = "results_data/" + "fourierClustering_for_" + label + ".csv"
    df_coeffs_FourierSeries.to_csv(label_Fourier_Result_data)
    df_fourierClustered.to_csv(label_FourierClustering)

    print("Successfully calculated the Fourier Transformation")
    print("Results will be stored in 'results_data/fourierResults.csv' ")
    print("-------------------------------")




    return df_coeffs_FourierSeries



#2D
def clusterData(rawData:pd.DataFrame,label:str) -> pd.DataFrame:


    #TODO: implement for n > 2
    n_max = 2
    n_dim = len(rawData.columns)
    if n_dim > n_max:
        raise ValueError("Number of Dimensions of handed Data is too high, please apply DRA's first")

    #To numpy array
    X = rawData.values

    # ----------- finding the optimal number Clusters K using the Silhouette Method --------
    sil = []
    kmax = 10
    k_start = 2

    for k in range(k_start, kmax + 1):
        kmeans_sil = KMeans(n_clusters=k).fit(X)
        labels = kmeans_sil.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))

    k_algorithmic = k_start + sil.index(max(sil))


    #

    kmeans = KMeans(
        n_clusters=k_algorithmic,
        init ="random",
        random_state=0,
        n_init= 15,
        max_iter= 100,
    )


    cluster_km = kmeans.fit_predict(X)
    df_result_cluster = df_rawData.copy()
    df_result_cluster['Cluster']=pd.Series(cluster_km)

    # ------- Regression in clusters -------- :
    results_C1 = sm.OLS(X[cluster_km == 0, 1], sm.add_constant(X[cluster_km == 0, 0])).fit()
    results_C2 = sm.OLS(X[cluster_km == 1, 1], sm.add_constant(X[cluster_km == 1, 0])).fit()
    print(results_C1.summary())
    # ------- Ploting the clusters -------- :

    #TODO: Loop for n != 2 cluster
    fig = plt.figure(figsize=(6.5, 4))

    plt.xlabel("Dim1")
    plt.ylabel("Dim2")

    # plot the 2 clusters
    plt.scatter(
        X[cluster_km == 0, 0], X[cluster_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )
    plt.plot(X[cluster_km == 0, 0], X[cluster_km == 0, 0] * results_C1.params[1] + results_C1.params[0], label="Linear regression - Cluster 1")
    plt.plot(X[cluster_km == 1, 0], X[cluster_km == 1, 0] * results_C2.params[1] + results_C2.params[0], label="Linear regression - Cluster 2")

    plt.scatter(
        X[cluster_km == 1, 0], X[cluster_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )



    # Centroids
    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.title(label)
    plt.grid()
    clusteringResult = "results_plots/"  + "Clustering_for_" + label + ".png"
    clusteringResultText = "results_plots/"  + "Clustering_for_" + label + ".pgf"
    plt.savefig(clusteringResult)

    #fig.set_size_inches(w=6.5, h=3.5)
    plt.savefig(clusteringResultText)
    #plt.show()



    #Elbow method for finding the optimal K as additional check for user.
    #TODO: Implement Distirbution Testing as another criteron?
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)

    # plt.plot(range(1, 11), distortions, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Distortion')
    # elbowName = "results_plots/"  + "Elbow_for_" + label + ".png"
    # plt.savefig(elbowName)
    #
    # label_cluster_Result_data = "results_data/" + "Clustering_for_" + label + ".csv"
    # df_result_cluster.to_csv(label_cluster_Result_data)

    label_kMeansClustering = "results_data/" + "k-Means_Clustering_for_" + label + ".csv"
    df_result_cluster.to_csv(label_kMeansClustering)
    return df_result_cluster


def autoRun(rawData:pd.DataFrame,label) -> dict:
    df_rawData = rawData
    dict_solutions = {}
    dict_result_clustering = {}
    dict_result_fourier = {}


# ------ 0: Data Prep -------
    # Non Transformation:
    df_sol_nonTrans = df_rawData.copy()

    # Linear Transformation:
    if len(df_rawData.columns) > 2:
        print("----- Running LinTrans (PCA)")
        df_sol_linTrans = applyFeature_extraction_lin(df_rawData,label)
    else:
        df_sol_linTrans = df_rawData
    # Non-Linear Transformation
    if len(df_rawData.columns) > 2:
        print("----- Running NonLinTrans (t-SNE)")

        df_sol_nonLinTrans = applyFeature_extraction_nonLin(df_rawData,label)

    else:
        df_sol_nonLinTrans = df_rawData



# ------ 1: Clustering -------

    # Non Transformation:
    if len(df_rawData.columns) == 2:
        print("----- Running Clustering for Result of NonTrans")
        label_nt = label + "[NonTrans]"
        df_cluster_nonTrans = clusterData(df_sol_nonTrans,label_nt)
        dict_result_clustering["nonTrans"] = df_cluster_nonTrans

    # Linear Transformation:
    if len(df_rawData.columns) > 2:
        print("----- Running Clustering for Result of LinTrans")
        label_lt = label + "[LinTrans]"
        df_cluster_linTrans = clusterData(df_sol_linTrans,label_lt)
        dict_result_clustering["linTrans"] = df_cluster_linTrans

    # Non-Linear Transformation
    if len(df_rawData.columns) > 2:
        print("----- Running Clustering for Result of NonLinTrans")
        label_nlt = label + "[NonLinTrans]"
        df_cluster_nonLinTrans = clusterData(df_sol_nonLinTrans,label_nlt)
        dict_result_clustering["nonLinTrans"] = df_cluster_nonLinTrans

    if len(df_rawData.columns == 3):
        #variational clustering along all axis:

        df_rawData_dim1_2 = pd.DataFrame(columns=["1", "2"])
        df_rawData_dim1_2["1"] = df_rawData.iloc[:, 0]
        df_rawData_dim1_2["2"] = df_rawData.iloc[:, 1]
        label12 = label + "clustering_dim12"
        clusterData(df_rawData_dim1_2,label12)
        applyFourierTransform(df_rawData_dim1_2,label12)

        df_rawData_dim2_3 = pd.DataFrame(columns=["2", "3"])
        df_rawData_dim2_3["2"] = df_rawData.iloc[:, 1]
        df_rawData_dim2_3["3"] = df_rawData.iloc[:, 2]
        label23 = label + "clustering_dim23"
        clusterData(df_rawData_dim2_3, label23)
        applyFourierTransform(df_rawData_dim2_3, label23)

        df_rawData_dim1_3 = pd.DataFrame(columns=["1", "3"])
        df_rawData_dim1_3["1"] = df_rawData.iloc[:, 0]
        df_rawData_dim1_3["3"] = df_rawData.iloc[:, 2]
        label13 = label + "clustering_dim13"
        clusterData(df_rawData_dim1_3,label13)
        applyFourierTransform(df_rawData_dim1_3, label13)

    dict_solutions["Clustering"] = dict_result_clustering


# ------ 2: Fourier -------


    # Non Transformation:
    if(len(df_sol_nonTrans.columns) == 2):
        print("----- Running Fourier for Result NonTrans")
        label_nt = label + "[NonTrans]"
        df_fourier_nonTrans = applyFourierTransform(df_sol_nonTrans,label_nt)
        dict_result_fourier["nonTrans"] = df_fourier_nonTrans

    # Linear Transformation:
    if len(df_rawData.columns) > 2:
        print("----- Running Fourier for Result of LinTrans")

        label_lt = label + "[LinTrans]"

        df_fourier_linTrans = applyFourierTransform(df_sol_linTrans,label_lt)
        dict_result_fourier["linTrans"] = df_fourier_linTrans

    # Non-Linear Transformation
        print("----- Running Fourier for Result of NonLinTrans")

        label_nlt = label + "[NonLinTrans]"
        df_fourier_nonLinTrans = applyFourierTransform(df_sol_nonLinTrans,label_nlt)
        dict_result_fourier["nonLinTrans"] = df_fourier_nonLinTrans

    dict_solutions["Fourier"] = dict_result_fourier


# ------ 3: Solution -------


    #
    # #TODO: Solve saving solution
    # with open('results/result_full.json', 'w') as fp:
    #     json.dump(dict_solutions, fp)

    return dict_solutions


if __name__ == '__main__':
    #df_rawData = load_rawData("dataOutliersClean.csv")
    df_rawData = load_rawData("old_data/dataOutliersClean.csv")
    #mpl.use('macosx')

    df_rawData = df_rawData.iloc[:, 1:]

    #3D Plot
    # x = df_rawData.iloc[:,0]
    # y = df_rawData.iloc[:,1]
    # z = df_rawData.iloc[:,2]
    # #clu = df_rawData.iloc[:,3]
    #
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    #
    # # Creating plot
    # ax.scatter3D(x, y, z)#, c= clu > 0,cmap = 'coolwarm')
    # #ax.scatter3D(x, y, z, cmap = 'coolwarm')
    # ax.set_xlabel('Depression')
    # ax.set_ylabel('Neuroticism')
    # ax.set_zlabel('Life Stress')
    # plt.title("3D Plot of raw data")
    # plt.savefig("3D-Plot-allOutliers_kMeans.png")
    # fig.set_size_inches(w=6.5, h=3.5)
    import tikzplotlib

    #plt.savefig('histogram.pgf')
    #tikzplotlib.save("test.tex")

    #plt.show()

    autoRun(df_rawData,"_outliers_")
    #applyFourierTransform(df_rawData,"outliers")


    print("Finished")



