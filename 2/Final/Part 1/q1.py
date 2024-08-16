import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

# for recalculation of clusters 
def cluster_recal(X, Y_pred, centroids, k):
    # Initiate empty clusters
    clusters = {}
    # Set the range for value of k (number of centroids)
    for i in range(k):
        clusters[i] = []
    for data in X:
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(data - centroids[j]))
        # Append the cluster of data to the dictionary
        clusters[euc_dist.index(min(euc_dist))].append(data)
        centroid = int((np.where(euc_dist[:] == min(euc_dist))[0]))
        Y_pred.append(centroid)
    return clusters
 
def centroids_recal(centroids, clusters, k):
    """ Recalculates the centroid position based on the plot """
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids

if __name__ == '__main__':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data"


    labels = ['ID' + str(i+1) for i in range(57)]

    df = pd.read_csv(url, names=labels)
    
    features = ['ID' + str(i+2) for i in range(56)]
    x = df.loc[:, features].values

    # filling '?' in data with the mean of that column
    avg1 = 0
    avg2 = 0
    count = 1
    for i in x[:,3]:
        if i != '?':
            avg1 = ((count-1)*avg1 + int(i))/count
    count = 1
    for i in x[:,37]:
        if i != '?':
            avg2 = ((count-1)*avg2 + int(i))/count

    x[0][3] = avg1
    x[14][3] = avg1
    x[18][3] = avg1
    x[20][3] = avg1
    x[25][37] = avg2
   
    y = df.loc[:,[labels[0]]].values
    x = MinMaxScaler().fit_transform(x)
    Y = df[labels[0]]

    """ Just taken 2 components for plotting """
     
   
    pca_cal_var = PCA()
    pc = pca_cal_var.fit_transform(x)
    cov_mat = pca_cal_var.get_covariance()
    relative_cov = pca_cal_var.explained_variance_ratio_
    print("Co-Variance_for_plot= ",np.sum(relative_cov[:2])/np.sum(relative_cov))


    pca_for_plot = PCA(n_components=2)
    principalComponents = pca_for_plot.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc 1', 'pc 2'])
    
    finalDf = pd.concat([principalDf, df[[labels[0]]]], axis = 1)

    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('pc 1', fontsize = 15)
    ax.set_ylabel('pc 2', fontsize = 15)

    ax.set_title('2 component PCA', fontsize = 20)
    targets = [1,2,3]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[labels[0]] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc 1']
                , finalDf.loc[indicesToKeep, 'pc 2']
                 , c = color
                 , s = 50)
        ax.legend(targets)
        ax.grid()
    plt.show()


    # This is the original extracted pca with greater than 0.95 covariance
     
    pca_original = PCA()

    X_pca = pca_original.fit_transform(x)
    cov_mat = pca_original.get_covariance()
    relative_cov = pca_original.explained_variance_ratio_
    print("Co-Variance_taken = ",np.sum(relative_cov[:21])/np.sum(relative_cov))
    # print(X_pca)
    
    # applying k-means and plotting k vs NMI
    NMI = []
    for k in range(2,9):
        prev_centroids = {}
        clusters = {}
        for i in range(k):
            clusters[i] = []

        for i in range(k):
            prev_centroids[i] = X_pca[i]
        
        for data in X_pca:
            euc_dist = []
            for j in range(k):
                euc_dist.append(np.linalg.norm(data - prev_centroids[j]))
            clusters[euc_dist.index(min(euc_dist))].append(data)

    # recalculating centroids and clusters until no change in centroid

        check = True
        while (check):
            centroids = centroids_recal(X_pca, clusters, k)

            if np.array_equal(prev_centroids, centroids):
                check = False
            Y_pred = []
            clusters = cluster_recal(X_pca, Y_pred, centroids, k)
            prev_centroids = centroids
        NMI.append(nmi(Y,Y_pred))
    

    k = np.arange(2,9,dtype = int)

    fig2 = plt.figure(figsize = (6,6))
    ax = fig2.add_subplot(1,1,1) 
    ax.set_xlabel('k', fontsize = 15)
    ax.set_ylabel('NMI', fontsize = 15)
    ax.set_title('k vs NMI', fontsize = 20)

    plt.plot(k, NMI)
    plt.show()

    max_p = NMI.index(max(NMI)) + 2
    print("The maximum value of NMI occurs at k = ",max_p)