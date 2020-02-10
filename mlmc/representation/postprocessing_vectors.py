from sklearn.decomposition import PCA
import numpy as np

def postprocess_embedding(x, D=10):
    """Remove principal components and mean value to make embeddings more discernable
    Mu and Visanath (2018): https://arxiv.org/pdf/1702.01417.pdf"""
    no_mean = x-x.mean(0)[None,:]
    pca = PCA(n_components=D, svd_solver='full')
    pca.fit(no_mean)
    return no_mean - np.matmul(np.matmul(pca.components_,x.transpose()).transpose(), pca.components_)

