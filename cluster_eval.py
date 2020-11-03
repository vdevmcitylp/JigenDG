from sklearn import metrics
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import os.path as osp
import warnings
warnings.filterwarnings('ignore')
import pickle

def acc(ypred, y, return_idx=False):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.
    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.
    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    #
    # indices = linear_sum_assignment(C)
    # row = indices[:][:, 0]
    # col = indices[:][:, 1]
    row, col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    if return_idx:
        return 1.0 * count / len(y), row, col
    else:
        return 1.0 * count / len(y)


def calculate_acc(y_pred, y_true):
    Y_pred = y_pred
    Y = y_true
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind


def calculate_nmi(predict_labels, true_labels):
    # NMI
    nmi = metrics.normalized_mutual_info_score(true_labels, predict_labels, average_method='geometric')
    return nmi


def calculate_ari(predict_labels, true_labels):
    # ARI
    ari = metrics.adjusted_rand_score(true_labels, predict_labels)
    return ari

def load_data(pkl_file):
    with open(pkl_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def cluster(data):

    kmeans = KMeans(n_clusters=7,max_iter=300, random_state=0).fit(data)
    return kmeans

def evaluate_clustering(model, data, labels):

    clusters = model.predict(data)
    acc_v, acc_i = calculate_acc(clusters, labels)
    nmi_v = calculate_nmi(clusters, labels)
    ari_v = calculate_ari(clusters, labels)
    print("Accuracy %f NMI %f ARI %f" % (acc_v, nmi_v, ari_v))
    return acc_v, nmi_v, ari_v

def shuffle(x, y):
    perm = np.random.permutation(len(y))
    x = x[perm]
    y = y[perm]
    return x,y


def get_data(source, paths):
    features = np.array([])
    labels = np.array([])
    for sp in source:
      data = load_data(osp.join(paths[sp],'act_labels.pkl'))
      if features.any():
        features = np.append(features,data['features'], axis=0)
        labels = np.append(labels,data['labels'],axis=0)
      else:
        features = data['features']
        labels = data['labels']
    return features, labels

class ClusteringData():

    def __init__(self, source, target, paths, split=0, dg_setting=False):
      self.target = target
      self.source = source
      self.paths = paths
      self.dg_setting = dg_setting

      if dg_setting:
        self.get_dg_data()
      
      else:
        features, labels = self.get_data(source=False)
        if split:
          self.split_data(features, labels, split)
        else:
          self.source_features = features
          self.source_labels = labels
          self.target_features = features
          self.target_labels = labels

      
    def split_data(self, features, labels, split):

        count = int(len(labels)*split)
        self.source_features = features[:count]
        self.source_labels = labels[:count]
        self.target_features = features[count:]
        self.target_labels = labels[count:]


    def get_dg_data(self):

      self.source_features, self.source_labels = self.get_data(source=True)
      self.target_features, self.target_labels = self.get_data(source=False)

    def get_data(self, source=True):
      features = np.array([])
      labels = np.array([])
      if source:
        domains = self.source
      else:
        domains = self.target
      for sp in domains:
          data = load_data(osp.join(paths[sp],'act_labels.pkl'))
          if features.any():
            features = np.append(features,data['features'], axis=0)
            labels = np.append(labels,data['labels'],axis=0)
          else:
            features = data['features']
            labels = data['labels']
      return features, labels


if __name__ == "__main__":
  
    source = ['art', 'photo','sketch']
    target = [ 'cartoon']
    paths = {}
    paths['photo'] = '/content/drive/My Drive/Codes/JigenDG/logs/photo_target_stylizedjigsaw/art-cartoon-sketch_to_photo'
    paths['art'] = '/content/drive/My Drive/Codes/JigenDG/logs/art_target_stylizedjigsaw/cartoon-photo-sketch_to_art'
    paths['cartoon'] = '/content/drive/My Drive/Codes/JigenDG/logs/cartoon_target_stylizedjigsaw/art-photo-sketch_to_cartoon'
    paths['sketch'] = '/content/drive/My Drive/Codes/JigenDG/logs/sketch_target_stylizedjigsaw/art-cartoon-photo_to_sketch'
    dg_setting = True
    #Use dg_setting=True to cluster on source domains and evaluate on target domain
    #Use dg_setting=False, split = 0 to cluster and evaluate on target 
    #Use dg_setting=False, split = 0.6 to cluster on 50% target data and evaluate on 40% target data
    dataset = ClusteringData(source, target, paths, split=0.6, dg_setting=True)
  

   
    print('Source features: {}'.format(dataset.source_features.shape))
    model = cluster(dataset.source_features)
    print('Target features: {}'.format(dataset.target_features.shape))
    evaluate_clustering(model, dataset.target_features, dataset.target_labels)


    
    


