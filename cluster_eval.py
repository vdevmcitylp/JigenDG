
import argparse

from sklearn import metrics
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import os.path as osp
import warnings
warnings.filterwarnings('ignore')
import pickle

# def load_data(pkl_file):
# 	with open(pkl_file, 'rb') as handle:
# 		data = pickle.load(handle)
# 	return data

# def shuffle(x, y):
# 	perm = np.random.permutation(len(y))
# 	x = x[perm]
# 	y = y[perm]
# 	return x,y


# def get_data(source, paths):
# 	features = np.array([])
# 	labels = np.array([])
# 	for sp in source:
# 		data = load_data(osp.join(paths[sp],'act_labels.pkl'))
# 		if features.any():
# 			features = np.append(features,data['features'], axis=0)
# 			labels = np.append(labels,data['labels'],axis=0)
# 		else:
# 			features = data['features']
# 			labels = data['labels']
# 	return features, labels

# class ClusteringData():

# 	def __init__(self, source, target, paths, split = 0, dg_setting = False):
# 		self.target = target
# 		self.source = source
# 		self.paths = paths
# 		self.dg_setting = dg_setting

# 		if dg_setting:
# 			self.get_dg_data()

# 		else:
# 			features, labels = self.get_data(source=False)
		
# 		if split:
# 			self.split_data(features, labels, split)
# 		else:
# 			self.source_features = features
# 			self.source_labels = labels
# 			self.target_features = features
# 			self.target_labels = labels

	  
# 	def split_data(self, features, labels, split):

# 		count = int(len(labels)*split)
# 		self.source_features = features[:count]
# 		self.source_labels = labels[:count]
# 		self.target_features = features[count:]
# 		self.target_labels = labels[count:]


# 	def get_dg_data(self):

# 		self.source_features, self.source_labels = self.get_data(source=True)
# 		self.target_features, self.target_labels = self.get_data(source=False)

# 	def get_data(self, source=True):
# 		features = np.array([])
# 		labels = np.array([])
# 		if source:
# 			domains = self.source
# 		else:
# 			domains = self.target
# 		for sp in domains:
# 			data = load_data(osp.join(paths[sp],'act_labels.pkl'))
# 			if features.any():
# 				features = np.append(features,data['features'], axis=0)
# 				labels = np.append(labels,data['labels'],axis=0)
# 			else:
# 				features = data['features']
# 				labels = data['labels']
# 		return features, labels

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

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


def cluster(data):

	kmeans = KMeans(n_clusters = 7, max_iter = 300).fit(data)
	return kmeans

def evaluate_clustering(model, data, labels):

	clusters = model.predict(data)
	acc_v, acc_i = calculate_acc(clusters, labels)
	nmi_v = calculate_nmi(clusters, labels)
	ari_v = calculate_ari(clusters, labels)
	print("Accuracy %f NMI %f ARI %f" % (acc_v * 100, nmi_v, ari_v))
	return acc_v, nmi_v, ari_v


def load_source_data(logs_folder, source_domains):

	features = np.array([])
	labels = np.array([])

	for src in source_domains:

		pkl_file = osp.join(logs_folder, "act_labels_{}.pkl".format(src))

		with open(pkl_file, 'rb') as handle:
			data = pickle.load(handle)		

		if features.any():
			features = np.append(features, data["features"], axis = 0)
			labels = np.append(labels, data["labels"], axis = 0)
		else:
			features = data["features"]
			labels = data["labels"]

	return features, labels


def load_target_data(logs_folder, target_domain):

	pkl_file = osp.join(logs_folder, "act_labels_{}.pkl".format(target_domain))

	with open(pkl_file, 'rb') as handle:
		data = pickle.load(handle)

	features = data['features']
	labels = data['labels']

	return features, labels

def get_args():

	parser = argparse.ArgumentParser()

	parser.add_argument("--source", help = "Source Domains", nargs = '+')
	parser.add_argument("--target", help = "Target Domain")

	parser.add_argument("--source_clustering", action = "store_true", help = "If true to cluster on source domains \
		and evaluate on target domain")

	parser.add_argument("--target_clustering", action = "store_true", help = "Cluster on target domain images")

	parser.add_argument("--split", type = float, help = "Proportion of target data used for clustering")

	parser.add_argument("--exp_type", choices = ["vanilla-jigsaw", "stylized-jigsaw"])

	parser.add_argument("--run_id", type = str, help = "Run ID of the experiment, act_label.pkl \
        will be loaded from args.exp_type/s1-s2-s3_to_s4/args.run_id")

	parser.add_argument("--seed", type = int, choices = [1, 2, 3])

	args = parser.parse_args()

	return args

def main(args, logs_root):

	exp_folder = "%s/%s_to_%s/%s/" % (args.exp_type, 
            "-".join(sorted(args.source)), args.target, args.run_id)

	logs_folder = osp.join(logs_root, exp_folder)
	
	target_features, target_labels = load_target_data(logs_folder, args.target)
	
	if args.source_clustering:
		
		print("Clustering on source domains ...")
		assert args.target_clustering == False

		source_features, source_labels = load_source_data(logs_folder, args.source)

		model = cluster(source_features)

	if args.target_clustering:
		
		print("Clustering on target domain images ...")
		assert args.source_clustering == False

		# print("Using {}% of target domain images for clustering".format(args.split * 100))

		model = cluster(target_features)
	
	print("Results for {}".format(args.target))
	evaluate_clustering(model, target_features, target_labels)


if __name__ == "__main__":
  
	args = get_args()

	logs_root = "/DATA1/vaasudev_narayanan/repositories/JigenDG/logs"

	set_seed(args.seed)
	
	main(args, logs_root)
