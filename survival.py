import pandas as pd
import numpy as np
import math

class RandomSurvivalForest():

	def __init__(self, n_trees = 10, max_features = 2, max_depth = 5, min_samples_split = 2, split = "auto"):
		self.n_trees = n_trees
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.split = split
		self.max_features = max_features

	def logrank(self, x, feature):
		c = x[feature].median()
		if x[x[feature] <= c].shape[0] < self.min_samples_split or x[x[feature] > c].shape[0] <self.min_samples_split:
			return 0
		t = list(set(x["time"]))
		get_time = {t[i]:i for i in range(len(t))}
		N = len(t)
		y = np.zeros((3,N))
		d = np.zeros((3,N))
		feature_inf = x[x[feature] <= c]
		feature_sup = x[x[feature] > c]
		count_sup = np.zeros((N,1))
		count_inf = np.zeros((N,1))
                for _, r in feature_sup.iterrows():
			t_idx = get_time[r["time"]]
                        count_sup[t_idx] = count_sup[t_idx] + 1
			if r["event"]:
				d[2][t_idx] = d[2][t_idx] + 1
                for _, r in feature_inf.iterrows():
			t_idx = get_time[r["time"]]
                        count_inf[t_idx] = count_inf[t_idx] + 1
			if r["event"]:
				d[1][t_idx] = d[1][t_idx] + 1
		nb_inf = feature_inf.shape[0]
		nb_sup = feature_sup.shape[0]
		for i in range(N):
			y[1][i] = nb_inf
			y[2][i] = nb_sup
			y[0][i] = y[1][i] + y[2][i]
			d[0][i] = d[1][i] + d[2][i]
			nb_inf = nb_inf - count_inf[i]
			nb_sup = nb_sup - count_sup[i]
		num = 0
		den = 0
		for i in range(N):
			if y[0][i] > 0:
				num = num + d[1][i] - y[1][i] * d[0][i] / float(y[0][i])
			if y[0][i] > 1:
				den = den + (y[1][i] / float(y[0][i])) * y[2][i] * ((y[0][i] - d[0][i]) / (y[0][i] - 1)) * d[0][i]
		L = num / math.sqrt(den)
		return abs(L)

	def find_best_feature(self, x):
		split_func = {"auto" : self.logrank}
		features = [f for f in x.columns if f not in ["time", "event"]]
		information_gains = [split_func[self.split](x, feature) for feature in features]
		highest_ig = max(information_gains)
		if highest_ig == 0:
			return None
		else:
			return features[information_gains.index(highest_ig)]

	def compute_leaf(self, x, tree):
		count = {}
                for _, r in x.iterrows():
                        count.setdefault((r["time"], 0), 0)
                        count.setdefault((r["time"], 1), 0)
                        count[(r["time"], r["event"])] = count[(r["time"], r["event"])] + 1
		t = list(set([c[0] for c in count]))
                t.sort()
		total = x.shape[0]
		tree["count"] = count
		tree["t"] = t
		tree["total"] = total

	def build(self, x, tree, depth):
		unique_targets = pd.unique(x["time"])

		if len(unique_targets) == 1 or depth == self.max_depth:
			self.compute_leaf(x, tree)
			return
	
		best_feature = self.find_best_feature(x)

		if best_feature == None:
                        self.compute_leaf(x, tree)
                        return

		feature_median = x[best_feature].median()
	
		tree["feature"] = best_feature
		tree["median"] = feature_median

		left_split_x = x[x[best_feature] <= feature_median]
		right_split_x = x[x[best_feature] > feature_median]
		split_dict = [["left", left_split_x], ["right", right_split_x]]
	
		for name, split_x in split_dict:
			tree[name] = {}
			self.build(split_x, tree[name], depth + 1)

	def fit(self, x, event):
		self.trees = [{} for i in range(self.n_trees)]
		event.columns = ["time", "event"]
		features = list(x.columns)
		x = pd.concat((x,event), axis=1)
		x = x.sort_values(by="time")
		x.index = range(x.shape[0])
		for i in range(self.n_trees):
			sampled_x = x.sample(frac = 1, replace = True)
			sampled_x.index = range(sampled_x.shape[0])
			sampled_features = list(np.random.permutation(features))[:self.max_features] + ["time","event"]
			self.build(sampled_x[sampled_features], self.trees[i], 0)

	def compute_chf(self, row, tree):
		count = tree["count"]
		t = tree["t"]
		total = tree["total"]
		h = 1
		survivors = float(total)
		for ti in t:
			if ti <= row["time"]:
				h = h * (1 - count[(ti,1)] / survivors)
			survivors = survivors - count[(ti,1)] - count[(ti,0)]
		return h
	
	def predict_row(self, tree, row):
		if "count" in tree:
			return self.compute_chf(row, tree)
	
		if row[tree["feature"]] > tree["median"]:
			return self.predict_row(tree["right"], row)
		else:
			return self.predict_row(tree["left"], row)

	def predict_proba(self, x):
		assert "time" not in x.columns[:-1]
		x.columns = list(x.columns)[:-1] + ["time"]
		compute_trees = [x.apply(lambda u: self.predict_row(self.trees[i], u), axis=1) for i in range(self.n_trees)]
		return sum(compute_trees) / self.n_trees

	def print_with_depth(self, string, depth):
		print("{0}{1}".format("    " * depth, string))

	def print_tree(self, tree, depth = 0):
    		if "count" in tree:
        		self.print_with_depth(tree["t"], depth)
        		return
    		self.print_with_depth("{0} > {1}".format(tree["feature"], tree["median"]), depth)
		self.print_tree(tree["left"], depth + 1)
		self.print_tree(tree["right"], depth + 1)

	def draw(self):
		for i in range(len(self.trees)):
			print "==========================================\nTree ", i
			self.print_tree(self.trees[i])
