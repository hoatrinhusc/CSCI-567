import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			branches = np.array(branches)
			Ptot = np.sum(branches, axis = 0)
			nsamples = np.sum(Ptot)
			fraction = Ptot/nsamples
			
			P = branches/Ptot
			N,D = P.shape
			logP = np.zeros((N,D)) 
			
			for i in range(N):
				for j in range(D):
					if P[i,j] != 0:
						logP[i,j] = np.log2(P[i,j])
			PlogP = np.multiply(P, logP)
			entropy = np.sum(PlogP, axis = 0)
			Econd = np.sum(np.multiply(entropy, fraction))
			
			return Econd
			########################################################
			
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		# Pick one feature as root, extract all the values of all training examples for that feature
			values = np.array(self.features)[:,idx_dim]
		# Determine what branches of the root are
			clf_branch = np.unique(values)
		# Collect examples into a branch
			branches = []
			for branch in clf_branch:
				br_bool = [values == branch]

				# There might be labels 0, hence when we multiply with 
				# boolean variable, False, will create wrong results.
				# To avoid label 0, we add 1 to every label when we multiply.
				# This does not affect to counting of how many times a label appears.

				lbs_of_br = np.squeeze((np.array(self.labels)+1) * br_bool)
				# In each branch, devide examples based on their labels
				lst = lbs_of_br.tolist()
				lst = [i-1 for i in lst if i != 0]

				yn = np.unique(self.labels).tolist()
				count = [[x,lst.count(x)] for x in set(yn)]
				
				bi = []
				for ci in count:
					bi.append(ci[1])
				branches.append(bi)
			entropy = conditional_entropy(branches)
			
			# Compare
			if idx_dim == 0:
				entropy_min = entropy
			if entropy <= entropy_min:
				entropy_min = entropy
				self.dim_split = idx_dim
				self.feature_uniq_split = clf_branch.tolist()
		############################################################




		############################################################
		# TODO: split the node, add child nodes
		
		for branch in self.feature_uniq_split:
			Xchild = []
			ychild = []
			
			for i,j in enumerate(self.features):
				if j[self.dim_split] == branch:
					Xchild.append(j)
					ychild.append(self.labels[i])
			Xchild = Xchild[:self.dim_split] + Xchild[self_dim.split+1:]
						
			child = TreeNode(Xchild, ychild, self.num_cls)
			

			self.children.append(child)
		############################################################




		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



