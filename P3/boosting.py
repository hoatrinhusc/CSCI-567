import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		Ntump = len(self.betas)
		N = len(features)
		H = np.zeros(N,)

		for i in range(Ntump):
			Htemp = np.array([])
			hi = np.array(self.clfs_picked[i].predict(features))
			H += np.append(Htemp, self.betas[i] * hi)
		H = np.sign(H).astype(int).tolist() 
	
		return H
                
		########################################################
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"

                # Step 1: Initialization
		N = len(features)
		D = np.full((N),1/N)
		label_arr = np.array(labels)
		

                # Step 2: Loop over all weak classifiers
		for t in range(self.T):
			Error = np.array([])
                # Step 3: Find ht
			for clf in self.clfs:
				stump = np.array(clf.predict(features))
				bool_arr = [stump != label_arr]
				err_weight = float(np.dot(bool_arr,D))
				Error = np.append(Error, err_weight)
			ht = list(self.clfs)[np.argmin(Error)]
			self.clfs_picked.append(ht)

                # Step 4: Compute error
			error = np.amin(Error)

                # Step 5: Compute beta
			beta = np.log((1-error)/error)/2
			self.betas.append(beta)

                # Step 6: Update weight Dt
			for n in range(N):
				D[n] *= np.exp(-beta) if label_arr[n] == ht.predict(features)[n] else np.exp(beta)

                # Step 7: Normalize
			D /= np.sum(D)

                          
                ############################################################
                     
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	
