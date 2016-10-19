import numpy as np
from sklearn.base import clone
from joblib import Parallel, delayed

class ClassImbalanceRedux:
	def __init__(self, clf, n_bags = 30):
		self.clf = clf
		self.n_bags = n_bags
	def fit(self, X, y, n_jobs = -1, seed = 93):
		# Fix seed for reproducibility of results
		np.random.seed(seed)
		# Check which class is imbalance
		counts = [np.sum(y == 0), np.sum(y == 1)]
		self.under = np.argmin(counts)
		self.under_count = counts[self.under]	
		self.fit_clfs = Parallel(n_jobs = n_jobs, verbose = 11)(delayed(self._fitBag)(X, y) for i in range(self.n_bags))

	def _fitBag(self, X, y):
		# Clean clf parameters
		clf = clone(self.clf)
		# Generate bootstrapped sample
		y_under_indexes = (y == self.under)
		y_under = y[y_under_indexes]
		y_over_indexes = np.random.choice(np.where(y != self.under)[0], self.under_count)
		y_over = y[y_over_indexes]
		X_bag = np.concatenate((X[y_under_indexes], X[y_over_indexes]), axis = 0)
		y_bag = np.concatenate((y_under, y_over), axis = 0)
		# Fit model in bag
		clf.fit(X_bag, y_bag)
		return clf

	def _predBag(self, clf, X_test):
		# Predict clf class
		self.pred_bag += 1
		print("Predicting for model in bag {}/{}".format(self.pred_bag, self.n_bags))
		return clf.predict(X_test)

	def predict(self, X_test, threshold = 0.5):
		self.pred_bag = 0
		self.y_hat = np.squeeze([self._predBag(clf, X_test) for clf in self.fit_clfs])
		prob = np.sum(self.y_hat, axis = 0)/self.y_hat.shape[0]
		return prob > threshold

	def predict_proba(self, X_test):
		self.pred_bag = 0
		self.y_hat = np.squeeze([self._predBag(clf, X_test) for clf in self.fit_clfs])
		prob = np.sum(self.y_hat, axis = 0)/self.y_hat.shape[0]
		return prob

		

			 
			
			
