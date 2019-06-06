# Code based on https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Modified from 2 dimensional data to show features from multidimentsional data (breast cancer)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import datasets

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

clf_full = None
score_full = None

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)


ds = datasets.load_breast_cancer()

plot_features = [
	( 'mean smoothness', 'mean compactness'),
	('mean smoothness','mean area'),
	('mean smoothness','mean symmetry')
]

figure = plt.figure(figsize=(27, 9))
i = 1

print "All features"
print ds.feature_names

title = ""

for ft_cnt, feature in enumerate(plot_features):

	title += "| " + feature[0] + "," + feature[1] + " | "

	if hasattr(ds, "target"): 
		X = ds.data  # data to train 
		y = ds.target  # matrix from which class value belongs 
	else:   
		X = ds[0] 
		y = ds[1] 


	X = StandardScaler().fit_transform(X)
	X_train, X_test, y_train, y_test = \
	    train_test_split(X, y, test_size=.4, random_state=42)

	feature1, = np.where(ds.feature_names == feature[0])
	feature1 = feature1[0]	
	feature2, = np.where(ds.feature_names == feature[1])
	feature2 = feature2[0]	

	print "----------------------------------------"
	print "\nPlots row ", ft_cnt, ". Choosed features: ", feature
	
	X_train_plot = np.stack((X_train[:,feature1],X_train[:,feature2]), axis=-1)
	X_test_plot = np.stack((X_test[:,feature1],X_test[:,feature2]), axis=-1)

	y_train_plot = y_train #np.stack((y_train[:,feature1],y_train[:,feature2]), axis=-1)
	y_test_plot = y_test #np.stack((y_test[:,feature1],y_test[:,feature2]), axis=-1)

	x_min, x_max = X[:, feature1].min() - .5, X[:, feature1].max() + .5
	y_min, y_max = X[:, feature2].min() - .5, X[:, feature2].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	
	xx = np.swapaxes(xx,0,1)
	yy = np.swapaxes(yy,0,1)
	

	# just plot the dataset first
	cm = plt.cm.RdBu
	cm_bright = ListedColormap(['#FF0000', '#0000FF'])
	ax = plt.subplot(len(plot_features), len(classifiers) + 1, i)

	if ft_cnt == 0:
		ax.set_title("Input data")

	# Plot the training points
	ax.scatter(X_train_plot[:, 0], X_train_plot[:, 1], c=y_train, cmap=cm_bright,
	           edgecolors='k')
	# Plot the testing points
	ax.scatter(X_test_plot[:, 0], X_test_plot[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
	           edgecolors='k')

	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())
	i += 1

	if ft_cnt == len(plot_features) - 1:
		print "----------------------------------------"
		print "\nAccuracy evaluated on full dataset: \n"
	# iterate over classifiers
	for name, clf in zip(names, classifiers):
		
		ax = plt.subplot(len(plot_features), len(classifiers) + 1, i)
	
		# Based on 2 features from dataset predicion and accuracy count
		clf.fit(X_train_plot, y_train_plot)
		score = clf.score(X_test_plot, y_test_plot)
			


		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		
		if hasattr(clf, "decision_function"):
			Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		else:
			Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
		
		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
		
		# Plot the training points
		ax.scatter(X_train_plot[:, 0], X_train_plot[:, 1], c=y_train_plot, cmap=cm_bright,
		           edgecolors='k')
		# Plot the testing points
		ax.scatter(X_test_plot[:, 0], X_test_plot[:, 1], c=y_test_plot, cmap=cm_bright,
		           edgecolors='k', alpha=0.6)
		
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())

		if ft_cnt == 0:
			ax.set_title(name)

		ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
		        size=15, horizontalalignment='right')
		i += 1

		# Preditcion based on complete dataset		
		if ft_cnt == len(plot_features) - 1:
			clf_full = clf
			clf_full.fit(X_train, y_train)
			score_full = clf_full.score(X_test, y_test)
			print name, " classifier accuraccy: %.2f " % score_full


print "----------------------------------------"
print "Plotting classifiers results based on partial dataset"
figure.canvas.set_window_title(title)	
plt.tight_layout()
plt.show()
