import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl 
mpl.use('Tkagg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import numpy as np

from datasetup.data_preprocessing import openFile

from sklearn.cluster import KMeans
from sklearn import metrics


file = 'imported/Origin_NonFiltered.csv'
init_data = openFile(file)
init_data = init_data.set_index('TCID')


n_clusters = [3,5]

def qualityCheck(data, coefficients):
	data.loc[:, 'SIL_VALUES'] = coefficients
	count = len(data)
	data = data.loc[data.SIL_VALUES > 0.0]

	return data, count - len(data)

def silhouette_plot(k, cluster_labels, sample_sil_coefficients, sil_score):
	y_lower = 10
	plt.figure()
	for i in range(k):
		ith_cluster_silhouette_values = \
				sample_sil_coefficients[cluster_labels == i]
		ith_cluster_silhouette_values.sort()

		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i

		color = cm.nipy_spectral(float(i) / k)
		plt.fill_betweenx(np.arange(y_lower, y_upper),
                        			0, ith_cluster_silhouette_values,
                        			facecolor=color, edgecolor=color, 
                        			alpha=0.7)

		plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

		y_lower = y_upper + 10  

	plt.xlabel("The silhouette coefficient values")
	plt.ylabel("Cluster label")

	plt.axvline(x=sil_score, color="red", linestyle="--")
	plt.savefig('exported/plots/Silhouette_graph_Origin_' + str(k) + '.png')
	plt.close()

clusters = {}

for k in n_clusters:
	data = init_data.copy()

	# QUALITY CHECK DATA
	for j in range(1000):
		X = data[['LATITUDE', 'LONGITUDE']].values
		kmeans = KMeans(n_clusters = k)
		cluster_labels = kmeans.fit_predict(X)

		sample_sil_coefficients = metrics.silhouette_samples(X, cluster_labels)

		data, count_negative = qualityCheck(data, sample_sil_coefficients)

		print('Number of data retained after quality check', len(data))

		if count_negative == 0:
			sil_score = metrics.silhouette_score(X, cluster_labels)
			#silhouette_plot(k, cluster_labels, sample_sil_coefficients, sil_score)
			data.loc[:, 'CLUSTER_LABELS'] = cluster_labels
			data[['TIME', 'CLUSTER_LABELS']].to_csv('exported/files/Clusters_Origin_NonFiltered_' + str(k) + '.csv')
			break

	
	
		
	

