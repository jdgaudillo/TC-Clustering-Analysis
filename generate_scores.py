import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np
import time

from datasetup.data_preprocessing import openFile

from sklearn.cluster import KMeans
from sklearn import metrics

file = 'imported/Origin_NonFiltered.csv'
init_data = openFile(file)
init_data = init_data.set_index('TCID')
init_data = init_data[['LATITUDE', 'LONGITUDE']]

n_clusters = range(3, 11)
scores = {}
counter = 0

init_data.loc[:, 'SIL_VALUES'] = 0.0

print('Original number of data', len(init_data))

def qualityCheck(data, coefficients):
	data.loc[:, 'SIL_VALUES'] = coefficients
	count = len(data)
	data = data.loc[data.SIL_VALUES > 0.0]

	return data, count - len(data)

for i in range(100):
	print('Trial number ', i)
	start_time = time.time()

	for k in n_clusters:
		data = init_data.copy()

		# QUALITY CHECK DATA
		for j in range(1000):
			X = data.values
			kmeans = KMeans(n_clusters = k)
			cluster_labels = kmeans.fit_predict(X)

			sample_sil_coefficients = metrics.silhouette_samples(X, cluster_labels)

			data, count_negative = qualityCheck(data, sample_sil_coefficients)

			print('Number of data retained after quality check', len(data))

			if count_negative == 0:
				X = data.values
				kmeans = KMeans(n_clusters = k)
				cluster_labels = kmeans.fit_predict(X)
				sil_score = metrics.silhouette_score(X, cluster_labels)
				ssd_center = kmeans.inertia_
				scores[counter] = {'trial': i + 1,
							'cluster_number': k,
							'silhouette_score': sil_score,
							'inertia': ssd_center}
				counter += 1
				break
	print('Runtime', time.time() - start_time)
	

result = pd.DataFrame.from_dict(scores, orient = 'index')
result = result.sort_values(['silhouette_score', 'inertia'], ascending = [False, True])
result.to_csv('exported/files/Scores_Origin_NonFiltered.csv')