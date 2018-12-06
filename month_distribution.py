import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl 
mpl.use('Tkagg')

import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np
import re
from datasetup.data_preprocessing import openFile
import calendar
import matplotlib.ticker as plticker

file = 'exported/files/Clusters_Origin_NonFiltered_5.csv'
data = openFile(file)
k = int(file.split('_')[3].split('.')[0])
#data = data.set_index('TCID')
maximum = 130
plt.figure()

for i in range(k):
	cluster = data.loc[data['CLUSTER_LABELS'] == i, ['TIME', 'TCID']].copy()
	cluster = cluster.set_index('TCID')
	month = cluster['TIME'].values
	month = [m.split('/')[0] for m in month]
	month = [re.sub('0', '', m) for m in month]
	month = [int(m) for m in month]
	cluster.loc[:, 'MONTH'] = month
	cluster = cluster.drop('TIME', axis = 1)
	month_dist = pd.DataFrame(cluster['MONTH'].value_counts(sort = False).reset_index())


	month_dist.columns = ['MONTH', 'FREQUENCY']
	
	month_dist = month_dist.sort_values('MONTH')
	
	

	
	
	

	plt.subplot(2,3,i+1)
			
	plt.bar(month_dist.MONTH, height = month_dist.FREQUENCY)
	plt.xlabel('Calendar Month')
	plt.ylabel('Number of TCs')
	plt.title('Cluster' + str(i+1))
	#plt.xticks(np.arange(1, 13), calendar.month_name[1:13], rotation=90)
	plt.xticks(np.arange(1, 13))
	plt.ylim((0,maximum))
	plt.tight_layout()
	

plt.savefig('exported/plots/calendar-month-distribution/with-filter/Origin_Month_Distribution_5.png')


	