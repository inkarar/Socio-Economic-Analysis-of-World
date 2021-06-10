import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Socio-Economic Analysis App",layout='wide')

st.write('''
# Socio-Economic analysis of Countries
''')

st.write('''
### Following are the only factors included in Cluster analysis
''')

info = pd.read_csv('data-dictionary.csv')
st.table(info)

data = pd.read_csv('Country-data.csv')

st.write('''
## Pick any 5 socio-economic factors you wish to analyze
''')

option = st.multiselect('Select/Unselect only 5 socio-economic factors',
                        ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp'],
                                ['exports','imports', 'health','inflation','life_expec'],
                        help = 'Please select/unselect only 5 of these factors for better user experience and results' )

# import figure factory
import plotly.figure_factory as ff
data_matrix = data.loc[:,option]
data_matrix["index"] = np.arange(1,len(data_matrix)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data_matrix, diag='histogram', index='index',colormap='Portland',
                                  colormap_type='cat',
                                   title="Visualising Scatter Matrix Plot of Selected Factors",
                                  height= 800, width = 1100)
st.plotly_chart(fig)


st.write('''
### For simplicity, all countries are divided into 3 clusters purely based on selected factors, cluster 0,1,2 corresponding to under developed, developing and developed countries respectively
#### cluster 0 ----> Under Developed Nations
#### cluster 1 ----> Developing Nations
#### cluster 2 ----> Developed Nations
''')

dataset = data.drop(['country'], axis =1)

dataset = dataset[option]

columns = dataset.columns

scaler = MinMaxScaler()

rescaled_dataset_minmax = scaler.fit_transform(dataset)

df_minmax = pd.DataFrame(data= rescaled_dataset_minmax , columns = columns )

# import PCA 
from sklearn.decomposition import PCA

# fit and transform
pca = PCA()
pca.fit(df_minmax)
pca_data_minmax = pca.transform(df_minmax)

per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]
pca_df_minmax = pd.DataFrame(pca_data_minmax, columns = labels)


data2 = pca_df_minmax[['PC1','PC2','PC3']]

km = KMeans (n_clusters = 3, init = 'random', n_init = 10,max_iter = 300, tol = 1e-4,random_state = 0)

y_predicted_minmax = km.fit_predict(df_minmax)

y_predicted_data2 = km.fit_predict(data2) 

df_minmax['cluster'] = y_predicted_minmax

dataset['cluster'] = y_predicted_data2

data['cluster'] = y_predicted_minmax.tolist()
data.astype({'cluster': 'category'})

option.append('cluster')


st.write('''
## Average metric value for each cluster
''')
clusters_table = pd.pivot_table(data[option], index=['cluster'])
st.table(clusters_table)

st.write('''
## Visual represenation of Clusters
Every country belongs to one of the three clusters
''')

fig = px.scatter_matrix(data[option],
    dimensions=option,
    color="cluster",height= 800, width = 1100)
st.plotly_chart(fig)


col1, col2, col3 = st.beta_columns(3)

# cluster 0 
cluster_0 = data.loc[data['cluster'] == 0]
col1.write('''
### Cluster 0 - Under Developed Nations
''')
col1.table(cluster_0.country.unique().tolist())

# cluster 1 
cluster_1 = data.loc[data['cluster'] == 1]
col2.write('''
### Cluster 1 - Developing Nations
''')
col2.table(cluster_1.country.unique().tolist())

# cluster 2 
cluster_2 = data.loc[data['cluster'] == 2]
col3.write('''
### Cluster 2 - Developed Nations
''')
col3.table(cluster_2.country.unique().tolist())

st.write('''
## Please be mindful of the following facts! 
This is not a comprehensive analysis by any means.
The clustering method alone was not sufficient to provide a final verdict of analysis, however it will contribute to guide actions
for further analysis and explore the data in more detail.

Further analysis could be done by adding more features related to the context and constraints that the recommended countries might
be facing, or systemic challenges that could hinder growth and development. Issues like corruption, political/civic society crisis/ natural
disasters and other risks could expand this analysis to develop a more suitable criteria for Socio-Economic analysis depending on the current
context of a country beyond these macro indicators.
''')

