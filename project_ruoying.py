# -*- coding: utf-8 -*-
"""
Ruoying Dai
"""

#%% Import packages
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn import decomposition # For PCA
import networkx as nx
import seaborn as sns


#%% Import data
data = genfromtxt('C:/1NetworkAnalysis/project/department.csv',
              delimiter=',',
              skip_header = 1,
              usecols = (tuple(range(1,30))))


#%% Define peudo name
names = ['Bill','Ana','Jim', 'Sam', 'Lou',
         'Kim', 'Mel', 'Rob', 'Ray', 'Ali',
         'Ken', 'Wes', 'Mia', 'Sue', 'Mic',
         'Pam', 'Joe', 'Liv', 'Sid', 'Zoe',
         'Tom', 'Eva', 'Amy', 'Leo', 'May',
         'Del', 'Flo', 'Bas', 'Wu']
#%% 
# K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
y_kmeans = kmeans.predict(data)

#%% PCA 
# In order to visualize all the points
pca = decomposition.PCA(n_components=2)
pca.fit(data)
X = pca.transform(data)
X2 = X.transpose()
xpts = X2[0]
xpts2 = np.interp(xpts, (xpts.min(), xpts.max()), (0, 1))
ypts = X2[1]
ypts2 = np.interp(ypts, (ypts.min(), ypts.max()), (0, 1))

#%%
# Fuzzy c-means 1
# For original data
# Transcribed from source:
# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
# Set up the loop and plot
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
alldata = data
fpcs = []

colors = ['b', 'orange', 'g', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    #The FPC is defined on the range from 0 to 1, with 1 being best. It is a metric which tells us how cleanly our data is described by a certain model. 
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    
    if ncenters ==5 :
        cluster2 = cluster_membership
        cluster2u = u

    for j in range(ncenters):

        ax.plot(xpts2[cluster_membership == j],
                ypts2[cluster_membership == j],
                '.', markersize=15, 
                color=colors[j])

    # Mark the center of each fuzzy cluster
    #for pt in cntr:
    #    ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Clusters = {0}, FPC = {1:.2f}'.format(ncenters, fpc),
                 fontsize=18)
    #ax.axis('off')

fig1.tight_layout()

#%%
# Fuzzy c-means 2
# For row-stochastic matrix
# Transcribed from source:
# Set up the loop and plot
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
alldata = data
alldata2 = alldata/alldata.sum(axis=1)[:,None]
fpcs = []

colors = ['b', 'orange', 'g', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata2, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    #The FPC is defined on the range from 0 to 1, with 1 being best. It is a metric which tells us how cleanly our data is described by a certain model. 
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    
    if ncenters ==2 :
        cluster2 = cluster_membership
        cluster2u = u
        
    for j in range(ncenters):

        ax.plot(xpts2[cluster_membership == j],
                ypts2[cluster_membership == j],
                '.', markersize=15, 
                color=colors[j])

    # Mark the center of each fuzzy cluster
    #for pt in cntr:
    #    ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Clusters = {0}, FPC = {1:.2f}'.format(ncenters, fpc),
                 fontsize=18)
    #ax.axis('off')

fig1.tight_layout()

#%%
u = cluster2u.transpose()
u2_1 = u[:,0]>0.4
u2_2 = u[:,1]>0.4

#%% Draw network by adjacency matrix
# source:
# https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file


def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix != 0 )
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, 
            node_size=1200, 
            font_size=16,
            labels=mylabels, with_labels=True)
    plt.show()



#%% Plot original graph
G=nx.Graph()
i=1
G.add_node(i,pos=(i,i))
G.add_node(2,pos=(2,2))
G.add_node(3,pos=(1,0))
G.add_edge(1,2,weight=0.5)
G.add_edge(1,3,weight=9.8)
pos=nx.get_node_attributes(G,'pos')
nx.draw(G,pos)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.savefig(<wherever>)
#%%
colors = ['gold','goldenrod','forestgreen',
          'darkgreen','cornflowerblue','mediumblue',
          'darkslategray']
nodecolors_list = ['b', 'g']
nodecolors_list5 = ['b', 'orange', 'g', 'c', 'm']
def draw_graph(adjacency_matrix, mylabels):
    gr = nx.Graph()
    for i in range(29):
        for j in range(29):
            wei = int(adjacency_matrix[i][j])
            if wei!= 0:
                gr.add_edge(i, j, #weight =wei/6, 
                            color = colors[wei])
    print(nx.voterank(gr, 5))
    #labels = nx.get_edge_attributes(gr,'weight')
    #values = [val_map.get(node, 0.25) for node in G.nodes()]
    edges = gr.edges()
    node_colors = [nodecolors_list5[cluster2[i]] for i in range(29)]
    colors2 = [gr[u][v]['color'] for u,v in edges]
    #pos=nx.spring_layout(gr,scale=2)
    plt.figure(figsize=(16,10))
    nx.draw(gr, 
            #pos=nx.random_layout(gr),
            pos = nx.fruchterman_reingold_layout(gr),
            #pos=nx.circular_layout(gr),
            node_size=1200, 
            font_size=16,
            labels=mylabels, with_labels=True,
            #####edge_labels=labels, 
            #edge_color=colors2,
            node_color=node_colors, 
            font_color='white')
    #nx.draw_networkx_edge_labels(gr,)
    plt.show()
draw_graph(data, {i:names[i] for i in range(29)})
#%% Build a preferential attachment network
# Format:barabasi_albert_graph(n, m, seed=None)

def pre_attach_generator(node_num, edge_num, seed):
    # number of nodes
    n = node_num
    # number of edges
    m = edge_num
    # sample a network
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    # adjacency matrix of g
    adj = nx.adjacency_matrix(g)
    adjarray = adj.toarray()
    plot_degree_dist(g)
    
    return(adjarray)

dif = np.zeros((29,29))

A= (data!=0).astype(int)

seed_num = 10

for seed in range(seed_num):
    plt.figure(figsize=(10,10))
    a_h = pre_attach_generator(29, 5, seed)
#%%    
    h = a_h - A
    dif +=  h/seed_num
    
#%% plot the H matrix
plt.figure(figsize=(12,10))
plt.rcParams.update({'font.size': 18})
sns.diverging_palette(220, 20, as_cmap=True)
ax = sns.heatmap(dif, linewidth=0.5,
                 cmap = sns.diverging_palette(220, 20, as_cmap=True))
plt.show()

#%% plot degree distribution
# source:
# https://stackoverflow.com/questions/53958700/plotting-the-degree-distribution-of-a-graph-using-nx-degree-histogram
def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins = 18)
    plt.show()
    
#%% plot degree distribtion
df = pd.DataFrame()
for i in range(10):
    g = nx.barabasi_albert_graph(n=29, m=5, seed=i)
    degrees = [g.degree(n) for n in g.nodes()]
    new_df = pd.DataFrame({'Degree': degrees,
                          'Run': [i for t in range(29)]})
    df = df.append(new_df, 
                   ignore_index=True)


sns.displot(data=df, x='Degree', hue='Run', kind='kde', fill=True, palette=sns.color_palette('bright')[:10], height=5, aspect=1.5)
#%%
#print(adj.todense())
dif 0 
plt.figure(figsize=(10,6))
show_graph_with_labels(adj.toarray(), {i:names[i] for i in range(29)})

