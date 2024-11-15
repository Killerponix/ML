import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from dbscan import DBSCAN
import geopandas as gpd 
import pandas as pd 

df = pd.read_csv('nations.csv')
df['income'].replace(to_replace=['Low income']         , value=  647.5, inplace=True)
df['income'].replace(to_replace=['Lower middle income'], value= 2445.5, inplace=True)
df['income'].replace(to_replace=['Upper middle income'], value= 7975.5, inplace=True)
df['income'].replace(to_replace=['High income']        , value=27218.0, inplace=True)
df['income'].replace(to_replace=['High income: OECD']  , value=42380.0, inplace=True)
df['region'] = pd.factorize(df.region)[0]
for i in range(df['year'].min(),df['year'].max()+1):
    indexYear = (df['year'] == i)  
    for j in range(df['region'].min(),df['region'].max()+1):
        indexRegion = (df['region'] == j)
        index = np.logical_and(indexYear,indexRegion)
        columnMean = df[index].mean()
        df[index] = df[index].fillna(columnMean)
df.drop(columns='population', inplace=True)
df.drop(columns='region', inplace=True)    
df = df[np.logical_or(df['year'] == 1994 ,df['year'] == 2014)]
Features    = df[['iso3c','country','year','gdp_percap', 'life_expect', 'birth_rate', 'neonat_mortal_rate', 'income']]
FeaturesStd = Features.copy()
numFeatureStart = 3
xbar = np.mean(FeaturesStd.iloc[:,numFeatureStart:],axis=0) 
sigma = np.std(FeaturesStd.iloc[:,numFeatureStart:],axis=0)
FeaturesStd.iloc[:,numFeatureStart:] =  (FeaturesStd.iloc[:,numFeatureStart:] - xbar) / sigma 

projektionsDim = 2
Sigma = np.cov(FeaturesStd.iloc[:,numFeatureStart:].T)
(lamb, W) = np.linalg.eig(Sigma)
eigenVarIndex = np.argsort(lamb)[::-1]
WP = W[:,eigenVarIndex[0:projektionsDim]]

Features1994    = Features[FeaturesStd['year'] == 1994]
FeaturesStd1994 = FeaturesStd[FeaturesStd['year'] == 1994]
XProj1994 = ( WP.T @ FeaturesStd1994.iloc[:,numFeatureStart:].T ).T
XFull1994 = FeaturesStd1994.iloc[:,numFeatureStart:]
Features2014    = Features[FeaturesStd['year'] == 2014]
FeaturesStd2014 = FeaturesStd[FeaturesStd['year'] == 2014]
XProj2014 = ( WP.T @ FeaturesStd2014.iloc[:,numFeatureStart:].T ).T
XFull2014 = FeaturesStd2014.iloc[:,numFeatureStart:]

def defOutlier(c):
    clusterDaten = np.unique(c,return_counts=True)
    for i, clusterNo in enumerate(clusterDaten[0]):
        if clusterDaten[1][i] < 3: 
            idx = np.flatnonzero(c==clusterDaten[0][i])
            c[idx] = -1
    return(c)
        
def addGDPandStat(c, features, featuresStd, verbose=True):    
    clusterDaten   = np.unique(c,return_counts=True)
    gdpClusterMean = -1*np.ones(clusterDaten[0].shape[0])
    
    for i, clusterNo in enumerate(clusterDaten[0]):
        if clusterDaten[0][i] != -1: 
            gdpClusterMean[i] = features.loc[c == clusterNo,'gdp_percap'].mean()
        if verbose: 
            print('Cluster %d: %3d Elemente' % (clusterNo,clusterDaten[1][i]),end='')
            print(np.array(features[c == clusterNo].iso3c))
            for feat in ['gdp_percap', 'life_expect', 'birth_rate', 
                         'neonat_mortal_rate', 'income']:
                m     = features.loc[c == clusterNo,feat].mean()
                s     = features.loc[c == clusterNo,feat].std()
                mStd  = featuresStd.loc[c == clusterNo,feat].mean()
                sStd  = featuresStd.loc[c == clusterNo,feat].std()
                print(feat, ': mean : %5.2f std %5.2f ' % (m,s) )
                print(feat, '[std]: mean : %5.2f std %5.2f ' % (mStd,sStd) )
            print()

    gdpClusterNo = -1*np.ones(c.shape[0])
    numbers = np.argsort(gdpClusterMean)
    for i, clusterNo in enumerate(clusterDaten[0]):
        if gdpClusterMean[numbers == i] != -1:
            gdpClusterNo[c == clusterNo] = np.flatnonzero(numbers == i)
    
    cluster = pd.DataFrame({'cluster': c,'iso_a3': features.iso3c, 'gdpIdx': gdpClusterNo})
    return cluster

def fixMissingCodes(world):
    world2 = world.copy()
    world2.loc[world['name'] == 'France', 'iso_a3'] = 'FRA'
    world2.loc[world['name'] == 'Norway', 'iso_a3'] = 'NOR'
    world2.loc[world['name'] == 'Somaliland', 'iso_a3'] = 'SOM'
    world2.loc[world['name'] == 'Kosovo', 'iso_a3'] = 'RKS'
    return world2

def visualCluster(c, name):  
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = fixMissingCodes(world)
    world = world.merge(c, on='iso_a3')
    ax = world.plot(column='gdpIdx', cmap='hot_r', legend=True, vmin=-1, 
                    legend_kwds={'label': 'Cluster von Laendern',
                                 'orientation': "horizontal"})
    ax.set_title(name, fontsize=9)
    return ax

print('-----------    1994 mit DBSCAN    -----------')
clusterAlg = DBSCAN(XProj1994.to_numpy() )
c1994DB = clusterAlg.fit_predict(eps=0.275,MinPts=4)
c1994DB = defOutlier(c1994DB)
cluster1994DB = addGDPandStat(c1994DB,Features1994,FeaturesStd1994)
visualCluster(cluster1994DB, '1994 mit DBSCAN')

print('-----------    1994 mit hierarchischem Clustering [single]    -----------')
D = pdist(XFull1994.to_numpy(), metric='euclidean')
Z = linkage(D, 'single', metric='euclidean')
plt.figure()
dendrogram(Z,truncate_mode='lastp',p=12)
c1994SG = fcluster(Z,10,criterion='maxclust')
c1994SG = defOutlier(c1994SG)
cluster1994SG = addGDPandStat(c1994SG,Features1994,FeaturesStd1994) 
visualCluster(cluster1994SG, '1994 mit hierarchischem Clustering [single]')

print('-----------    1994 mit hierarchischem Clustering [centroid]    -----------')
D = pdist(XFull1994.to_numpy(), metric='euclidean')
Z = linkage(D, 'centroid', metric='euclidean')
plt.figure()
dendrogram(Z,truncate_mode='lastp',p=12)
c1994HC = fcluster(Z,10,criterion='maxclust')
c1994HC = defOutlier(c1994HC)
cluster1994HC = addGDPandStat(c1994HC,Features1994,FeaturesStd1994)
visualCluster(cluster1994HC, '1994 mit hierarchischem Clustering [centroid]')

print('-----------    2014 mit DBSCAN    -----------')
clusterAlg = DBSCAN(XProj2014.to_numpy() )
c2014DB = clusterAlg.fit_predict(eps=0.275,MinPts=4)
c20144DB = defOutlier(c2014DB)
cluster2014DB = addGDPandStat(c2014DB,Features2014,FeaturesStd2014)
visualCluster(cluster2014DB, '2014 mit DBSCAN')

print('-----------    2014 mit hierarchischem Clustering [single]    -----------')
D = pdist(XFull2014.to_numpy(), metric='euclidean')
Z = linkage(D, 'single', metric='euclidean')
plt.figure()
dendrogram(Z,truncate_mode='lastp',p=12)
c2014HS = fcluster(Z,10,criterion='maxclust')
c2014HS = defOutlier(c2014HS)
cluster2014HS = addGDPandStat(c2014HS,Features2014,FeaturesStd2014)
visualCluster(cluster2014HS, '2014 mit hierarchischem Clustering [single]')

print('-----------    2014 mit hierarchischem Clustering [centroid]    -----------')
D = pdist(XFull2014.to_numpy(), metric='euclidean')
Z = linkage(D, 'centroid', metric='euclidean')
plt.figure()
dendrogram(Z,truncate_mode='lastp',p=12)
c2014HC = fcluster(Z,10,criterion='maxclust')
c2014HC = defOutlier(c2014HC)
cluster2014HC = addGDPandStat(c2014HC,Features2014,FeaturesStd2014)
visualCluster(cluster2014HC, '2014 mit hierarchischem Clustering [centroid]')

from kNearestNeighborConsistency import kNearestNeighborConsistency
       
clusterEval = kNearestNeighborConsistency(XFull1994.to_numpy(),k=3)
indexDB1994 = clusterEval.consistency(cluster1994DB.cluster.to_numpy())
indexSG1994 = clusterEval.consistency(cluster1994SG.cluster.to_numpy())
indexHC1994 = clusterEval.consistency(cluster1994HC.cluster.to_numpy())
print(indexDB1994,indexSG1994,indexHC1994)

clusterEval = kNearestNeighborConsistency(XFull2014.to_numpy(),k=3)
indexDB2014 = clusterEval.consistency(cluster2014DB.cluster.to_numpy())
indexHS2014 = clusterEval.consistency(cluster2014HS.cluster.to_numpy())
indexHC2014 = clusterEval.consistency(cluster2014HC.cluster.to_numpy())
print(indexDB2014,indexHS2014,indexHC2014)


def visualGroups(cluster, XProj, Features, year,f1='neonat_mortal_rate',f2='gdp_percap',highlight=False, boldL=['DEU']):
    if highlight:
        alphaGeneral = 0.1
    else:
        alphaGeneral = 0.95
        
    myTitle = 'Laender im Jahr ' + str(year)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    markers=['*', 's', '<', 'o', 'X', '^', 'h', 'H', 'D', 'd', 'P']
    colorName = ['red','black','blue','green', 'c', 'm', 'y']    
    
    XProj = XProj.to_numpy()
    for i in range(0,cluster.max()+1):
        index = np.flatnonzero(cluster == i)
        ax.scatter(XProj[index,0],XProj[index,1],c=colorName[i],s=60,alpha=alphaGeneral,marker=markers[i])
    for i in range(cluster.shape[0]):
        ax.annotate(Features.iso3c.iloc[i], (XProj[i,0], XProj[i,1]),alpha=alphaGeneral*0.8)
    
    if highlight:
        print('Zuordnung einzelner Laender: ',end='')
        for i in boldL:
            j = np.where(Features['iso3c'] == i)
            j = j[0][0]
            ax.scatter(XProj[j,0],XProj[j,1],c=colorName[cluster[j]],s=60,marker=markers[cluster[j]])
            ax.annotate(i, (XProj[j,0], XProj[j,1]),color='k')
            print('%s %d; ' % (i,cluster[j]), end='' )
        print('\n')
    
    ax.set_xlabel('1. Hauptkomponente')
    ax.set_ylabel('2. Hauptkomponente')     
    ax.set_title(myTitle)
    ax.set_xlim([-4,5])
    ax.set_ylim([-2,1.5])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    markers=['*', 's', '<', 'o', 'X', '^', 'h', 'H', 'D', 'd', 'P']
    colorName = ['red','black','blue','green', 'c', 'm', 'y'] 
    for i in range(0,cluster.max()+1):
        index = np.flatnonzero(cluster == i)
        ax.scatter(Features.neonat_mortal_rate.iloc[index],Features.gdp_percap.iloc[index],c=colorName[i],s=60,alpha=alphaGeneral,marker=markers[i])
    for i in range(cluster.shape[0]):
        ax.annotate(Features.iso3c.iloc[i], (Features.neonat_mortal_rate.iloc[i], Features.gdp_percap.iloc[i]),alpha=alphaGeneral*0.8)
    
    if highlight:
        for i in boldL:
            j = np.where(Features['iso3c'] == i)
            j = j[0][0]
            ax.scatter(Features.neonat_mortal_rate.iloc[j],Features.gdp_percap.iloc[j],c=colorName[cluster[j]],s=60,marker=markers[cluster[j]])
            ax.annotate(i, (Features.neonat_mortal_rate.iloc[j],Features.gdp_percap.iloc[j]),color='k')
    
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(myTitle)
    ax.set_xlim([0,75])
    ax.set_ylim([0,60000])
    
    
Laenderliste = []
Laenderliste.append('DEU') # Deutschland 
Laenderliste.append('GBR') # UK 
Laenderliste.append('ITA') # Italien 
Laenderliste.append('GRC') # Griechenland
Laenderliste.append('POL') # Polen
Laenderliste.append('ESP') # Spanien
Laenderliste.append('USA') # USA
Laenderliste.append('SYR') # Syrien
Laenderliste.append('BRA') # Brasilien
Laenderliste.append('IRQ') # Irak
Laenderliste.append('MMR') # Myanmar
Laenderliste.append('CHN') # Volksrepublik China
Laenderliste.append('IND') # Indien 
Laenderliste.append('SAU') # Saudi-Arabien
Laenderliste.append('BHS') # Bahamas
Laenderliste.append('GHA') # Ghana
Laenderliste.append('RWA') # Ruanda

visualGroups(c1994DB, XProj1994, Features1994, 1994, highlight=True, boldL = Laenderliste)
visualGroups(c2014DB, XProj2014, Features2014, 2014, highlight=True, boldL = Laenderliste)
