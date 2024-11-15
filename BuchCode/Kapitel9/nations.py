import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
df = pd.read_csv('nations.csv')
pd.options.display.max_columns = 10 
pd.options.display.width = 180
print(df.head())
print(df.iloc[0:2, 0:5])
print(df.loc[0:2, ['income','neonat_mortal_rate']])
idx = df.loc[:,'region'] == 'South Asia'
print(idx.sum(), type(idx))
maxBirth = df.loc[idx,'birth_rate'].max()
minBirth = df.loc[idx,df.columns[7]].min()
print('In der Region South Asia ist die minimale Geburtenrate %.2f und die maximale %.2f .' 
      % (minBirth,maxBirth))
print(df[df.iso3c=='AFG'])
df.drop(columns='iso3c', inplace=True)  #df.set_index('iso3c',inplace=True)
df = df.rename(columns={'country': 'Staat', 'year': 'Jahr', 'gdp_percap': 'BIPproKopf'})
df.rename(columns={'region': 'Region', 'life_expect': 'Lebenserwartung'}, inplace=True)
df.rename(columns={'population': 'Bevoelkerung', 'birth_rate': 'Geburtenrate'}, inplace=True)
df = df.rename(columns={'neonat_mortal_rate': 'Kindersterblichkeit', 'income': 'Einkommen'})
df.hist(bins=25, color='gray')
plt.figure()
ax = df.loc[:,'Einkommen'].hist(color='gray')
ax.set_xlabel('Einkommen')
ax.set_ylabel('Haeufigkeit')
df.plot.scatter(x='Jahr',y='Kindersterblichkeit',c='BIPproKopf',colormap='gray',alpha=0.4)
print(df.Staat.unique())
justOneCountry = df[df.Staat=='Iraq']
justOneCountry.plot.scatter(x='Jahr', y='Kindersterblichkeit', c='k', title='Iraq')
justOneCountry = df[df.Staat=='Rwanda']
justOneCountry.plot.scatter(x='Jahr', y='Kindersterblichkeit', c='k', title='Rwanda')
dfV = df.copy()
dfV['Einkommen'].replace(to_replace=['Low income']         , value=  647.5, inplace=True)
dfV['Einkommen'].replace(to_replace=['Lower middle income'], value= 2445.5, inplace=True)
dfV['Einkommen'].replace(to_replace=['Upper middle income'], value= 7975.5, inplace=True)
dfV['Einkommen'].replace(to_replace=['High income']        , value=27218.0, inplace=True)
dfV['Einkommen'].replace(to_replace=['High income: OECD']  , value=42380.0, inplace=True)
print(dfV['Region'].describe())
print(pd.factorize(dfV.Region))
dfV['Region'] = pd.factorize(dfV.Region)[0]
print(dfV.describe())
df2011 = dfV[dfV.Jahr == 2011].drop(columns=['Jahr'])
df2011num = df2011.select_dtypes('number') #*\label{code:nations:0}
data2011N = (df2011num - df2011num.min()) / (df2011num.max() - df2011num.min())  #*\label{code:nations:1}
data2011N = pd.concat((df2011.Staat, data2011N), axis='columns')
print(data2011N.mean(),'\n',data2011N.std(),'\n',data2011N.median() )  #*\label{code:nations:2}
data2011S = (df2011num - df2011num.mean()) / df2011num.std()
data2011S = pd.concat((df2011.Staat, data2011S), axis='columns') 
pd.options.display.max_rows = None #*\label{code:nations:3-1}
print(data2011N.sort_values(by=['Bevoelkerung'], ascending=False))
print(data2011S.sort_values(by=['Bevoelkerung'], ascending=False))
fig = plt.figure()
pop2011 = data2011S.Bevoelkerung
s = pop2011.std(); m = pop2011.mean(); xmax = pop2011.max()
print(s,m) # Es gilt immer: s = 1, m = 0, weil standardisiert!
ax1 = pop2011.hist(color='gray', bins=50)
ax2 = ax1.twinx()
x = np.linspace(m-s, xmax, 10000)
y = np.exp(-0.5 * ((x-m)/s)**2) / np.sqrt(2*np.pi) / s
ax2.plot(x, y, c='k', lw=2)
x = np.linspace(m-s, m+s, 10000)
y = np.exp(-0.5 * ((x-m)/s)**2) / np.sqrt(2*np.pi) / s
ax2.fill_between(x, y, 0, alpha=0.3, color='r')
ax2.set_ylim(0) 
print(dfV.isna().sum())
dropIdx = ( dfV.Bevoelkerung.isna() | df.Lebenserwartung.isna() )
dfV = dfV[~dropIdx]
print(dfV.isna().sum())
dropIdx = dfV.Geburtenrate.isna()
dfV = dfV[~dropIdx]
withNaN = (dfV.BIPproKopf.isna() | dfV.Kindersterblichkeit.isna())
print("%.2f Prozent der Eintraege enthalten mind. ein NaN" % (withNaN.sum()/dfV.shape[0]*100))
print(dfV[withNaN].Staat.value_counts())
idx = (dfV.BIPproKopf.isna() & dfV.Kindersterblichkeit.isna() )  #*\label{code:nations:3}
dfV = dfV[~idx].copy()
withNaN = (dfV.BIPproKopf.isna() | dfV.Kindersterblichkeit.isna())  #*\label{code:nations:4}
dfnum = dfV.select_dtypes('number') #*\label{code:nations:5}
Y = dfnum.Lebenserwartung.copy() #*\label{code:nations:6}
feature = dfnum.drop(columns=['Lebenserwartung']) #*\label{code:nations:7}
X  = (feature - feature.mean()) / feature.std() #*\label{code:nations:8}

def trainSet(X,Y,percent):
    TrainSet     = np.random.choice(X.shape[0],int(X.shape[0]*percent),replace=False)
    XTrain       = X.iloc[TrainSet,:]
    YTrain       = Y.iloc[TrainSet]
    TestSet      = np.delete(np.arange(0,len(Y)), TrainSet) 
    XTest        = X.iloc[TestSet,:]
    YTest        = Y.iloc[TestSet]
    return XTrain, XTest, YTrain, YTest

from knnRegression import knnRegression
np.random.seed(42)
XTrainP, XTestP, YTrainP, YTestP = trainSet(X[~withNaN],Y[~withNaN],0.7)
model = knnRegression()
model.fit(XTrainP.to_numpy(), YTrainP.to_numpy())
yP    = model.predict(XTestP.to_numpy(),k=2, smear = 10**-3)
print('MAE auf Testset ohne Luecken: %.3f' % (np.mean(np.abs(yP-YTestP.to_numpy()))))

Xdoped = X.drop(columns=['BIPproKopf']) #*\label{code:nations:9}
Xdoped = Xdoped.drop(columns=['Kindersterblichkeit']) #*\label{code:nations:10}
XTDrop = Xdoped.drop(XTestP.index) #*\label{code:nations:11}
YTDrop = Y.drop(XTestP.index)  #*\label{code:nations:12}
modelDrop = knnRegression()
modelDrop.fit(XTDrop.to_numpy(), YTDrop.to_numpy())
yP    = modelDrop.predict(XTestP.drop(columns=['BIPproKopf','Kindersterblichkeit']).to_numpy()
                          ,k=2, smear = 10**-3)
print('MAE auf Testset mit weniger Merkmalen: %.3f' % (np.mean(np.abs(yP-YTestP))))

XTMean = X.drop(XTestP.index).fillna(X.mean())
YTMean = Y.drop(XTestP.index) 
modelMean = knnRegression()
modelMean.fit(XTMean.to_numpy(), YTMean.to_numpy())
yP    = modelMean.predict(XTestP,k=2, smear = 10**-3)
print('MAE auf Testset mit Mean-Imputer: %.3f' % (np.mean(np.abs(yP-YTestP))))

BIP  = X[~withNaN].BIPproKopf.copy()  #*\label{code:nations:13}
fBIP = X.drop(columns=['BIPproKopf']) #*\label{code:nations:14}
modelBIP = knnRegression()
modelBIP.fit(fBIP[~withNaN].to_numpy(), BIP.to_numpy())
idx = X.BIPproKopf.isna() 
Imput    = modelBIP.predict(fBIP[idx].to_numpy(),k=2, smear = 10**-3)
X.loc[idx, 'BIPproKopf'] = Imput #*\label{code:nations:15}
Kid  = X[~withNaN].Kindersterblichkeit.copy()
fKid = X.drop(columns=['Kindersterblichkeit'])
modelKid = knnRegression()
modelKid.fit(fKid[~withNaN].to_numpy(), Kid.to_numpy())
idx = X.Kindersterblichkeit.isna() 
Imput    = modelKid.predict(fKid[idx].to_numpy(),k=2, smear = 10**-3)
X.loc[idx, 'Kindersterblichkeit'] = Imput
XNaNKnn = X[withNaN] 
YNaN = Y [withNaN].to_numpy()
XTKnn = np.vstack( (XNaNKnn.to_numpy(),XTrainP.to_numpy()) )
YTKnn = np.hstack( (YNaN,YTrainP.to_numpy()) )
modelKnn = knnRegression()
modelKnn.fit(XTKnn, YTKnn)
yP    = modelKnn.predict(XTestP,k=2, smear = 10**-3)
print('MAE auf Testset mit kNN-Imputer: %.3f' % (np.mean(np.abs(yP-YTestP))))

yKnn  = model.predict(XNaNKnn.to_numpy(),k=2, smear = 10**-3)
print('PureModel: MAE fuer Daten mit kNN-Imputer: %.3f' % (np.mean(np.abs(yKnn-YNaN))))
yMean = model.predict(XTMean.drop(XTrainP.index).to_numpy(),k=2, smear = 10**-3)
print('PureModel: MAE fuer Daten mit Mean-Imputer: %.3f' % (np.mean(np.abs(yMean-YNaN))))

modelPureDrop = knnRegression()
modelPureDrop.fit(XTrainP.drop(columns=['BIPproKopf','Kindersterblichkeit']).to_numpy(),
                  YTrainP.to_numpy())
XTestNaN = XTMean.drop(XTrainP.index)
XTestNaN = XTestNaN.drop(columns=['BIPproKopf','Kindersterblichkeit'])
yMean = modelPureDrop.predict(XTestNaN.to_numpy(),k=2, smear = 10**-3)
print('DropModel: MAE fuer Daten mit NaN %.3f' % (np.mean(np.abs(yMean-YNaN))))

dfSort = dfV[~dfV.isna().any(axis=1)]
dfSort = dfSort[dfSort.Jahr==2011]    
dfSort = dfSort.select_dtypes('number').drop(columns=['Region','Einkommen','Jahr'])
dfSort = (dfSort - dfSort.mean()) / dfSort.std()
plt.rcParams.update({'figure.max_open_warning': 0})
for feature in dfSort.columns:
    for f in dfSort.columns:
        if f == feature: continue
        plt.figure()
        z = np.polyfit(dfSort.loc[:,f], dfSort.loc[:,feature], 1)
        plt.scatter(dfSort.loc[:,f], dfSort.loc[:,feature], c='k')
        plt.plot(dfSort.loc[:,f], z[0]*dfSort.loc[:,f]+z[1],'r--',lw=3)
        plt.xlabel(f + ' [Standardisiert]', fontsize=14)
        plt.ylabel(feature + ' [Standardisiert]', fontsize=14)
        plt.xlim([-2.5,2.5]); plt.ylim([-2.5,2.5])
        
print(df.corr())
from CARTRegressionTree import bRegressionTree

fullTree = bRegressionTree()
fullTree.fit(XTrainP.to_numpy(),YTrainP.to_numpy())
yP = fullTree.predict(XTestP.to_numpy())
print('CART alle Merkmale: %.3f' % (np.mean(np.abs(yP-YTestP.to_numpy()))))
for f1 in XTrainP.columns:
    for f2 in XTrainP.columns:
        if f1 == f2 : continue
        fTrain = XTrainP.loc[:, [f1,f2] ]
        fTest  = XTestP.loc[:, [f1,f2] ]
        reduTree = bRegressionTree()
        reduTree.fit(fTrain.to_numpy(),YTrainP.to_numpy())
        yP = reduTree.predict(fTest.to_numpy())
        print('CART (',f1,f2,') : %.3f' % (np.mean(np.abs(yP-YTestP.to_numpy()))))
    
plt.figure()
plt.plot(dfV[dfV.Staat=='Germany'].Jahr,dfV[dfV.Staat=='Germany'].Lebenserwartung, 'k-',label='Germany')
plt.plot(dfV[dfV.Staat=='Samoa'].Jahr,dfV[dfV.Staat=='Samoa'].Lebenserwartung, 'k:',label='Samoa')
plt.plot(dfV[dfV.Staat=='Malta'].Jahr,dfV[dfV.Staat=='Malta'].Lebenserwartung, 'r--',label='Malta')
plt.legend()
plt.xlabel('Jahre')
plt.ylabel('Lebenserwartung')

def SBS(X,y,k, verbose=False):
    l=X.shape[1]
    MainSet = np.arange(0,X.shape[0])
    ValSet  = np.random.choice(X.shape[0],int(X.shape[0]*0.25), replace=False)
    TrainSet  = np.delete(MainSet,ValSet)    
    suggestedFeatures = np.arange(0,l) 
    while (k<l):
        Q = np.zeros(l)
        for i in range(l):
            Xred = np.delete(X, i, axis=1)
            reduTree = bRegressionTree()
            reduTree.fit(Xred[TrainSet,:],y[TrainSet])
            error = y[ValSet] - reduTree.predict(Xred[ValSet,:])
            Q[i] = np.mean(np.abs(error)) 
        i = np.argmin(Q)
        if verbose: print(Q);print(suggestedFeatures[i])
        suggestedFeatures = np.delete(suggestedFeatures,i) 
        X = np.delete(X, i, axis=1)
        l = l -1
    return(suggestedFeatures)

np.random.seed(42)
suggestedFeatures = SBS(XTrainP.to_numpy(),YTrainP.to_numpy(),k=2, verbose=True)

def SFS(X,y,k, verbose=False):
    MainSet = np.arange(0,X.shape[0])
    ValSet  = np.random.choice(X.shape[0],int(X.shape[0]*0.25), replace=False)
    TrainSet  = np.delete(MainSet,ValSet)    
    featuresLeft = np.arange(0,X.shape[1]) 
    suggestedFeatures = np.zeros(1,dtype=int)
    l=0
    while (k>l):
        Q = np.inf*np.ones(X.shape[1])
        for i in featuresLeft:
            suggestedFeatures[l] = i
            reduTree = bRegressionTree()
            reduTree.fit(X[np.ix_(TrainSet,suggestedFeatures)],y[TrainSet])
            error = y[ValSet] - reduTree.predict(X[np.ix_(ValSet,suggestedFeatures)])
            Q[i] = np.mean(np.abs(error)) 
        i = np.argmin(Q)
        if verbose: print(Q);print(i)
        suggestedFeatures[l] = i
        featuresLeft = np.delete(featuresLeft,np.argwhere(featuresLeft == i) ) 
        suggestedFeatures = np.hstack( (suggestedFeatures,np.array([0]) ) )
        l = l +1
    suggestedFeatures = suggestedFeatures[0:l]
    return(suggestedFeatures)

np.random.seed(42)
suggestedFeatures = SFS(XTrainP.to_numpy(),YTrainP.to_numpy(),k=2, verbose=True)
