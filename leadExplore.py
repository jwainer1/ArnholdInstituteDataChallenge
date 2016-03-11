import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from random import uniform
import pickle
import smopy
import urllib
import pylab
from sklearn.cluster import KMeans, MiniBatchKMeans

allValues = pickle.load(open("data/allValues.pkl", "rb" ) )
years = [2002, 2005, 2008, 2011]

#add in information about which borough the data came from
boroughs = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'StatenIsland'] 
d = {True:1, False:0}
allValues['Borough'] = allValues['UHF'] / 100
allValues['Borough'] = allValues['Borough'].round()
for num, eachBorough in enumerate(boroughs):
    allValues[eachBorough] = allValues['Borough'] == (num + 1)
#print allValues[allValues['Borough'] == 1].count()
#allValues['Bronx'] = allValues[allValues['Borough'] == 1]
print allValues.head(20)
categories = ['Elevated (>=10 ug/dL) Blood Lead Levels'] #, 'Homes with Cracks or Holes', 'Pre-1960 Homes with Peeling Paint']
coordinates = allValues['LatLon'] 

#calculate correlations between real-valued attributes and make a "heat graph" of them:
#dark red is 1, violet is -1, cyan is 0
corMat = DataFrame(allValues.corr())
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom = 0.1)
ax.matshow(corMat)
#xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] for patch in ax]
plt.yticks(range(len(corMat.columns)), corMat.columns);
plt.xticks(range(len(corMat.columns)), corMat.columns, rotation = 90);
plt.show()
#plt.pcolor(corMat)
#plt.title('Correlation heat-map between variables')
#plt.show()

#graph the target values on a scatterplot to see if the outputs are clustered in any way
numEntries = allValues['Elevated (>=10 ug/dL) Blood Lead Levels'].count()
target = []
for i in range(numEntries):
    #add some dither to signal's classification
    target.append(1)# + uniform(-0.1, 0.1))
#plot elevated blood levels with semi-opaque points
data = allValues['Elevated (>=10 ug/dL) Blood Lead Levels']
plt.scatter(data, target, alpha=0.1, s=120)
plt.title("Elevated (>=10 ug/dL) Blood Lead Levels")
plt.xlabel("Rate / 1000 people")
plt.show()

#now let's try it with a histogram
ax = data.hist(bins=20, range=(0,data.max()), color = 'r', label = 'All years')
plt.setp(ax.get_xticklabels(),  rotation=45)
plt.title('Elevated (>=10 ug/dL) Blood Lead Levels')
#plt.legend(loc='upper right')
plt.xlabel("Rate / 1000 people")
plt.ylabel("# of neighborhoods")

plt.show()

#or sets of histograms separated by year
data.hist(bins=20, range=(0,data.max()), by=allValues['Year'] )
#plt.title('Elevated blood lead levels by year')
plt.show()

#or sets of histograms separated by year
data.hist(bins=20, range=(0,data.max()), by=allValues['Borough'] )
#plt.title('Elevated blood lead levels by year')
plt.show()

data = allValues['Pre-1960 Homes with Peeling Paint'] 
data.hist(bins=20, range=(0,data.max()), by=allValues['Year'] )
fig = plt.figure()
fig.canvas.set_window_title('Pre-1960 Homes with Peeling Paint')
#plt.title('Percentages of pre-1960 homes with peeling paint by year')
plt.show()

data = allValues['Homes with Mice or Rats in the Building'] 
data.hist(bins=20, range=(0,data.max()), by=allValues['Year'] )
#plt.title('Percentages of pre-1960 homes with peeling paint by year')
plt.show()

names = ['Homes with Cracks or Holes', 'Homes with Mice or Rats in the Building', 'Pre-1960 Homes with Peeling Paint', 'Year', 'Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'StatenIsland']
features = allValues[names] #  'Homes with Leaks',
normFeatures = (features - features.min()) / (features.max() - features.min()) 
target = allValues[['Elevated (>=10 ug/dL) Blood Lead Levels']]

X = normFeatures.values
y = np.ravel(target.values)

#non-regularized model
clf = LinearRegression()
clf.fit(X, y)
predictions = clf.predict(X)
print 'Non-regularized linear regression coefficients: ', clf.coef_
print 'R^2 value : ', r2_score(y,predictions)


BLLModel = LassoCV(cv=10).fit(X, y)

plt.figure()
plt.plot(BLLModel.alphas_, BLLModel.mse_path_, ':')
plt.plot(BLLModel.alphas_, BLLModel.mse_path_.mean(axis=-1), label='Average MSE Across Folds', linewidth=2)
plt.axvline(BLLModel.alpha_, linestyle='--', label='CV Estimate of Best alpha')
plt.semilogx()
plt.legend()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Mean Square Error')
plt.axis('tight')
plt.title('Determining alpha via LASSO 10-fold CV')
plt.show()

print "alpha Value that Minimizes CV Error ", BLLModel.alpha_
print "Minimum MSE ", min(BLLModel.mse_path_.mean(axis=-1))
bestAlpha = BLLModel.alpha_

alphas, coefs, _ = lasso_path(X, y, return_models=False)
plt.plot(alphas,coefs.T)
plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.axis('tight')
plt.title('Variables as they enter the model')
plt.semilogx()
plt.legend(loc='upper left')
ax = plt.gca()
ax.invert_xaxis()
plt.show()

nattr, nalpha = coefs.shape	

#find coefficient ordering
nzList = []
for iAlpha in range(1,nalpha):
    coefList = list(coefs[: ,iAlpha])
    nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
    for q in nzCoef:
        if not(q in nzList):
            nzList.append(q)

nameList = [names[nzList[i]] for i in range(len(nzList))]
print("Attributes Ordered by How Early They Enter the Model", nameList)

indexLTbestAlpha = [index for index in range(100) if alphas[index] > bestAlpha]
bestIndex = max(indexLTbestAlpha)
#here's the set of coefficients to deploy
bestCoef = list(coefs[:,bestIndex])
print("Best Coefficient Values ", bestCoef)

clf = Lasso(alpha=bestAlpha)
clf.fit(X, y)
predictions = clf.predict(X)
print 'Lasso (L1) regression coefficients: ', clf.coef_
print 'Lasso (L1) regression intercept: ', clf.intercept_
print 'R^2 value : ', r2_score(y,predictions)

#try clustering the outputs
num_clusters = 3
km = MiniBatchKMeans(n_clusters = num_clusters, n_init = 1000, random_state = 1000)
km.fit(target.values)
clusters = km.labels_.tolist()
allValues['BloodLevelCluster'] = clusters

#graph the clustered outputs on a scatterplot to see if the clustering looks good
colors = ['r','g','b']
labels = ['Lower levels','Higher levels','Medium levels']
for eachCluster in range(0,num_clusters):
    clusterValues = allValues[allValues['BloodLevelCluster'] == eachCluster]['Elevated (>=10 ug/dL) Blood Lead Levels'] 
    numEntries = clusterValues.count()
    target = []
    for i in range(numEntries):
        #add some dither to signal's classification
        target.append(1)
    #plot elevated blood levels with semi-opaque points
    plt.scatter(clusterValues, target, color = colors[eachCluster], alpha=0.1, s=120, label=labels[eachCluster])
plt.title("K-means clustering of elevated blood lead level data")
plt.legend()
plt.xlabel("Rate / 1000 people")
plt.show()

#now try doing regression on points in one of the clusters
featuresClust = allValues[allValues['BloodLevelCluster'] == 1][['Homes with Cracks or Holes', 'Homes with Mice or Rats in the Building', 'Pre-1960 Homes with Peeling Paint', 'Year', 'Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'StatenIsland']] #'Homes with Leaks', 
normFeatures = (featuresClust - features.min()) / (features.max() - features.min()) 
target = allValues[allValues['BloodLevelCluster'] == 1][['Elevated (>=10 ug/dL) Blood Lead Levels']]

X = normFeatures.values
y = np.ravel(target.values)
clf = LinearRegression()
clf.fit(X, y)
predictions = clf.predict(X)
print 'cluster 1s Linear regression coefficients: ', clf.coef_
print 'cluster 1s R^2 value : ', r2_score(y,predictions)

#separates out the lat/long values and put them back in the dataframe 
latlongs = zip(*coordinates)
lons = list(latlongs[1][:])
lats = list(latlongs[0][:])
allValues['Latitude'] = lats
allValues['Longitude'] = lons

#map the elevated blood lead levels for each neighborhood, using a different map for each year
min = allValues[['Longitude', 'Latitude']].min()
max = allValues[['Longitude', 'Latitude']].max()
colors = ['red', 'blue', 'green', 'orange', 'brown']

f,ax = plt.subplots(4, sharex=True, sharey=True)

for num1, eachYear in enumerate(years):
    map = smopy.Map((min[1],min[0]), (max[1],max[0]), z=11, margin=0.1)
    ax[num1] = map.show_mpl(figsize=(8, 6))
    ax[num1].set_xlabel("Longitude")
    ax[num1].set_ylabel("Latitude")
    title = "Rates of elevated blood lead levels / 1000 people in " + str(eachYear)
    ax[num1].set_title(title)
    #oldSizes = np.zeros(42)
    for num, eachBorough in enumerate(boroughs):
        sizes = allValues[(allValues['Year'] == eachYear) & (allValues['Borough'] == (num+1))][['Elevated (>=10 ug/dL) Blood Lead Levels']].values * 10
        sizes.shape = (len(sizes),)
        points = allValues[(allValues['Year'] == eachYear) & (allValues['Borough'] == (num+1))][['Longitude', 'Latitude']].values
        lons, lats = map.to_pixels(points[:,1], points[:,0]) 
        ax[num1].scatter(lons, lats, marker='o', s=sizes, color=colors[num], label=eachBorough)
    ax[num1].legend(loc='upper left')
    #plt.show()
f.subplots_adjust(hspace=0.1)
plt.show()

