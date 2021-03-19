import pandas as pd 
import numpy as np
df = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Clustering\\Universities.csv")

df_new=scale(df.iloc[:,1:])

df.head()
df.describe()

help(PCA)

#Model building
pca=PCA(n_components=6)

model=pca.fit_transform(df_new)
model

model.shape

#The amount of variance that each PCA explains is 
var=pca.explained_variance_ratio_
var

#Cumulative 

var1=np.cumsum(np.round(var,decimals=4)*100)
var1

plt.plot(var1,color="green")

###############To apply Clustering##################
new_df=pd.DataFrame(model[:,0:4])

from sklearn.cluster import KMeans

help(KMeans)

model1=KMeans(n_clusters=3)
model1.fit(new_df)
model1.labels_
new_df["cluster"]=model1.labels_
new_df=new_df.iloc[:,[4,0,1,2,3]]