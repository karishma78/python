#import dataset as dataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)


dataset = pd.read_csv('CC.csv')

#QUESTION 1

# finding  Null values in the dataset
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

print(nulls)

dataset.loc[(dataset['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=dataset['MINIMUM_PAYMENTS'].mean()
dataset.loc[(dataset['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=dataset['CREDIT_LIMIT'].mean()


#elbow method to find the good no. of clusters
x = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]


wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,10),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#QUESTION 2:

#Calculate the silhoutte score
km = KMeans(n_clusters=3)
km.fit(x)
Y_cluster_kmeans= km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, Y_cluster_kmeans)
print('normal score:', score)

#QUESTION 3:

#  Trying feature scaling
scaler = StandardScaler()
scaler.fit(x)
x_scaler = scaler.transform(x)
km = KMeans(n_clusters=3)
km.fit(x_scaler)
Y_cluster_kmeans= km.predict(x_scaler)
from sklearn import metrics
scaledscore = metrics.silhouette_score(x_scaler, Y_cluster_kmeans)
print('scaled score:', scaledscore)

#QUESTION 4
# Apply PCA on normal x.
pca = PCA(3)
x_pca = pca.fit_transform(x)

#BONUS QUESTION 1:

  # PCA + Kmeans

km = KMeans(n_clusters=3)
km.fit(x_pca)
Y_cluster_kmeans= km.predict(x_pca)
from sklearn import metrics
pca_means_score = metrics.silhouette_score(x_pca, Y_cluster_kmeans)
print('PCA+ Kmeans score is:', pca_means_score)


 # KMeans +PCA + scaling

pca = PCA(3)
x_pcascale = pca.fit_transform(x_scaler)

km = KMeans(n_clusters=3)
km.fit(x_pcascale)
Y_cluster_kmeans= km.predict(x_pcascale)
from sklearn import metrics
pca_means_scale_score = metrics.silhouette_score(x_pcascale, Y_cluster_kmeans)
print('PCA+KMEANS+ Scale score is:', pca_means_scale_score)

#BONUS QUESTION 2:

plt.scatter(x_pca[:, 0], x_pca[:, 1], c = Y_cluster_kmeans, s =50, cmap= 'viridis')
plt.show()
plt.scatter(x_pcascale[:, 0], x_pcascale[:, 1], c = Y_cluster_kmeans, s =50, cmap= 'viridis')
plt.show()