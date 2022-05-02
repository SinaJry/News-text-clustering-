from ast import literal_eval
from hazm import *
from hazm import stopwords_list
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from functools import lru_cache
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3d
from operator import itemgetter
import pickle
import pandas as pd 
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples , silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
tqdm.pandas()

## Loading Data and stopwords
path = r'\news.csv'
data = pd.read_csv(path)

sw_path = '\final_stopwords'

with open(sw_path) as f:
    stopwords = pickle.load(f)


## Cleaning Data

normalizer =Normalizer(remove_extra_spaces=True, persian_style=True, persian_numbers=True, remove_diacritics=True, affix_spacing=True, token_based=True, punctuation_spacing=True)

lemmatize = lru_cache(maxsize=500000)(Lemmatizer().lemmatize)

def clean(text,stopwords):
    Text = normalizer.normalize(text)
    Text= re.sub(r'[^\w\s]','',Text)
    Text = re.sub(r'[^\D]',"",Text)
    Text = re.sub(r'([a-zA-z][\w]+)','',Text)
    Text = re.sub(r'\s*[A-Za-z]+\b',"",Text)
    Text = re.sub(r'\[^A-Za-z]',"",Text)
    Text = re.sub('(_[\w]+)',"",Text)
    
    tokens = [lemmatize(i) for i in word_tokenize(Text) if i not in stopwords]
    return tokens

data.drop_duplicates(subset = 'text',inplace = True)
data.dropna(inplace = True)

data['Final'] = data[['title','desc','text']].apply(lambda x : '|'.join(x),axis=1)

data['clean'] = data['Final'].progress_apply(lambda x :clean(x,stopwords))


#Remove duplicated values after preprocessing

_,idx = np.unique(data['clean'],return_index = True)
data = data.iloc[idx,:]

# Remove values with less than 100 words (commercial or incomplete news)
df=data.loc[data.Final.map(lambda x :len(x) > 100),['Final','clean']]


mylist = list()
for i,j in tqdm(df.iterrows()):
    sent = j['clean']
    mylist.append(sent)


#Loading Fasttext pretrained word2vec model
model_path = r'D:\Fasttext\cc.fa.300.vec'
model = KeyedVectors.load_word2vec_format(model_path)

#Testing model by giving words
model.most_similar('فوتبال')

#Generating Vectors for list of documents (list of cleaned tokens)
def vectorize(list_of_docs, model):

    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.wv.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features
    


#Generating clusters and sillhouete score by kmeans alg (MiniBatchKMeans could be used too)
def kmeans_clusters(
	X, 
    k,  
    print_silhouette_values, 
):

    km = KMeans(init='k-means++',max_iter =10000,n_clusters=k).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print("Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

vectorized_docs = vectorize(mylist, model=model)
#choosing right number of clusters w.r.t scores

clustering, cluster_labels = kmeans_clusters(
	X=vectorized_docs,
    k=5,
    print_silhouette_values=True,
)

k=5

### Printing Most representative terms per cluster based on kmeans centeroid
for i in range(k):
    tokens_per_cluster = ""
    most_representative = model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=10)
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    print(f"Cluster {i}: {tokens_per_cluster}")

###Reducing data with pca



def pca_reduction(feature_matrix , method='exact'):
    pca = PCA(n_components=2, random_state=0)

    feature_matrix = pca.fit_transform(feature_matrix)
    return feature_matrix

reduced_data=pca_reduction(vectorized_docs)

### Trying model with dimension reduction algs
wcss = []
for i in range(2,11):
    print(f"k:{i}")
    print("*"*50)
    clustering, cluster_labels = kmeans_clusters(
        	X=reduced_data,
            k=i,
            print_silhouette_values=True,
        )
    wcss.append(clustering.inertia_)

### plotting withing clusters samples distance
plt.figure(figsize = (10,10))
plt.plot(range(2,11),wcss,marker = '*', linestyle = '--')
plt.xlabel('Numbers of K clusters')
plt.ylabel('Within cluster sum of square')
plt.title('K-Means with PCA')
plt.show()


#plotting clusters
labels = clustering.labels_.tolist()
plt.figure()
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080",'#800000']
color = [label1[i] for i in labels]
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=color)

### 3D plot
fig = plt.figure(figsize = (10,10))
ax = Axes3d(fig , rect = [0,0,0.95,1] , elev = 48 , azim = 134)
ax.scatter(reduced_data[:,0] , reduced_data[:,1] , reduced_data[:,2] , c= df['clusters'] , cmap = 'rainbow')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_facecolor('white')
plt.title('3d Plot of Clusters')



df['clusters'] = cluster_labels

cluster_groups = (df[['Final', 'clusters']]
                  .sort_values(by=['clusters'], 
                               ascending=False)
                  .groupby('clusters').head(5))

for cluster_num in range(5):
    news = cluster_groups[cluster_groups['clusters'] == cluster_num]['Final'].values.tolist()
    print('CLUSTER #'+str(cluster_num+1))
    print(news[0])
    print('-'*80)
    print(news[1])
    print('-'*80)
    print(news[2])










