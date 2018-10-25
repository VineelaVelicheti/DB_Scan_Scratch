
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import sparse
train_data = pd.read_csv("/Users/vineevineela/Documents/Semester-2/CMPE-255/Assignments/PR3/train.dat",header=None)
train = train_data.values.tolist()


# In[2]:


train_int=[]
ind=[]
val=[]
ptr=[]
ptr.append(0)

for l in train:
    str=l[0].split()
    train_int.append(str)
    
for sl in train_int:
    ptr.append(len(sl)/2)
    for i in range(len(sl)):
        if i % 2 != 0:
            val.append(int(sl[i]))
        else:
            ind.append(int(sl[i]))
    
for i in range(len(ptr)):
    if(i!=0):
        ptr[i]=ptr[i]+ptr[i-1]

ind=np.asarray(ind)
val=np.asarray(val)
ptr=np.asarray(ptr)
nrows=len(train)
ncols=max(ind)

mat = sparse.csr_matrix((val, ind, ptr), shape=(nrows, ncols))


# In[5]:


# scale matrix and normalize its rows
from collections import defaultdict

def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat
    
mat_idf = csr_idf(mat, copy=True)


# In[6]:


mat = mat_idf


# In[7]:


from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold()
variance_selector.fit(mat)
mat = variance_selector.transform(mat)


# In[8]:


from sklearn.neighbors import NearestNeighbors
min_points=[3,5,7,9,11,13,15,17,19,21]
distances=[]
indices=[]
for i in min_points:
    n = NearestNeighbors(n_neighbors=i, algorithm='auto').fit(mat)
    dist, ind = n.kneighbors(mat)
    distances.append(dist)
    indices.append(ind)


# In[9]:


ilist = indices[0][:,0].tolist()
distancelist=[]
pos = 0
for i in min_points:
    dlist = distances[pos][:,i-1].tolist()
    pos = pos+1
    dlist=sorted(dlist)
    distancelist.append(dlist)


# In[10]:


import matplotlib 
from matplotlib import pyplot as plt

x = ilist
for i in range(len(distancelist)):
    #plt.figure(figsize=(10,10))
    y = distancelist[i]
    plt.plot(x,y)
    plt.grid()
    #plt.yticks(np.arange(0,1200,50))
    plt.show()


# In[ ]:


import sklearn.metrics.pairwise

def distance_calc(mat,nrows):
    distance_int = sklearn.metrics.pairwise.pairwise_distances(mat,mat)
    distance_ext = distance_int.tolist()
    return distance_ext

# custom proximity function
#def euclidean_distance(mat1,mat2):
#    euclidean_dist = np.linalg.norm(mat1 - mat2)
#    return euclidean_dist

#def distance_calc(mat,nrows):
#    distance_int = [] 
#    distance_ext = []
#    pts_count = nrows
#    mat_dense = mat.todense()
#    for i in range(pts_count):
#        for j in range(pts_count): 
#            distance = euclidean_distance(mat_dense[i],mat_dense[j])
#            distance_int.append(distance)
#        distance_ext.append(distance_int)
#        distance_int = []
                        
#    return distance_ext

def points_classifier(distance_ext,nrows,eps,min_pts):
    points_classification = []
    neighbor_count=[]
    for i in range(nrows):
        points_classification.append('N')   
    for i in distance_ext:
        count = 0
        for j in i:
            if(j < eps):
                count = count + 1
        neighbor_count.append(count)
    for i in range(nrows):
        if(neighbor_count[i] > min_pts):
            points_classification[i] = 'C'
    for i in range(nrows):
        if(points_classification[i] == 'N'):
            for k in range(len(distance_ext[i])):
                if ((distance_ext[i][k] < eps) and (points_classification[k] == 'C' )):
                    points_classification[i] = 'B'
                    break
    return points_classification


# In[ ]:


def dbscan(data, eps=150, min_pts=15):
    nrows = data.shape[0]
    ncols = data.shape[1]
    distance_list =  []
    connected_points=[]
    clusters_list=[]
    noise_cluster=[]
    cluster_point_index=[]
    distance_ext = distance_calc(data,nrows)
    points_classification = points_classifier(distance_ext,nrows,eps,min_pts)
    for i in range(nrows):
        if(i not in connected_points):
            if(points_classification[i]=='C'):
                cluster=[]
                connected_points.append(i)
                clusterformation(i,points_classification,distance_ext,cluster,clusters_list,connected_points,nrows,eps)
                clusters_list.append(cluster)
  
    
    for i in range(nrows):
        if(i not in connected_points):
            if(points_classification[i]=='B'):
                connected_points.append(i)
                distance_sort = sorted(distance_ext[i]) 
                for index in range(len(distance_sort)):
                    if(distance_sort[index]!=0):
                        nearest_neighbour = distance_sort[index]
                        nn_index = distance_ext[i].index(nearest_neighbour)
                        if(points_classification[nn_index]=='C'):
                            break
                for c in clusters_list:
                    for k in c:
                        if(k == nn_index):
                            c.append(i)
  
                         
    for i in range(nrows):
        if(i not in connected_points):
            if(points_classification[i]=='N'):
                connected_points.append(i)
                noise_cluster.append(i)
    clusters_list.append(noise_cluster)
    
   
    
    for i in range(nrows):
        for j in range(len(clusters_list)):
            for k in clusters_list[j]:
                if(k==i):
                    cluster_point_index.append(j)
                    break
               
    return cluster_point_index       
                
                  
                            
def clusterformation(i,points_classification,distance_ext,cluster,clusters_list,connected_points,nrows,eps):
    cluster.append(i)
    for j in range(nrows):
        if (distance_ext[i][j] < eps and j not in connected_points and points_classification[j]=='C'):
            cluster.append(j)
            connected_points.append(j)
            for k in range(nrows):
                if (distance_ext[j][k] < eps and k not in connected_points and points_classification[k]=='C'):
                    connected_points.append(k)
                    cluster.append(k)
              
    return
       


# In[ ]:


cluster_points = dbscan(mat,97,5)


# In[ ]:


f = open("/Users/vineevineela/Desktop/dbscan17.txt", "w")
for item in cluster_points:
    f.write("%s\n" % item)
f.close()


# In[ ]:


from sklearn.metrics import silhouette_score
eps_list=[80,97,150]
cluster_pts_list = []
silhouette_score_lst=[]

for i in eps_list:
    for j in min_points:
        cluster_points=dbscan(mat,i,j)
        cluster_pts_list.append(cluster_points)
        silhouette_avg = silhouette_score(mat, cluster_points)
        silhouette_score_lst.append(silhouette_avg)
        


# In[ ]:


cluster_count=[]
for i in cluster_pts_list:
    cluster_count.append(max(i)+1)


# In[ ]:


x = cluster_count[0:10]
y = silhouette_score_lst[0:10]
plt.plot(x,y)
plt.grid()
plt.yticks(np.arange(0,1,0.05))
plt.show()

x = cluster_count[10:20]
y = silhouette_score_lst[10:20]
plt.plot(x,y)
plt.grid()
plt.yticks(np.arange(0,1,0.05))
plt.show()


x = cluster_count[20:30]
y = silhouette_score_lst[20:30]
plt.plot(x,y)
plt.grid()
plt.yticks(np.arange(0,1,0.05))
plt.show()

