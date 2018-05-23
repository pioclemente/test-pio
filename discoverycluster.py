# -*- coding: utf-8 -*-
import json
import os
from StringIO import StringIO
from pprint import pprint as pp
from time import time
import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.sparse import lil_matrix
import sys
from lcluster.clusterlog import clusterLog as logger
class SingleClusterCenter():
    def __init__(self,dataset,features,vectorizer):
        self.dataset = dataset
        self.features = features
        self.vectorizer = vectorizer
    def _affinityPropagation(self):
        km = AffinityPropagation(damping=.5)
        km.fit(self.features)
        cluster_centers = km.cluster_centers_indices_
        return len(cluster_centers),cluster_centers
    def process(self):
        num_cluster,cluster_centers=self._affinityPropagation()
        big_array = []
        nfeatures = self.features.toarray()
        for x in cluster_centers:
            y = x +1
            big_array.append(nfeatures[x:y][0])
        centroids = np.array(big_array)
        return num_cluster,centroids

class BacthClusterCenter():
    def __init__(self,dataset,features,vectorizer):
        self.dataset = dataset
        self.features = features
        self.vectorizer = vectorizer
    def _affinityPropagation(self,features):
        km = AffinityPropagation(damping=.5)
        km.fit(features)
        cluster_centers = km.cluster_centers_indices_
        return len(cluster_centers),cluster_centers
    def _process(self,features):
        #num_cluster,cluster_centers=self._affinityPropagation(features)
        #return num_cluster,cluster_centers
        num_cluster,cluster_centers=self._affinityPropagation(features)
        """
        big_array = []
        nfeatures = features.toarray()
        for x in cluster_centers:
            y = x +1
            print "ECCO X",x
            big_array.append(nfeatures[x:y][0])
        centroids = np.array(big_array)
        """
        return num_cluster,cluster_centers
        #return num_cluster,big_array
    def _get_sub_f(self,start,end):
        #print "START [%s] END [%s]"%(start,end)
        big_array = []
        doc_map = []
        for x in xrange(start,end):
            y = x +1
            try:
                big_array.append(self.features[x:y].todense())
                doc_map.append(x)
            except:
                continue
        nfeatures=lil_matrix(np.array(big_array)).tocsr()
        return nfeatures,doc_map

    def process(self):
        chunk_size = 2000
        chunk_size = 400
        big_array = []
        idx_centroids = []
        dati_iter = {}
        tot_cluster = 0
        for x in range(0,self.dataset.numDoc(),chunk_size):
            #print "process chunk ....",x
            nfeatures,doc_map = self._get_sub_f(x,x+chunk_size)
            if len(doc_map) == chunk_size:
                num_cluster,tmp_centroids = self._process(nfeatures)
                dati_iter[x] = {'num_cluster':num_cluster,'centroids':tmp_centroids}
        for t in dati_iter:
            tot_cluster += dati_iter[t]['num_cluster']
            for ci in dati_iter[t]['centroids']:
                idx_centroids.append(ci)

        #print "calcolo i centroidi...."
        nfeatures = self.features.toarray()
        for x in idx_centroids:
            y = x +1
            big_array.append(nfeatures[x:y][0])
        centroids = np.array(big_array)
        return tot_cluster,centroids







class ClusterCenter():
    def __init__(self,dataset,features,vectorizer):
        self.dataset = dataset
        self.features = features
        self.vectorizer = vectorizer
        self._num_doc = 400
    def process(self):
        num_docs = self.dataset.numDoc()
        if num_docs < self._num_doc:
            c = SingleClusterCenter(self.dataset,self.features,self.vectorizer)
            num_cluster,centroids = c.process()
            return num_cluster,centroids
        else:
            c = BacthClusterCenter(self.dataset,self.features,self.vectorizer)
            num_cluster,centroids = c.process()
            return num_cluster,centroids

