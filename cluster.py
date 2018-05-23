# -*- coding: utf-8 -*-
import json
import os
from StringIO import StringIO
from pprint import pprint as pp
from time import time
import logging
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans,AffinityPropagation,DBSCAN,SpectralClustering
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial import distance
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import lil_matrix
import sys
import networkx as nx
import operator
from scipy.spatial import distance
import datetime
import shutil
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os import listdir
from os import makedirs
import re
from lcluster.textprofilesignature import textProfileSignature
from lcluster.terms import Terms
from lcluster.clusterlog import clusterLog as logger
from lcluster.clusterutils import clusterUtils
from lunicode import lunicode

THR_SIMILARITY_2DOCS = 0.2

class SuperCluster():
    def __init__(self,clusters,clusters_maps):
        self.clusters = clusters
        self.clusters_maps = clusters_maps
        self.superCluster = {}
        self._used_cluster = []
    def _getClusterFromId(self,id):
            for x in self.clusters:
                if x.id == id:
                    return x
            return False
    def _createSuperCluster(self,idx,clusters_label):
        if len(clusters_label):
            self.superCluster[idx] = {}
            self.superCluster[idx]['type'] = 'similarity'
            self.superCluster[idx]['cluster'] = []
            for clabel in clusters_label:
                self.superCluster[idx]['cluster'].append(self._getClusterFromId(clabel))
                self._used_cluster.append(clabel)

    def fit(self):
        generic_supercluster_idx = len(self.clusters_maps)
        for idx,cmap in enumerate(self.clusters_maps):
            self._createSuperCluster(idx,self.clusters_maps[cmap])
        self.superCluster[generic_supercluster_idx]={}
        self.superCluster[generic_supercluster_idx]['type'] = 'generic'
        self.superCluster[generic_supercluster_idx]['cluster'] = []
        #pp(self._used_cluster)
        for c in self.clusters:
            if c.id not in self._used_cluster:
                #print "C",c.label
                self.superCluster[generic_supercluster_idx]['cluster'].append(c)
        return self.superCluster



class Cluster():
    def __init__(self,label,iterate_number,centroids,silhouette_score,inertia,top_terms=None):
        self.min_cluster_doc = 2
        self.min_clustering_coefficient = 0.5
        self.soglia_perc_prec = 45.0
        self.soglia_perc_first = 19.0

        self.valid_doc_score = {}
        self.invalid_doc_score = {}



        self.label = label
        self.id = label
        self.iterate_number = iterate_number
        self._calcId()


        self.silhouette_score = silhouette_score
        self.inertia = inertia
        self.average_clustering = 0

        self.docid = []
        self.invalid_docid = []
        self.valid_docid = []


        self.valid = False
        self.centroids = None
        self.top_terms = top_terms

        self.features = None
        self.dataset = None
        self.doc_map = None
        self.centroids = centroids
        self.max_top_terms = 20

    def _calcId(self):
        self.id = "%s_%s_%s"%(clusterUtils.getRunDate(),self.iterate_number,self.label)

    def getDocIdx(self,s_docid):
        for idx,docid in enumerate(self.doc_map):
            if docid == s_docid:
                return idx
        return None

    def initDataSetFeautures(self,features,dataset):
        self.features,self.doc_map = self.getFeaturesForCluster(features,self.docid)
        self.dataset = dataset.newDataSetFromIds(self.docid)


    def getTotTerm(self,ret_dict=True):
        inversed_data = self.top_terms
        top_terms={}
        for x,p in enumerate(inversed_data[0]):
            #print "TEXT ",x,p,self.centroids[x]
            if self.centroids[x] > 0:
                top_terms[p]=self.centroids[x]
        if not len(top_terms):
            return []

        top_terms = sorted(top_terms.iteritems(), key=operator.itemgetter(1),reverse=True)
        tmp = []
        for t,s in top_terms:
            tmp.append(u"%s:%s"%(t,s))

        terms = u"|".join(tmp)
        out = Terms.normalizeTerms(terms)
        top_terms = {}
        for o in out:
            top_terms[o['term']] =  o['rank_order']
        top_terms = sorted(top_terms.iteritems(), key=operator.itemgetter(1),reverse=True)

        num_top_terms = min(self.max_top_terms,len(top_terms))
        top_terms =  top_terms[0:num_top_terms]

        return  top_terms


    def getTotTermTTT(self,ret_dict=True):
        if self.centroids is not None and self.vectorized is not None:
            inversed_data = self.vectorized.inverse_transform(self.centroids)

            top_terms={}
            for x,p in enumerate(inversed_data[0]):
                top_terms[p]=self.centroids[x]
            if not len(top_terms):
                return []
            top_terms = sorted(top_terms.iteritems(), key=operator.itemgetter(1),reverse=True)
            #pp(top_terms)
            return top_terms



        else:
            return None

    def toJson(self,dump_body=True,full_data=True,dump_valid_doc_id=True):
      ret = {}
      ret['id'] = self.id
      if full_data:
          ret['valid_docid'] = list()
          ret['docs'] = list()
          for item in self.valid_docid:
                doc = self.dataset.getDoc(self.getDocIdx(item))
                ret['valid_docid'].append(doc.getId())
                ret['docs'].append(doc)
      else:
          ret['docs'] = list()
          for item in self.valid_docid:
                doc = self.dataset.getDoc(self.getDocIdx(item))
                ret['docs'].append(doc.getId())
      if not dump_valid_doc_id:
        del ret['valid_docid']
      return ret      
      

    def dumpTxt(self,fp=None,dump_body=False):
        logger.info("Dump Cluster %s"%self.label)
        s = StringIO()
        s.write(u"*"*100)
        s.write(u"\n")
        s.write(u"Dump Cluster %s - silhouette_score [%0.3f]  inertia [%s] Clustering coefficient G [%s] - NumValidDoc [%s] NumInvalidDoc [%s]\n"%(self.label,self.silhouette_score,self.inertia,self.average_clustering,len(self.valid_docid),len(self.invalid_docid)))

        s.write(u"*"*100)
        s.write(u"\n")
        #pp(dataset.data)
        for item in self.valid_docid:
            doc =  self.dataset.getDoc(self.getDocIdx(item))
            if dump_body:
                s.write(u"%s\n"%(doc.getBody()))
            else:
                s.write(u"%s\n"%(doc.getOrigTitle()))
        s.write(u"\n")
        for item in self.valid_docid:
            doc =  self.dataset.getDoc(self.getDocIdx(item))
            s.write(u"%s\n"%(doc.getUrl()))
        s.write(u"\n")
        s.write("\n")


        """
        s.write("TOP TERM\n")
        top_terms = self.getTotTerm()
        for term in top_terms:
            s.write("Term [%s] Score [%s] \n"%(term[0],term[1]))
        """
        if fp is not None:
            fp.write(s.getvalue().encode("utf-8"))
        else:
            try:
                print s.getvalue().encode("utf-8")
            except:
                print "errore su questo cluster"

            #pp(top_terms)

    def to_xml(self,root,doc,dump_body=False):
        field_elem = doc.createElement('head')

        elem = doc.createElement('field')
        elem.setAttribute('name', u"iterator")
        elem.appendChild(doc.createTextNode(u"%s"%self.iterate_number))
        field_elem.appendChild(elem)


        elem = doc.createElement('field')
        elem.setAttribute('name', u"id")
        elem.appendChild(doc.createTextNode(u"%s"%self.id))
        field_elem.appendChild(elem)


        elem = doc.createElement('field')
        elem.setAttribute('name', u"label")
        elem.appendChild(doc.createTextNode(u"%s"%self.label))
        field_elem.appendChild(elem)

        elem = doc.createElement('field')
        elem.setAttribute('name', u"silhouette_score")
        elem.appendChild(doc.createTextNode(u"%s"%self.silhouette_score))
        field_elem.appendChild(elem)

        elem = doc.createElement('field')
        elem.setAttribute('name', u"inertia")
        elem.appendChild(doc.createTextNode(u"%s"%self.inertia))
        field_elem.appendChild(elem)

        elem = doc.createElement('field')
        elem.setAttribute('name', u"average_clustering")
        elem.appendChild(doc.createTextNode(u"%s"%self.average_clustering))
        field_elem.appendChild(elem)

        elem = doc.createElement('field')
        elem.setAttribute('name', u"valid_docid")
        elem.appendChild(doc.createTextNode(u"%s"%len(self.valid_docid)))
        field_elem.appendChild(elem)

        elem = doc.createElement('field')
        elem.setAttribute('name', u"invalid_docid")
        elem.appendChild(doc.createTextNode(u"%s"%len(self.invalid_docid)))
        field_elem.appendChild(elem)



        root.appendChild(field_elem)
        field_elem = doc.createElement('docs')
        #print "KKKKKKKKKKKKK"
        #pp(self.valid_docid)
        for item in self.valid_docid:
            #print "IL valore di ITEM ---- ",item
            clusterDoc = self.dataset.getDoc(self.getDocIdx(item))
            #print "CIAO ",clusterDoc.pid
            field_elem.appendChild(clusterDoc.to_xml(root,doc,True,dump_body))
        root.appendChild(field_elem)

        for item in self.invalid_docid:
            clusterDoc = self.dataset.getDoc(self.getDocIdx(item))
            field_elem.appendChild(clusterDoc.to_xml(root,doc,False,dump_body))
        root.appendChild(field_elem)


        top_terms = self.getTotTerm()
        field_elem = doc.createElement('terms')
        for term in top_terms:
            elem = doc.createElement('term')
            elem.setAttribute('score', u"%s"%term[1])
            elem.appendChild(doc.createTextNode(term[0]))
            field_elem.appendChild(elem)
        root.appendChild(field_elem)
        return True



    def add(self,docid):
        self.docid.append(docid)
    def _get_doc_similarity(self,doc,idx_s,idx_d):
        cosine_similarities = linear_kernel(self.features[idx_s:idx_d], self.features).flatten()
        related_docs_indices = cosine_similarities.argsort()
        related_docs_indices1 = list(reversed(related_docs_indices))
        tmp = {}
        s = 20.0
        prec_score = 1.0
        for idx,i in enumerate(related_docs_indices1):
            score = cosine_similarities[related_docs_indices1][idx]
            perc = float(score/prec_score * 100)
            if perc > s:
                tmp[i]={}
                tmp[i]= score
        return tmp


    def _get_2doc_similarity(self,idx_s,idx_d):
	cosine_similarities = linear_kernel(self.features[idx_s:idx_d], self.features).flatten()
	related_docs_indices = cosine_similarities.argsort()
	related_docs_indices1 = list(reversed(related_docs_indices))
	return cosine_similarities[related_docs_indices1].tolist()[1]
    def _validateCluster2Doc(self):
        idx_s = self.docid[0]
        idx_d = self.docid[1]
        doc_sim = self._get_2doc_similarity(0,1)
        if doc_sim >= THR_SIMILARITY_2DOCS:
            self.valid = True
            self.valid_docid = self.docid
        else:
            self.valid = False
            self.invalid_docid = self.docid
            self.valid_docid = []
    def getFeaturesForCluster(self,features,docid):
        big_array = []
        doc_map = []
        for x in docid:
            y = x +1
            big_array.append(features[x:y].todense())
            doc_map.append(x)
        nfeatures=lil_matrix(np.array(big_array)).tocsr()
        return nfeatures,doc_map
    def validateDocCluster(self):
        G=nx.Graph()
        average_clustering = 0
        for pidx,doc in enumerate(self.docid):
            idx_s = pidx
            idx_d = pidx + 1
            doc_sim = self._get_doc_similarity(doc,idx_s,idx_d)
            G.add_node(doc,label=doc)
            for doc1 in doc_sim:
                if doc != self.doc_map[doc1]:
                    G.add_edge(doc,self.doc_map[doc1],weight=doc_sim[doc1])
        self.average_clustering = nx.average_clustering(G)

        logger.warning("Clustering coefficient G %s "%self.average_clustering)
        if self.average_clustering > self.min_clustering_coefficient:
            try:
                res = nx.pagerank(G)
                sorted_x = sorted(res.iteritems(), key=operator.itemgetter(1),reverse=True)
                pred_score =sorted_x[0][1]
                first_score =sorted_x[0][1]
                for idx,tmp in enumerate(sorted_x):
                    doc_id = tmp[0]
                    score = tmp[1]
                    perc_prec = int(score/pred_score * 100)
                    perc_first = int(score/first_score * 100)
                    doc_dataset = self.dataset.getDoc(self.getDocIdx(doc_id))
                    doc_dataset.setPid(doc_id)
                    doc_dataset.setDocScore(score,pred_score,perc_prec,perc_first)
                    if perc_prec >= self.soglia_perc_prec and perc_first >= self.soglia_perc_first:
                        self.valid_docid.append(doc_id)
                        self.valid_doc_score[doc_id] = score
                    else:
                        self.invalid_doc_score[doc_id] = score
                        self.invalid_docid.append(doc_id)
                    pred_score = sorted_x[idx][1]

            except:
                import traceback
                traceback.print_exc()
        else:
            logger.info("Invalid cluster [%s]  "%self.label)


    def validateCluster(self):
        #print "validateCluster ITERATOR [%s] LABEL [%s] NUM DOC [%s]"%(self.iterate_number,self.label,len(self.docid))
        logger.info("Start Validate Cluster [%s] "%self.label)

        if len(self.docid) == 2:
            return self._validateCluster2Doc()
        self.validateDocCluster()
        #ultima validazione basata sul numero dei documenti presenti nel cluster
        if len(self.valid_docid) >= self.min_cluster_doc:
            self.valid = True
            #self.invalid_docid = list(set(self.docid)-set(self.valid_docid))
        else:
            logger.info("Cluster [%s] not Valid!!"%self.label)
            self.valid = False
            self.invalid_docid = self.docid
            self.valid_docid = []
        logger.debug("Cluster [%s] INFO: Valid Doc [%s] Invalid Doc [%s] average_clustering %s"%(self.label,len(self.valid_docid),len(self.invalid_docid),self.average_clustering))

