## import packages
import os
import string
import glob
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score





## Cluster ensemble
class ClusterEnsemble:
    def __init__(self, train_docs:list, test_docs:list, members_dir="", save_dir=""):
        '''
        membership dict: dict (contains metadata of the types of clustering models with constructors and hyper-parameters)
            {"hdbscan": {"constructor": model_constructor - callable, "type":"model_type", "params_dict":{"min_cluster_size":[15,20,25], "min_samples":[5,10,15], "cluster_selection_epsilon":[0.4, 0.5, 0.7]}}}
        '''
        # self.members_dict = self.get_members_specs(membership_dict)
        self.members_dir = members_dir
        self.save_dir = save_dir
        self.create_save_dir()

        self.train_docs = train_docs
        self.test_docs = test_docs

        self.membership_matrix = None

        # self.embedding_config = embedding_config
        # self.embedding_dict = {key: {} for key in embedding_config.keys()}
        
        # # self.clustering_dict = {key: {} for key in clustering_config.keys()}
        # self.vectorizer_config = vectorizer_config
        # self.ctfidf_config = ctfidf_config
        # self.representation_config = representation_config
        # self.evaluation_dict = {}

        # self.save_dir = save_dir
    
    
    # calculate partial-membership matrices
    def get_partial_membership(self):
        
        print("[INFO] Creating Partial Membership Matrix for all members...")
        embedding_names = list(filter(lambda x: True if ("." not in x) else False, os.listdir(self.members_dir)))
        
        for embedding_model in embedding_names:
            print("[INFO] Model: " + embedding_model + "...")
            with np.load(os.path.join(self.members_dir, embedding_model, "embeddings.npz")) as f:
                train_features = f["train_features"]
                test_features = f["test_features"]
                train_embeddings = f["train_embeddings"]
                test_embeddings = f["test_embeddings"]
            
            clustering_names = list(filter(lambda x: True if ("." not in x) else False, os.listdir(os.path.join(self.members_dir, embedding_model))))
            
            for cluster_model in clustering_names:
                cluster_variant_names = os.listdir(os.path.join(self.members_dir, embedding_model, cluster_model))
                
                for model_variant in tqdm(cluster_variant_names, desc=(embedding_model + ", " + cluster_model)):
                    self.save_matrix_from_model(train_features=train_features, test_features=test_features, embedding_model_name=embedding_model, cluster_model_name=cluster_model, variant_name=model_variant)
        
        print("[INFO] All partial membership matrix saved.")

    
    ## Load model, get predictions and save to csv
    def save_matrix_from_model(self, train_features, test_features, embedding_model_name, cluster_model_name, variant_name):
        save_path = os.path.join(self.partial_path, embedding_model_name, cluster_model_name)
        info_path = os.path.join(save_path, "info")
        preds_path = os.path.join(save_path, "preds")
        os.makedirs(info_path, exist_ok=True)
        os.makedirs(os.path.join(preds_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(preds_path, "test"), exist_ok=True)

        try:
            topic_model = BERTopic.load(os.path.join(self.members_dir, embedding_model_name, cluster_model_name, variant_name))
            topic_info = topic_model.get_topic_info()

            if topic_model.get_topic_info().shape[0] > 2:
                train_preds, train_probs = topic_model.transform(self.train_docs, train_features)
                test_preds, test_probs = topic_model.transform(self.test_docs, test_features)

                if train_probs is not None:
                    columns = ["predictions"] + list(range(0,train_probs.shape[1]))
                    train_vals = np.concatenate([np.array(train_preds)[:,None], train_probs], axis=1)
                    test_vals = np.concatenate([np.array(test_preds)[:,None], test_probs], axis=1)
                    train_df = pd.DataFrame(train_vals, columns=columns)
                    test_df = pd.DataFrame(test_vals, columns=columns)

                else:
                    columns = ["predictions"]
                    train_vals = np.array(train_preds)
                    test_vals = np.array(test_preds)
                    train_df = pd.DataFrame(train_vals, columns=columns)
                    test_df = pd.DataFrame(test_vals, columns=columns)
                    
                topic_info.to_csv(os.path.join(info_path, variant_name + ".csv"), index=False)
                train_df.to_csv(os.path.join(preds_path, "train", variant_name + ".csv"), index=False)
                test_df.to_csv(os.path.join(preds_path, "test", variant_name + ".csv"), index=False)
        
        except Exception as e:
            print("[INFO] (" + embedding_model_name + ", " + cluster_model_name + ", {" + variant_name + "})\tThe error is: ",e)


    ## Create save directory heirarchy
    def create_save_dir(self):
        self.save_path = os.path.join(self.save_dir, "cluster_ensemble")
        self.partial_path = os.path.join(self.save_path, "partial_membership_matrix")
        os.makedirs(self.partial_path, exist_ok=True)


    ## Combine all the partial matrix of memberships
    def combine_partial_membership_matrix(self, num_min_clusters=5, num_max_clusters=25):
        
        print("[INFO] Loading Partial Membership Matrix for all members...")
        embedding_names = os.listdir(self.partial_path)
        
        membership_matrix = []
        cluster_index_dict = {}
        cluster_idx = 0

        for embedding_model in embedding_names:
            
            print("[INFO] Model: " + embedding_model + "...")
            clustering_names = os.listdir(os.path.join(self.partial_path, embedding_model))
            
            for cluster_model in clustering_names:
                cluster_model_path = os.path.join(self.partial_path, embedding_model, cluster_model)
                cluster_variant_names = os.listdir(os.path.join(cluster_model_path, "info"))
                
                info_path = os.path.join(cluster_model_path, "info")
                train_path = os.path.join(cluster_model_path, "train")
                test_path = os.path.join(cluster_model_path, "test")

                for model_variant in tqdm(cluster_variant_names, desc=(embedding_model + ", " + cluster_model)):
                    # self.save_matrix_from_model(train_features=train_features, test_features=test_features, embedding_model_name=embedding_model, cluster_model_name=cluster_model, variant_name=model_variant)
                    clusters = pd.read_csv(os.path.join(info_path, model_variant))["Topic"].values
                    cluster_preds = pd.read_csv(os.path.join(train_path, model_variant))["predictions"]
                    
                    if num_max_clusters >= clusters.shape[0] >= num_min_clusters:
                        one_hot_preds = pd.get_dummies(cluster_preds["predictions"], prefix=str(cluster_idx))
                        membership_matrix.append(one_hot_preds)
                        cluster_index_dict[cluster_idx] = [embedding_model, cluster_model, model_variant]
                        cluster_idx += 1
        
        with open(os.path.join(self.complete_path, " meta_info.json"), "w") as f:
            json.dump(cluster_index_dict, f)

        membership_matrix = pd.concat(membership_matrix, axis=1)
        membership_matrix.to_csv(os.path.join(self.complete_path, "matrix.csv"), index=False)
        
        print("[INFO] Complete Membership matrix saved.")