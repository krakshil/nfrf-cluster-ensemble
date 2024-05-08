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
    def __init__(self, train_docs:list, test_docs:list, alpha_1=0.6, min_alpha_1=0.1, alpha_diff_1=0.05, alpha_2=0.5, final_k=12, members_dir="", save_dir=""):
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

        self.membership_matrices = {}
        self.alpha_1 = alpha_1
        self.min_alpha_1 = min_alpha_1
        self.alpha_diff_1 = alpha_diff_1
        self.alpha_2 = alpha_2
        self.final_k = final_k
        # self.embedding_config = embedding_config
        # self.embedding_dict = {key: {} for key in embedding_config.keys()}
        
        # # self.clustering_dict = {key: {} for key in clustering_config.keys()}
        # self.vectorizer_config = vectorizer_config
        # self.ctfidf_config = ctfidf_config
        # self.representation_config = representation_config
        # self.evaluation_dict = {}

        # self.save_dir = save_dir
    
    
    ## calculate partial-membership matrices
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
            doc_info = topic_model.get_document_info(self.train_docs)

            if topic_model.get_topic_info().shape[0] > 2:
                # train_preds, train_probs = topic_model.transform(self.train_docs, train_features)
                train_preds, train_probs = doc_info["Topic"].tolist(), None
                test_preds, test_probs = topic_model.transform(self.test_docs, test_features)

                if test_probs is not None:
                    columns = ["predictions"] + list(range(0,test_probs.shape[1]))
                    test_vals = np.concatenate([np.array(test_preds)[:,None], test_probs], axis=1)
                    train_df = pd.DataFrame(train_preds, columns=columns[0:1])
                    test_df = pd.DataFrame(test_vals, columns=columns)

                else:
                    columns = ["predictions"]
                    test_vals = np.array(test_preds)
                    train_df = pd.DataFrame(train_preds, columns=columns)
                    test_df = pd.DataFrame(test_vals, columns=columns)
                    
                topic_info.to_csv(os.path.join(info_path, variant_name + ".csv"), index=False)
                train_df.to_csv(os.path.join(preds_path, "train", variant_name + ".csv"), index=False)
                test_df.to_csv(os.path.join(preds_path, "test", variant_name + ".csv"), index=False)
        
        except Exception as e:
            print("[INFO] (" + embedding_model_name + ", " + cluster_model_name + ", {" + variant_name + "})\tThe error is: ",e)


    ## Create save directory heirarchy
    def create_save_dir(self):
        self.save_path = os.path.join(self.save_dir, "cluster_ensemble")
        self.final_path = os.path.join(self.save_path, "final_merge")
        self.partial_path = os.path.join(self.save_path, "partial_membership_matrix")
        self.complete_path = os.path.join(self.save_path, "complete_membership_matrix")
        os.makedirs(self.partial_path, exist_ok=True)
        os.makedirs(self.complete_path, exist_ok=True)


    ## Combine all the partial matrix of memberships
    def combine_partial_membership_matrix(self, num_min_clusters=7, num_max_clusters=15, load=False):
        
        if not load:
            print("[INFO] Loading Partial Membership Matrix for all members...")
            embedding_names = os.listdir(self.partial_path)
            

            for embedding_model in embedding_names:
                
                print("[INFO] Model: " + embedding_model + "...")

                membership_matrix = []
                cluster_index_dict = {}
                cluster_idx = 0
                
                model_path = os.path.join(self.complete_path, embedding_model)
                os.makedirs(model_path, exist_ok=True)

                clustering_names = os.listdir(os.path.join(self.partial_path, embedding_model))
                
                for cluster_model in clustering_names:
                    cluster_model_path = os.path.join(self.partial_path, embedding_model, cluster_model)
                    cluster_variant_names = os.listdir(os.path.join(cluster_model_path, "info"))
                    
                    info_path = os.path.join(cluster_model_path, "info")
                    train_path = os.path.join(cluster_model_path, "preds", "train")
                    test_path = os.path.join(cluster_model_path, "preds", "test")

                    for model_variant in tqdm(cluster_variant_names, desc=(embedding_model + ", " + cluster_model)):
                        # self.save_matrix_from_model(train_features=train_features, test_features=test_features, embedding_model_name=embedding_model, cluster_model_name=cluster_model, variant_name=model_variant)
                        clusters = pd.read_csv(os.path.join(info_path, model_variant))["Topic"].values
                        cluster_preds = pd.read_csv(os.path.join(train_path, model_variant))["predictions"]
                        
                        if num_max_clusters >= clusters.shape[0] >= num_min_clusters:
                            one_hot_preds = pd.get_dummies(cluster_preds, prefix=str(cluster_idx))
                            membership_matrix.append(one_hot_preds)
                            cluster_index_dict[cluster_idx] = [embedding_model, cluster_model, model_variant]
                            cluster_idx += 1
            
                with open(os.path.join(model_path, "meta_info.json"), "w") as f:
                    json.dump(cluster_index_dict, f)

                membership_matrix = pd.concat(membership_matrix, axis=1)
                membership_matrix.to_csv(os.path.join(model_path, "matrix.csv"), index=False)
                self.membership_matrices[embedding_model] = membership_matrix
            print("[INFO] Complete Membership matrix saved.")

        else:
            print("[INFO] Loading Complete Membership Matrix for all embedding models...")
            embedding_names = os.listdir(self.complete_path)

            for embedding_model in embedding_names:
                print("[INFO] Model: " + embedding_model + "...")
                self.membership_matrices[embedding_model] = pd.read_csv(os.path.join(self.complete_path, embedding_model ,"matrix.csv"))
            print("[INFO] Complete Membership matrix loaded.")


    ## Consensus function
    def consensus_fn(self):
        
        def cluster_similarity(c1, c2):
            n = c1.shape[0]
            c1_sum = c1.sum()
            c2_sum = c2.sum()
            intersection = (c1*c2).sum()
            c1c2 = c1_sum*c2_sum
            return (intersection - (c1c2/n))/(np.sqrt(c1c2*(n-c1_sum)*(n-c2_sum))/n)
        
        def member_similarity(matrix, clusters_dict):
            membership_df = []
            columns = []
            for combination_name, cluster_combination in clusters_dict.items():
                columns.append(combination_name)
                membership_df.append(matrix[cluster_combination].sum(axis=1))
            
            membership_df = pd.DataFrame(membership_df, columns=columns)
            membership_df = membership_df/membership_df.max(axis=1)
            return membership_df
        
        def get_cluster_similarity(matrix, iter=1):
            columns = matrix.columns
            matrix_vals = matrix.values
            columns_cluster = list(map(lambda x: int(x.split("_")[0]), columns)) if iter==1 else None
            current_k = matrix.shape[1]
            c_s = np.zeros((current_k, current_k))+(-2)
            
            if iter==1:
                with tqdm(total=((current_k*(current_k-1)/2)), desc="Iteration - " + str(iter) + ", C_S Progress") as pbar:
                    for i in range(current_k):
                        for j in range(i+1,current_k):
                            if (columns_cluster[i] != columns_cluster[j]):
                                c_s[i,j] = cluster_similarity(matrix_vals[:, i], matrix_vals[:, j])
                            pbar.update(1)
            else:
                with tqdm(total=((current_k*(current_k-1)/2)), desc="Iteration - " + str(iter) + ", C_S Progress") as pbar:
                    for i in range(current_k):
                        for j in range(i+1,current_k):
                            if (i!=j):
                                c_s[i,j] = cluster_similarity(matrix_vals[:, i], matrix_vals[:, j])
                            pbar.update(1)
            return c_s

        def get_merge_dict(c_s, columns, alpha_01, iter=1):
            cluster_merge = {}
            clusters_visited = set()
            flag = False
            new_cluster_idx = 0

            for i in tqdm(range(len(columns)), desc="Iteration - " + str(iter) + ", Merge Progress"):
                if columns[i] in clusters_visited:    
                    continue
                else:
                    flag = True
                    key = "cluster_" + str(new_cluster_idx)
                    cluster_merge[key] = [columns[i]]
                    clusters_visited.add(columns[i])
                    current_idx = i
                    while flag:
                        next_idx = c_s[current_idx].argmax()
                        if columns[next_idx] in clusters_visited:
                            flag = False
                        else:
                            if c_s[current_idx, next_idx] >= alpha_01:
                                cluster_merge[key].append(columns[next_idx])
                                clusters_visited.add(columns[next_idx])
                                current_idx = next_idx
                            else:
                                flag = False
                    new_cluster_idx += 1
            return cluster_merge

        def merge_clusters(merge_dict, matrix, iter=1):
            new_matrix = {}
            for combination_name, cluster_combination in merge_dict[iter].items():        
                new_matrix[combination_name] = (matrix[cluster_combination].sum(axis=1).apply(lambda x: 0 if x<1 else 1))
            new_matrix = pd.DataFrame(new_matrix)
            return new_matrix

        def update_merge_dict(iter_dict, merge_dict, iter):
            updated_dict = {}
            for key, value in iter_dict.items():
                updated_dict[key] = []
                for val in value:
                    updated_dict[key] += merge_dict[iter-1][val]
            return updated_dict
        
        def save_ensemble_results(path, matrix, merge_dict):
            os.makedirs(path, exist_ok=True)
            matrix.to_csv(os.path.join(path, "ensemble_soft_clusters.csv"), index=False)
            with open(os.path.join(path, "ensemble_merge_info.json"), "w") as f:
                json.dump(merge_dict, f)
        

        for embedding_model in list(self.membership_matrices.keys())[1:]:
            print("[INFO] Embedding Model: " + str(embedding_model) + "...")
            matrix = self.membership_matrices[embedding_model].copy()
            matrix = matrix[matrix.columns[:1000]]
            columns = matrix.columns
            alpha_01 = self.alpha_1
            merge_dict = {}

            c_s = get_cluster_similarity(matrix, iter=1)
            merge_dict[1] = get_merge_dict(c_s, columns, alpha_01, iter=1)
            matrix_interim = merge_clusters(merge_dict, matrix)
            current_k = matrix_interim.shape[1]

            flag, alpha_flag, iter = True, False, 2
            while flag:
                if (current_k <= self.final_k):
                    flag = False
                else:
                    if alpha_flag:
                        flag = False
                    
                    c_s = get_cluster_similarity(matrix_interim, iter=iter)
                    merge_dict_interim = get_merge_dict(c_s, matrix_interim.columns, alpha_01, iter=iter)
                    
                    if len(merge_dict_interim) >= current_k:
                        if (alpha_01) <= self.min_alpha_1:
                            alpha_flag = True
                        else:
                            alpha_01 = alpha_01 - self.alpha_diff_1
                            merge_dict_interim = get_merge_dict(c_s, matrix_interim.columns, alpha_01, iter=iter)

                    if not alpha_flag:    
                        merge_dict[iter] = update_merge_dict(merge_dict_interim, merge_dict, iter)
                        matrix_interim = merge_clusters(merge_dict, matrix, iter)
                        current_k = matrix_interim.shape[1]
                        iter += 1

            save_path = os.path.join(self.final_path, embedding_model)
            save_ensemble_results(path=save_path, matrix=matrix_interim, merge_dict=merge_dict)
            print("[INFO] Saved soft ensemble clusters and merge dict.")