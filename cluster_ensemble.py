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
    def __init__(self, train_docs:list, test_docs:list, alpha_1=0.8, min_alpha_1=0.5, alpha_diff_1=0.025, alpha_2=0.7, final_k=12, min_samples=5, members_dir="", save_dir=""):
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
        self.min_samples = min_samples
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
    def combine_partial_membership_matrix(self, ver=0, n_threshold=100, num_min_clusters=7, num_max_clusters=15, load=False):
        
        def normalize(value, min_val, max_val):
            return ((value - min_val) / (max_val - min_val))
        
        if not load:
            print("[INFO] Loading Partial Membership Matrix for all members...")
            embedding_names = os.listdir(self.partial_path)
            
            #### n-best
            clustering_names = list(filter(lambda x: False if "." in x else True, os.listdir(os.path.join(self.partial_path, embedding_names[0]))))

            scores_dict = dict()
            normalized_dict = dict()
            paths_dict = dict()

            for embedding_model in embedding_names:

                print("[INFO] Model: " + embedding_model + "...")
                scores = []
                paths = []
                for clustering_model in clustering_names:
                    scores.extend(np.load(os.path.join(self.members_dir, embedding_model, clustering_model, "scores.npy")).tolist())
                    with open(os.path.join(self.members_dir, embedding_model, clustering_model, "params_combo.json"), "r") as f:
                        variants = json.load(f)
                    variants = list(map(lambda x: os.path.join(self.partial_path, embedding_model, clustering_model, "preds", "train", x+".csv"), variants))
                    
                    paths.extend(variants)
                
                scores = np.array(scores)
                
                normalized_scores = scores.copy()
                normalized_scores[:,:, 1] = np.divide(1,np.add(1, normalized_scores[:,:, 1]))
                max_c_score, min_c_score = normalized_scores[:,:, 2].max(), normalized_scores[:,:, 2].min()
                normalize_vfunc = np.vectorize(normalize)
                normalized_scores[:, :, 2] = normalize_vfunc(normalized_scores[:, :, 2], min_c_score, max_c_score)
                # normalized_scores = np.sum(normalized_scores, axis=2)

                
                scores_dict[embedding_model] = scores
                normalized_dict[embedding_model] = normalized_scores[:, :, 1]
                paths_dict[embedding_model] = np.array(paths)

                model_path = os.path.join(self.complete_path, "v"+str(ver), embedding_model)
                os.makedirs(model_path, exist_ok=True)

                membership_matrix = []
                cluster_index_dict = {}
                cluster_idx = 0

                sorted_idx = normalized_dict[embedding_model][:,1].argsort()
                idx_of_interest = sorted_idx[-n_threshold:]
                # print(normalized_scores[sorted_idx[0],1], normalized_scores[sorted_idx[-1],1])
                paths_of_interest = paths_dict[embedding_model][idx_of_interest]

                for path_idx, path in enumerate(paths_of_interest):
                    cluster_preds = pd.read_csv(path)["predictions"]
                    one_hot_preds = pd.get_dummies(cluster_preds, prefix=str(cluster_idx))
                    membership_matrix.append(one_hot_preds)
                    cluster_index_dict[cluster_idx] = path
                    cluster_idx += 1
                
                with open(os.path.join(model_path, "meta_info.json"), "w") as f:
                    json.dump(cluster_index_dict, f)

                membership_matrix = pd.concat(membership_matrix, axis=1)
                membership_matrix.to_csv(os.path.join(model_path, "matrix.csv"), index=False)
                self.membership_matrices[embedding_model] = membership_matrix
            
            print("[INFO] Complete Membership matrix saved.")


            # for embedding_model in embedding_names:
                
            #     print("[INFO] Model: " + embedding_model + "...")

            #     membership_matrix = []
            #     cluster_index_dict = {}
            #     cluster_idx = 0
                
            #     model_path = os.path.join(self.complete_path, "v"+str(ver), embedding_model)
            #     os.makedirs(model_path, exist_ok=True)


            #     clustering_names = os.listdir(os.path.join(self.partial_path, embedding_model))
                
            #     for cluster_model in clustering_names:
            #         cluster_model_path = os.path.join(self.partial_path, embedding_model, cluster_model)
            #         cluster_variant_names = os.listdir(os.path.join(cluster_model_path, "info"))
                    
            #         info_path = os.path.join(cluster_model_path, "info")
            #         train_path = os.path.join(cluster_model_path, "preds", "train")
            #         test_path = os.path.join(cluster_model_path, "preds", "test")

            #         for model_variant in tqdm(cluster_variant_names, desc=(embedding_model + ", " + cluster_model)):
            #             # self.save_matrix_from_model(train_features=train_features, test_features=test_features, embedding_model_name=embedding_model, cluster_model_name=cluster_model, variant_name=model_variant)
            #             clusters = pd.read_csv(os.path.join(info_path, model_variant))["Topic"].values
            #             cluster_preds = pd.read_csv(os.path.join(train_path, model_variant))["predictions"]
                        
            #             if num_max_clusters >= clusters.shape[0] >= num_min_clusters:
            #                 one_hot_preds = pd.get_dummies(cluster_preds, prefix=str(cluster_idx))
            #                 membership_matrix.append(one_hot_preds)
            #                 cluster_index_dict[cluster_idx] = [embedding_model, cluster_model, model_variant]
            #                 cluster_idx += 1
            
            #     with open(os.path.join(model_path, "meta_info.json"), "w") as f:
            #         json.dump(cluster_index_dict, f)

            #     membership_matrix = pd.concat(membership_matrix, axis=1)
            #     membership_matrix.to_csv(os.path.join(model_path, "matrix.csv"), index=False)
            #     self.membership_matrices[embedding_model] = membership_matrix
            # print("[INFO] Complete Membership matrix saved.")

        else:
            print("[INFO] Loading Complete Membership Matrix for all embedding models...")
            embedding_names = os.listdir(self.complete_path)

            for embedding_model in embedding_names:
                print("[INFO] Model: " + embedding_model + "...")
                self.membership_matrices[embedding_model] = pd.read_csv(os.path.join(self.complete_path, "v"+str(ver), embedding_model ,"matrix.csv"))
            print("[INFO] Complete Membership matrix loaded.")


    ## Consensus function
    def consensus_fn(self, ver=0, load=False):
        
        def cluster_similarity(c1, c2):
            n = c1.shape[0]
            c1_sum = c1.sum()
            c2_sum = c2.sum()
            intersection = (c1*c2).sum()
            c1c2 = c1_sum*c2_sum
            return (intersection - (c1c2/n))/(np.sqrt(c1c2*(n-c1_sum)*(n-c2_sum))/n)
        
        def get_cluster_similarity(matrix, iter=1):
            # columns = matrix.columns
            # matrix_vals = matrix.values
            # columns_cluster = list(map(lambda x: int(x.split("_")[0]), columns)) if iter==1 else None
            # current_k = matrix.shape[1]
            # c_s = np.zeros((current_k, current_k))+(-2)
            
            # if iter==1:
            #     with tqdm(total=((current_k*(current_k-1)/2)), desc="Iteration - " + str(iter) + ", C_S Progress") as pbar:
            #         for i in range(current_k):
            #             for j in range(i+1,current_k):
            #                 if (columns_cluster[i] != columns_cluster[j]):
            #                     c_s[i,j] = cluster_similarity(matrix_vals[:, i], matrix_vals[:, j])
            #                 pbar.update(1)
            # else:
            #     with tqdm(total=((current_k*(current_k-1)/2)), desc="Iteration - " + str(iter) + ", C_S Progress") as pbar:
            #         for i in range(current_k):
            #             for j in range(i+1,current_k):
            #                 if (i!=j):
            #                     c_s[i,j] = cluster_similarity(matrix_vals[:, i], matrix_vals[:, j])
            #                 pbar.update(1)
            # return c_s
            return np.triu(np.corrcoef(matrix.astype("int").values.T), k=1) + np.tril((np.zeros(matrix.shape[1])-2), k=0)

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
                new_matrix[combination_name] = (matrix[cluster_combination].sum(axis=1)) #.apply(lambda x: 0 if x<1 else 1))
            new_matrix = pd.DataFrame(new_matrix)
            return new_matrix

        def update_merge_dict(iter_dict, merge_dict, iter):
            updated_dict = {}
            for key, value in iter_dict.items():
                updated_dict[key] = []
                for val in value:
                    updated_dict[key] += merge_dict[iter-1][val]
            return updated_dict
        
        def save_ensemble_results(path, matrix, merge_dict, alphas):
            os.makedirs(path, exist_ok=True)
            matrix.to_csv(os.path.join(path, "ensemble_soft_clusters.csv"), index=False)
            with open(os.path.join(path, "ensemble_merge_info.json"), "w") as f:
                json.dump(merge_dict, f)
            with open(os.path.join(path, "ensemble_merge_alpha_1.json"), "w") as f:
                json.dump(alphas, f)
        
        def save_final_results(path, matrix):
            os.makedirs(path, exist_ok=True)
            matrix.to_csv(os.path.join(path, "ensemble_hard_clusters.csv"), index=True, index_label="index")
        
        def load_ensemble_results(path):
            matrix = pd.read_csv(os.path.join(path, "ensemble_soft_clusters.csv"))
            with open(os.path.join(path, "ensemble_merge_info.json"), "r") as f:
                merge_dict = json.load(f)
            with open(os.path.join(path, "ensemble_merge_alpha_1.json"), "r") as f:
                alphas = json.load(f)
            return matrix, merge_dict, alphas

        def member_similarity(matrix, clusters_dict):
            membership_df = []
            columns = []
            for combination_name, cluster_combination in clusters_dict.items():
                columns.append(combination_name)
                membership_df.append(matrix[cluster_combination].sum(axis=1))
            
            membership_df = pd.DataFrame(membership_df, columns=columns)
            membership_df = membership_df/membership_df.max(axis=1)
            return membership_df

        def get_membership_df(merge_dicts, matrices, iters=None):
            if iters is None:
                iters = [list(merge_dicts[key].keys())[-1] for key in list(merge_dicts.keys())]
            new_matrices = {}
            for key in list(merge_dicts.keys()):
                new_matrices[key] = dict()
                for combination_name, cluster_combination in merge_dicts[key][str(iters[key])].items():        
                    new_matrices[key][combination_name] = matrices[key][cluster_combination].sum(axis=1)
                new_matrices[key] = pd.DataFrame(new_matrices[key])
            return new_matrices

        def get_cluster_quality(matrix):
            certainty = matrix.sum(axis=0) / (matrix > 0).sum(axis=0)
            mask = (matrix > 0).astype("int")
            return (((matrix - (mask * certainty))**2).sum(axis=0)) / mask.sum(axis=0)

        def add_uncertain_member(certain_mat, uncertain_mat, mat, clusters, idx):
            clusters_oi = mat.columns[mat.loc[uncertain_mat.iloc[idx].name] > 0].tolist()
            quality_diff = (get_cluster_quality(certain_mat._append(mat.loc[uncertain_mat.iloc[idx].name], ignore_index=False)[clusters_oi]) - get_cluster_quality(certain_mat[clusters_oi]))
            new_row = {cluster:0 for cluster in clusters}
            new_row[quality_diff.index[quality_diff.argmin()]] = (mat.loc[uncertain_mat.iloc[idx].name])[quality_diff.index[quality_diff.argmin()]]
            certain_mat.loc[uncertain_mat.iloc[idx].name] = new_row
        
        if not load:
            for embedding_model in list(self.membership_matrices.keys()):
                print("[INFO] Embedding Model: " + str(embedding_model) + "...")
                matrix = self.membership_matrices[embedding_model].copy()
                columns = matrix.columns
                alpha_01 = self.alpha_1
                alphas = []
                merge_dict = {}

                c_s = get_cluster_similarity(matrix, iter=1)
                merge_dict[1] = get_merge_dict(c_s, columns, alpha_01, iter=1)
                alphas.append(alpha_01)
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
                        alphas.append(alpha_01)
                        
                        if len(merge_dict_interim) >= current_k:
                            if (alpha_01) <= self.min_alpha_1:
                                alpha_flag = True
                            else:
                                alpha_01 = alpha_01 - self.alpha_diff_1
                                merge_dict_interim = get_merge_dict(c_s, matrix_interim.columns, alpha_01, iter=iter)
                                alphas[-1] = alpha_01

                        if not alpha_flag:    
                            merge_dict[iter] = update_merge_dict(merge_dict_interim, merge_dict, iter)
                            matrix_interim = merge_clusters(merge_dict, matrix, iter)
                            current_k = matrix_interim.shape[1]
                            iter += 1

                save_path = os.path.join(self.final_path, "v"+str(ver), embedding_model)
                save_ensemble_results(path=save_path, matrix=matrix_interim, merge_dict=merge_dict, alphas=alphas)
                print("[INFO] Saved soft ensemble clusters and merge dict.")

        print("[INFO] Loading pre-computed merge info and final clusters")    
        merge_infos = dict()
        final_clusters = dict()
        alphas_dict = dict()
        for embedding_model in list(self.membership_matrices.keys()):
            load_path = os.path.join(self.final_path, "v"+str(ver), embedding_model)
            matrix, merge_dict, alphas = load_ensemble_results(load_path)
            merge_infos[embedding_model] = merge_dict
            final_clusters[embedding_model] = matrix
            alphas_dict[embedding_model] = alphas

        print("[INFO] Enforcing hard clustering...")
        last_iters = {key:(list(merge_infos[key].keys())[-1]) for key in list(merge_infos.keys())}

        membership_dfs = get_membership_df(merge_infos, self.membership_matrices, last_iters)
        membership_mats = {key:(membership_dfs[key].div(membership_dfs[key].sum(axis=1), axis=0)) for key in list(membership_dfs.keys())}
        
        cluster_certainties = {key:(membership_mats[key].sum(axis=0) / (membership_mats[key] > 0).sum(axis=0)) for key in list(membership_mats.keys())}
        theta1_clusters = {key:(cluster_certainties[key].sort_values(ascending=False)[:self.final_k].index.tolist()) for key in list(cluster_certainties.keys())}

        theta1 = {i:(membership_dfs[i][theta1_clusters[i]]) for i in list(theta1_clusters.keys())}
        theta1_mats = {i:(theta1[i].div(theta1[i].sum(axis=1), axis=0).round(3)) for i in list(theta1.keys())}

        theta1_clusters = {i:(list(filter(lambda x: ((theta1_mats[i][x] >= self.alpha_2).sum() > 5), theta1_clusters[i]))) for i in list(theta1_clusters.keys())}
        theta2_clusters = {i:(list(filter(lambda x: True if (x not in theta1_clusters[i]) else False, membership_mats[i].columns.tolist()))) for i in list(theta1_clusters.keys())}

        theta1 = {i:(membership_dfs[i][theta1_clusters[i]]) for i in list(theta1_clusters.keys())}
        theta1_mats = {i:(theta1[i].div(theta1[i].sum(axis=1), axis=0).round(3)) for i in list(theta1.keys())}

        theta1_max_masks = {i:(np.zeros(theta1_mats[i].shape)) for i in list(theta1_mats.keys())}
        for i in list(theta1_max_masks.keys()):
            theta1_max_masks[i][(np.arange(theta1_mats[i].shape[0]), theta1_mats[i].values.argmax(axis=1))] = 1

        theta1_certain_mats = {i:((theta1_mats[i]*theta1_max_masks[i]).iloc[np.where(((theta1_mats[i].values * theta1_max_masks[i]).sum(axis=1)) >= self.alpha_2)]) for i in list(theta1_mats.keys())}
        theta1_uncertain_mats = {i:((theta1_mats[i]*theta1_max_masks[i]).iloc[np.where(((theta1_mats[i].values * theta1_max_masks[i]).sum(axis=1)) < self.alpha_2)]) for i in list(theta1_mats.keys())}

        for i in list(merge_infos.keys()):
            for j in tqdm(range(theta1_uncertain_mats[i].shape[0]), desc=(i + " Hard Clustering")):
                add_uncertain_member(theta1_certain_mats[i], theta1_uncertain_mats[i], theta1_mats[i], theta1_clusters[i], j)
        
        for key in list(theta1_certain_mats.keys()):
            save_path = os.path.join(self.final_path, "v"+str(ver), key)
            save_final_results(save_path, theta1_certain_mats[key])
        
        print("[INFO] Saving final results.")