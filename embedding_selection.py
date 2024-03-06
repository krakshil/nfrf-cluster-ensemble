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


## model definition
class topicModel():
    def __init__(self, train_docs:list, test_docs:list, embedding_selection_config:dict, save_dir=""):
        """
        docs: list (list of documents for topic modelling)

        embedding_selection_config: dict (contain metadata of the embedding models, umap hyper-paraters and hdbscan hyper-parameters)
            {"embedding_models": {"model_name": ["type", "url_or_path"]},
             "umap_params_dict": {"n_neighbors":[], "min_dist":[], "n_components":[], "metric":[], "random_state":[42]},
             "hdbscan_params_dict": {"min_cluster_size":[15,20,25], "min_samples":[5,10,15], "cluster_selection_epsilon":[0.4, 0.5, 0.7]}
            }
        """

        self.train_docs = train_docs
        self.test_docs = test_docs
        
        self.embedding_selection_config = embedding_selection_config
        self.embedding_selection_umap_params_list = list(ParameterGrid(self.embedding_selection_config["umap_params_dict"]))
        self.embedding_selection_hdbscan_params_list = list(ParameterGrid(self.embedding_selection_config["hdbscan_params_dict"]))
        self.embedding_selection_dict = {key: {} for key in self.embedding_selection_config["embedding_models"].keys()}

        # self.embedding_config = embedding_config
        
        # self.clustering_config = clustering_config
        # # self.clustering_dict = {key: {} for key in clustering_config.keys()}
        # self.vectorizer_config = vectorizer_config
        # self.ctfidf_config = ctfidf_config
        # self.representation_config = representation_config
        # self.evaluation_dict = {}
        self.selection_evaluation_dict = {}
        self.save_directory = os.path.join(save_dir, "results")
    
    
    ## compute embeddings for each model (reduce calculation of embeddings for each run)
    def compute_selection_embeddings(self, create_save_dir=True): 
        for model_name in self.embedding_selection_dict.keys():
            if self.embedding_selection_config["embedding_models"][model_name][0] == "tensorflow":
                self.embedding_selection_dict[model_name]["model"] = hub.load(self.embedding_selection_config["embedding_models"][model_name][1])
                self.embedding_selection_dict[model_name]["train_embeddings"] = self.embedding_selection_dict[model_name]["model"](self.train_docs).numpy()
                self.embedding_selection_dict[model_name]["test_embeddings"] = self.embedding_selection_dict[model_name]["model"](self.test_docs).numpy()
        

            elif self.embedding_selection_config["embedding_models"][model_name][0] == "sentence_transformer":
                self.embedding_selection_dict[model_name]["model"] = SentenceTransformer(self.embedding_selection_config["embedding_models"][model_name][1])
                self.embedding_selection_dict[model_name]["train_embeddings"] = self.embedding_selection_dict[model_name]["model"].encode(self.train_docs)
                self.embedding_selection_dict[model_name]["test_embeddings"] = self.embedding_selection_dict[model_name]["model"].encode(self.test_docs)

        self.save_selection_embeddings(create_save_dir=create_save_dir)
    

    ## save selection embeddings
    def save_selection_embeddings(self, create_save_dir:bool=False):

        if create_save_dir:
            os.makedirs(os.path.join(self.save_directory,"embedding_selection"), exist_ok=True)
        
        # with open(os.path.join(self.save_directory, "embedding_selection", "docs.json"), "w") as f:
        #     f.write(json.dumps({"train_docs":self.train_docs, "test_docs":self.test_docs}))

        for model_name in self.embedding_selection_dict.keys():
            os.makedirs(os.path.join(self.save_directory, "embedding_selection", model_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_directory, "embedding_selection", model_name, "selection_results"), exist_ok=True)
            train_embeddings = self.embedding_selection_dict[model_name]["train_embeddings"]
            test_embeddings = self.embedding_selection_dict[model_name]["test_embeddings"]

            with open(os.path.join(self.save_directory, "embedding_selection", model_name, "embeddings.npz"), "wb") as f:
                np.savez(f, train_embeddings=train_embeddings, test_embeddings=test_embeddings)
    

    ## load selection embeddings
    def load_selection_embeddings(self):

        model_names = self.embedding_selection_dict["embedding_models"].keys()
            
        for model_name in model_names:
            embeddings = np.load(os.path.join(self.save_directory, "embedding_selection", model_name, "embeddings.npz"))
            self.embedding_selection_dict[model_name]["model"] = None
            self.embedding_selection_dict[model_name]["train_embeddings"] = embeddings["train_embeddings"]
            self.embedding_selection_dict[model_name]["test_embeddings"] = embeddings["test_embeddings"]
    
    
    ## run selection model
    def run_selection(self, load_embeddings=False, create_save_dir=True, verbose=False):
        
        if load_embeddings:
            print("[INFO] Loading embeddings and features from the documents using embedding models...")
            self.load_embeddings()
            print("[INFO] Embeddings loaded. Starting iterations of topic modelling...\n")
        else:
            print("[INFO] Computing embeddings from the documents using embedding models...")
            self.compute_selection_embeddings(create_save_dir=create_save_dir)
            print("[INFO] Embeddings calculated and saved. Starting iterations of topic modelling...\n")

        
        with open(os.path.join(self.save_directory, "embedding_selection", "umap_params_list.json"), "w") as f:
            json.dump(self.embedding_selection_umap_params_list, f)
        
        with open(os.path.join(self.save_directory, "embedding_selection", "hdbscan_params_list.json"), "w") as f:
            json.dump(self.embedding_selection_hdbscan_params_list, f)

        empty_dimensionality_model = BaseDimensionalityReduction()

        vectorizer_model = None
        ctfidf_model = None
        representation_model = None

        num_total_variants = len(list(self.embedding_selection_dict.keys()))*len(self.embedding_selection_umap_params_list)*len(self.embedding_selection_hdbscan_params_list)
        model_idx, umap_idx, hdb_idx = 0
        # try:
        with tqdm(total=num_total_variants, desc="Progress") as pbar:
            
            for model_idx, model_name in enumerate(self.embedding_selection_dict.keys()):

                train_embeddings = self.embedding_selection_dict[model_name]["train_embedding"]
                test_embeddings = self.embedding_selection_dict[model_name]["test_embedding"]

                for umap_idx, umap_param in enumerate(self.embedding_selection_umap_params_list):
                    
                    umap_model = UMAP(**umap_param)
                    train_features = umap_model.fit_transform(train_embeddings)
                    test_features = umap_model.transform(test_embeddings)

                    selection_evaluation_list = []
                    
                    for hdb_idx, hdbscan_param in enumerate(self.embedding_selection_hdbscan_params_list):
                        
                        cluster_model = HDBSCAN(**hdbscan_param)
                        topic_model, (train_labels, train_probs) = self.fit(self.train_docs, train_features, empty_dimensionality_model, cluster_model, vectorizer_model, ctfidf_model, representation_model, verbose=verbose)
                        (test_labels, test_probs) = topic_model.transform(self.test_docs, test_features)
                        train_s_score, train_d_score, train_c_score = self.calculate_internal_scores(features=train_features, labels=train_labels)
                        test_s_score, test_d_score, test_c_score = self.calculate_internal_scores(features=test_features, labels=test_labels)

                        selection_evaluation_list.append([[train_s_score, train_d_score, train_c_score],[test_s_score, test_d_score, test_c_score]])
                        pbar.update(1)
                        ################ incomplete: save each results in CSV and folder start here #######################
                        
                    # save_path = os.path.join(self.save_directory, model_name, cluster_model_name, save_name)
                
                    result_file_name = "___".join(str(key) + "_" + str(value) for (key, value) in umap_param.items()) + ".json"
                    with open(os.path.exists(self.save_directory, "embedding_selection", model_name, "selection_results", result_file_name), "w") as f:
                        json.dump(selection_evaluation_list, f)
        
                print("[INFO] " + model_name + ": Completed.\n")
    
            print("[INFO] All iterations complete. All the data was stored.")
            
        # except:
        #     print("Model Index: " + str(model_idx) + ", UMAP Index:" + str(umap_idx) + ", HDBSCAN Index:" + str(hdb_idx))
    

    ## train model
    def fit(self, docs, embeddings, umap_model, clustering_model, vectorizer_model=None, ctfidf_model=None, representation_model=None, verbose=False):
        
        topic_model = BERTopic(umap_model=umap_model, hdbscan_model=clustering_model, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, representation_model=representation_model, calculate_probabilities=True, low_memory=True, verbose=verbose)
        
        labels, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        # freqs = topic_model.get_topic_info()
        # tops = topic_model.get_topics()
        # representation_docs = topic_model.get_representative_docs()
        # question_topic_info = topic_model.get_document_info(docs)
        # topic_embeddings = topic_model.topic_embeddings_
        # hierarchical_topics = topic_model.hierarchical_topics(docs)
        # topic_tree = topic_model.get_topic_tree(hierarchical_topics)

        return topic_model, (labels, probs)

    
    ## calculate internal metrics for a model
    def calculate_internal_scores(self, features, labels):
        try:
            s_score, d_score, c_score = silhouette_score(X=features, labels=labels), davies_bouldin_score(X=features, labels=labels), calinski_harabasz_score(X=features, labels=labels)
        except:
            s_score, d_score, c_score = -1, float('inf'), 0
        return (s_score, d_score, c_score)