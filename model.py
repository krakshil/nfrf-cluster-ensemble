## import packages
import os
import string
import json
from tqdm import tqdm
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

## model definition
class topicModel():
    def __init__(self, train_docs:list, test_docs:list, embedding_config:dict, clustering_config:dict, vectorizer_config=None, ctfidf_config=None, representation_config=None, save_dir=""):
        """
        docs: list (list of documents for topic modelling)

        embedding_config: dict (contains metadata of the model and hyper-parameters)
            {"model_name": {"type":"tensorflow", "url":"url_or_path", "umap_params":{"n_neighbors":15, "min_dist":0.1, "n_components":2, "metric":"euclidean", "random_state":42}}}
        
        clustering_config: dict (contains metadata of the types of clustering models with constructors and hyper-parameters)
            {"hdbscan": {"constructor": model_constructor - callable, "type":"model_type", "params_dict":{"min_cluster_size":[15,20,25], "min_samples":[5,10,15], "cluster_selection_epsilon":[0.4, 0.5, 0.7]}}}
        
        vectorizer_config: dict [default: None] (contains metadata of the types of vectorizer models with constructors and hyper-parameters)
            {"name": "model_name", "constructor": model_constructor - callable, "params_dict":{"param_1":value, "param_2":value}}
        
        ctfidf_config: dict [default: None] (contains metadata of the types of ctf_idf models with constructors and hyper-parameters)
            {"name": "model_name", "constructor": model_constructor - callable, "params_dict":{"param_1":value, "param_2":value}}

        representation_config: dict [default: None] (contains metadata of the types of representation models with constructors and hyper-parameters)
            {"name": "model_name", "constructor": model_constructor - callable, "params_dict":{"param_1":value, "param_2":value}}
        """

        self.train_docs = train_docs
        self.test_docs = test_docs
        self.embedding_config = embedding_config
        self.embedding_dict = {key: {} for key in embedding_config.keys()}
        
        self.clustering_config = clustering_config
        # self.clustering_dict = {key: {} for key in clustering_config.keys()}
        self.vectorizer_config = vectorizer_config
        self.ctfidf_config = ctfidf_config
        self.representation_config = representation_config
        self.evaluation_dict = {}

        self.save_directory = save_dir
        
    ## compute embeddings for each model (reduce calculation of embeddings for each run)
    def compute_embeddings(self):

        for model_name in self.embedding_config.keys():
            if self.embedding_config[model_name]["type"] == "tensorflow":
                self.embedding_dict[model_name]["model"] = hub.load(self.embedding_config[model_name]["url"])
                self.embedding_dict[model_name]["train_embeddings"] = self.embedding_dict[model_name]["model"](self.train_docs).numpy()
                self.embedding_dict[model_name]["test_embeddings"] = self.embedding_dict[model_name]["model"](self.test_docs).numpy()
        

            elif self.embedding_config[model_name]["type"] == "sentence_transformer":
                self.embedding_dict[model_name]["model"] = SentenceTransformer(self.embedding_config[model_name]["url"])
                self.embedding_dict[model_name]["train_embeddings"] = self.embedding_dict[model_name]["model"].encode(self.train_docs)
                self.embedding_dict[model_name]["test_embeddings"] = self.embedding_dict[model_name]["model"].encode(self.test_docs)
        
            self.embedding_dict[model_name]["umap_model"] = UMAP(*self.embedding_config[model_name]["umap_params"])
            self.embedding_dict[model_name]["train_features"] = self.embedding_dict[model_name]["umap_model"].fit_transform(self.embedding_dict[model_name]["train_embeddings"])
            self.embedding_dict[model_name]["test_features"] = self.embedding_dict[model_name]["umap_model"].transform(self.embedding_dict[model_name]["test_embeddings"])
        
        self.save_embeddings()
    
    ## train model
    def fit(self, docs, embeddings, umap_model, clustering_model, vectorizer_model, ctfidf_model, representation_model, verbose=False):
        
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

    ## training loop
    def run(self, verbose=False):
        
        print("[INFO] Creating directory heirarchy for storing data and results...")
        self.create_save_dir()
        print("[INFO] Directories created. Computing embeddings and features from the documents using embedding models...")
        self.compute_embeddings()
        print("[INFO] Embeddings Calculated and saved. Starting iterations of topic modelling...")

        empty_dimensionality_model = BaseDimensionalityReduction()

        vectorizer_model = self.vectorizer_config["constructor"](*self.vectorizer_config["params_dict"]) if self.vectorizer_config is not None else None
        ctfidf_model = self.ctfidf_config["constructor"](*self.ctfidf_config["params_dict"]) if self.ctfidf_config is not None else None
        representation_model = self.representation_config["constructor"](*self.representation_config["params_dict"]) if self.representation_config is not None else None

        for model_name in self.embedding_dict.keys():
            
            if model_name not in self.evaluation_dict.keys():
                self.evaluation_dict[model_name] = {}

            train_features = self.embedding_dict[model_name]["train_features"]
            test_features = self.embedding_dict[model_name]["test_features"]

            for cluster_model_name in self.clustering_config.keys():

                if cluster_model_name not in self.evaluation_dict[model_name].keys():
                    self.evaluation_dict[model_name][cluster_model_name] = {}

                constructor = self.clustering_config[cluster_model_name]["constructor"]
                if "params_grid" not in self.clustering_config[cluster_model_name].keys():
                    params_grid = list(ParameterGrid(self.clustering_config[cluster_model_name]["params_dict"]))
                    self.clustering_config[cluster_model_name]["params_grid"] = params_grid
                
                for param_combo in tqdm(self.clustering_config[cluster_model_name]["params_grid"], desc= model_name + ", " + cluster_model_name):
                    cluster_model = constructor(*param_combo)

                    topic_model, (train_labels, train_probs) = self.fit(self.train_docs, train_features, empty_dimensionality_model, cluster_model, vectorizer_model, ctfidf_model, representation_model, verbose=verbose)
                    train_scores = self.calculate_scores(train_features, train_labels)
                    evaluation_scores = self.evaluate(self.test_docs, test_features, verbose=verbose)

                    save_name = ", ".join(str(key) + "=" + str(value) for (key, value) in param_combo.items())
                    self.evaluation_dict[model_name][cluster_model_name][save_name] = np.array([train_scores, evaluation_scores])
                    
                    save_path = os.path.join(self.save_directory, model_name, cluster_model_name, save_name)
                    topic_model.save(save_path, serialization="pickle", save_embedding_model=False)

                with open(os.path.join(self.save_directory, model_name, cluster_model_name, "params_combo.json"), "W") as f:
                    f.write(json.dumps(list(self.evaluation_dict[model_name][cluster_model_name].keys())))
                
                with open(os.path.join(self.save_directory, model_name, cluster_model_name, "scores.npy"), "wb") as f:
                    np.save(f, np.array(list(self.evaluation_dict[model_name][cluster_model_name].values())))
                
                print("[INFO] " + model_name + ", " + cluster_model_name + ": Complete. Moving to next model...")
            
            print("[INFO] " + model_name + ": Completed.")
        
        print("All iterations complete. All the data was stored.")
                
    
    ## evaluate model
    def evaluate(self, docs, embeddings, topic_model):
        labels, probs = topic_model.transform(docs, embeddings)
        s_score, d_score, c_score = self.calculate_scores(features=embeddings, labels=labels)
        return (s_score, d_score, c_score)

    def calculate_scores(self, features, labels):
        s_score, d_score, c_score = silhouette_score(X=features, labels=labels), davies_bouldin_score(X=features, labels=labels), calinski_harabasz_score(X=features, labels=labels)
        return s_score, d_score, c_score
    
    ## Save model
    def create_save_dir(self):
        
        for model_name in self.embedding_config.keys():
            for cluster_model_name in self.clustering_config.keys():
                os.makedirs(os.path.join(self.save_directory, model_name, cluster_model_name), exist_ok=True)
    
    ## save embeddings
    def save_embeddings(self):

        with open(os.path.join(self.save_directory, "docs.json"), "w") as f:
            f.write(json.dumps({"train_docs":self.train_docs, "test_docs":self.test_docs}))

        for model_name in self.embedding_dict.keys():
            train_embeddings = self.embedding_dict[model_name]["train_embeddings"]
            train_features = self.embedding_dict[model_name]["train_features"]
            test_embeddings = self.embedding_dict[model_name]["test_embeddings"]
            test_features = self.embedding_dict[model_name]["test_features"]
            umap_model = self.embedding_dict[model_name]["umap_model"]
            
            with open(os.path.join(self.save_directory, model_name, "embeddings.npz"), "wb") as f:
                np.savez(f, train_embeddings=train_embeddings, train_features=train_features, test_embeddings=test_embeddings, test_features=test_features)
            
            with open(os.path.join(self.save_directory, model_name, "embedding_dict.json"), "w") as f:
                f.write(json.dumps(self.embedding_config[model_name]))