## import packages
import os
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub

## model definition
class topicModel():
    def __init__(self, docs, embedding_dict, clustering_dict, vectorizer_dict, ctfidf_dict, representation_dict):
        """
        docs: list (list of documents for topic modelling)

        embedding_dict: dict (contains metadata of the model and hyper-parameters)
            {"model_name": {"type":"tensorflow", "url":"url_or_path", "umap":{"n_neighbors":15, "min_dist":0.1, "n_components":2, "metric":"euclidean", "random_state":42}}}
        
        clustering_dict: dict (contains metadata of the types of clustering models with constructors and hyper-parameters)
            {"hdbscan": {"constructor": model_constructor - callable, "name":"model_name", "params_dict":{"min_cluster_size":[15,20,25], "min_samples":[5,10,15], "cluster_selection_epsilon":[0.4, 0.5, 0.7]}}}
        
        vectorizer_dict: dict (contains metadata of the types of vectorizer models with constructors and hyper-parameters)
            {}
        
        ctfidf_dict: dict (contains metadata of the types of ctf_idf models with constructors and hyper-parameters)
            {}

        representation_dict: dict (contains metadata of the types of representation models with constructors and hyper-parameters)
            {}
        """

        self.docs = docs
        self.embedding_dict = embedding_dict
        
        self.clustering_dict = clustering_dict
        
    ## compute embeddings for each model (reduce calculation of embeddings for each run)
    def compute_embeddings(self):

        for model_name in self.embedding_dict.keys():
            if self.embedding_dict[model_name]["type"] == "tensorflow":
                self.embedding_dict[model_name]["model"] = hub.load(self.embedding_dict[model_name]["url"])
                self.embedding_dict[model_name]["embeddings"] = self.embedding_dict[model_name]["model"](self.docs).numpy()
        

            elif self.embedding_dict[model_name]["type"] == "sentence_transformer":
                self.embedding_dict[model_name]["model"] = SentenceTransformer(self.embedding_dict[model_name]["url"])
                self.embedding_dict[model_name]["embeddings"] = self.embedding_dict[model_name]["model"].encode(self.docs)
        
            self.embedding_dict[model_name]["umap_model"] = UMAP(**self.embedding_dict[model_name]["umap_params"])
    
    ## fit model
    def fit(self, docs, embeddings, umap_model, clustering_model, verbose=False):
        
        topic_model = BERTopic(umap_model=umap_model, hdbscan_model=clustering_model, calculate_probabilities=True, low_memory=True, verbose=verbose)
        
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        freqs = topic_model.get_topic_info()
        tops = topic_model.get_topics()
        representation_docs = topic_model.get_representative_docs()
        question_topic_info = topic_model.get_document_info(docs)
        topic_embeddings = topic_model.topic_embeddings_
        hierarchical_topics = topic_model.hierarchical_topics(docs)
        topic_tree = topic_model.get_topic_tree(hierarchical_topics)
    
    ## loop
    def run(self):
        pass
    
    ## Save model
    def save(self, topic_model):
        pass