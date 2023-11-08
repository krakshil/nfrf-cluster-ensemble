import os
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, Birch
from model import topicModel

data_dir = os.path.join("data", "preprocessed")
train_file = "questions_list_unique_train.xlsx"
test_file = "questions_list_unique_test.xlsx"

train_docs = pd.read_excel(os.path.join(data_dir, train_file))["Follow - up Question"].values.tolist()
test_docs = pd.read_excel(os.path.join(data_dir, test_file))["Follow - up Question"].values.tolist()

save_dir = os.path.join("data", "results")

## Add UMAP configuration for each embedding model
embedding_config = {
    "ST-MiniLM": {"type":"sentence_transformer", "url":"all-MiniLM-L6-v2", "umap_params":{"n_neighbors":0, "min_dist":0, "n_components":0, "metric":""}},
    "ST-mnpnet": {"type":"sentence_transformer", "url":"all-mpnet-base-v2", "umap_params":{"n_neighbors":0, "min_dist":0, "n_components":0, "metric":""}},
    "USE": {"type":"tensorflow", "url":"https://tfhub.dev/google/universal-sentence-encoder/4", "umap_params":{"n_neighbors":0, "min_dist":0, "n_components":0, "metric":""}}
}

## Add hyper-parameter combinations
clustering_config = {
    "hdbscan": {"constructor":HDBSCAN, "type":"density-based", "params_dict":{"min_cluster_size":[0,0,0], "min_samples":[0,0,0], "cluster_selection_epsilon":[0,0,0], "metric":["",""]}},
    "birch": {"constructor":Birch, "type":"birch", "params_dict":{"threshold":[0,0,0], "branching_factor":[0,0,0], "n_clusters":[0,0,0]}},
    "kmeans": {"constructor":KMeans, "type":"k-means", "params_dict":{"n_clusters":[0,0,0], "max_iter":[0,0,0], "n_init":[0,0,0], "algorithm":["",""], "random_state":42}}
}
vectorizer_config = None
ctfidf_config = None
representation_config = None


topic_model = topicModel(train_docs=train_docs, test_docs=test_docs, embedding_config=embedding_config, clustering_config=clustering_config, vectorizer_config=vectorizer_config, ctfidf_config=ctfidf_config, representation_config=representation_config, save_dir=save_dir)
topic_model.run()