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
    "ST-MiniLM": {"type":"sentence_transformer", "url":"all-MiniLM-L6-v2", "umap_params":{"n_neighbors":25, "min_dist":0.25, "n_components":10, "metric":"cosine", "low_memory":False}},
    "ST-mnpnet": {"type":"sentence_transformer", "url":"all-mpnet-base-v2", "umap_params":{"n_neighbors":25, "min_dist":0.25, "n_components":10, "metric":"cosine", "low_memory":False}},
    "USE": {"type":"tensorflow", "url":"https://tfhub.dev/google/universal-sentence-encoder/4", "umap_params":{"n_neighbors":25, "min_dist":0.25, "n_components":10, "metric":"cosine", "low_memory":False}}
}

## Add hyper-parameter combinations
clustering_config = {
    "hdbscan": {"constructor":HDBSCAN, "type":"density-based", "params_dict":{"min_cluster_size":[6,7,8,9,10,11,12], "min_samples":[1,2,3,4,5], "cluster_selection_epsilon":[0.0, 0.1, 0.2, 0.3, 0.4], "metric":["euclidean","cosine"], "prediction_data":True}},
    "birch": {"constructor":Birch, "type":"birch", "params_dict":{"threshold":[0.25,0.5, 0.75, 1], "branching_factor":[30, 40, 50, 60, 70], "n_clusters":[6, 7, 8, 9, 10, 11, 12]}},
    "kmeans": {"constructor":KMeans, "type":"k-means", "params_dict":{"n_clusters":[5, 6, 7, 8, 9, 10, 11, 12, 13], "max_iter":[300, 400, 500], "n_init":[10, 20, 30], "algorithm":["lloyd","elkan"], "random_state":42}}
}
vectorizer_config = None
ctfidf_config = None
representation_config = None


topic_model = topicModel(train_docs=train_docs, test_docs=test_docs, embedding_config=embedding_config, clustering_config=clustering_config, vectorizer_config=vectorizer_config, ctfidf_config=ctfidf_config, representation_config=representation_config, save_dir=save_dir)
topic_model.run()