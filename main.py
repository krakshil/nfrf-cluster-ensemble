import os
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, Birch
from model import topicModel, CustomSpectralClustering, CustomGaussianMixture

data_dir = os.path.join("data", "preprocessed")
train_file = "questions_list_unique_train.xlsx"
test_file = "questions_list_unique_test.xlsx"

train_docs = pd.read_excel(os.path.join(data_dir, train_file))["Follow - up Question"].values.tolist()
test_docs = pd.read_excel(os.path.join(data_dir, test_file))["Follow - up Question"].values.tolist()

save_dir = os.path.join("data", "results")

## Add UMAP configuration for each embedding model
embedding_config = {
    "ST-MiniLM": {"type":"sentence_transformer", "url":"all-MiniLM-L6-v2", "umap_params":{"n_neighbors":25, "min_dist":0.25, "n_components":10, "metric":"cosine", "low_memory":False, "random_state":42}},
    "ST-mnpnet": {"type":"sentence_transformer", "url":"all-mpnet-base-v2", "umap_params":{"n_neighbors":25, "min_dist":0.25, "n_components":10, "metric":"cosine", "low_memory":False, "random_state":42}},
    "USE": {"type":"tensorflow", "url":"https://tfhub.dev/google/universal-sentence-encoder/4", "umap_params":{"n_neighbors":25, "min_dist":0.25, "n_components":10, "metric":"cosine", "low_memory":False, "random_state":42}}
}

## Add hyper-parameter combinations - incomplete (test each algorithms)
clustering_config = {
    "hdbscan": {"constructor":HDBSCAN, "type":"density-based", "params_dict":{"min_cluster_size":[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], "min_samples":[1, 5, 10, 15, 20, 25, 30, 35], "cluster_selection_epsilon":[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 2], "metric":["euclidean","manhattan"], "prediction_data":[True]}},
    
    "birch": {"constructor":Birch, "type":"birch", "params_dict":{"threshold":[0.25, 0.37, 0.5, 0.62, 0.75, 0.87, 1, 1.25, 1.5, 2], "branching_factor":[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "n_clusters":[None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}},
    
    "kmeans": {"constructor":KMeans, "type":"k-means", "params_dict":{"n_clusters":[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "max_iter":[300, 400, 500, 600, 700], "n_init":[1, 10, 20, 30, 40, 50], "init":["k-means++", "random"], "algorithm":["lloyd","elkan"], "random_state":[42]}},
    
    "spectral": {"constructor":CustomSpectralClustering, "type":"spectral", "params_dict":[{"n_clusters":[5, 6, 7, 8, 9, 10], "n_components":[3,5,7], "affinity":["nearest_neighbors"], "n_neighbors":[5, 10, 15, 20, 25], "assign_labels":["kmeans","discretize", "cluster_qr"], "n_jobs":[-1], "random_state":[42]},
                                                                                           {"n_clusters":[5, 6, 7, 8, 9, 10], "n_components":[3,5,7], "affinity":["rbf", "cosine", "laplacian", "sigmoid"], "gamma":[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1], "assign_labels":["kmeans","discretize", "cluster_qr"], "random_state":[42]},
                                                                                           {"n_clusters":[5, 6, 7, 8, 9, 10], "n_components":[3,5,7], "affinity":["polynomial"], "degree":[3,4,5,6,7], "assign_labels":["kmeans","discretize", "cluster_qr"], "random_state":[42]},
                                                                                           ]},
    
    "guassian": {"constructor":CustomGaussianMixture, "type":"guassian", "params_dict":[{"n_components":[5, 6, 7, 8, 9, 10], "covariance_type":["full", "tied", "diag", "spherical"], "tol":[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "n_init":[1, 10, 50], "init_params":["k-means++", "random_from_data"], "reg_covar":[1e-4], "random_state":[42]},
                                                                                  {"n_components":[5, 6, 7, 8, 9, 10], "covariance_type":["full", "tied", "diag", "spherical"], "tol":[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "warm_start":[True], "init_params":["k-means++", "random_from_data"], "reg_covar":[1e-4], "random_state":[42]},
                                                                                  ]},
}

vectorizer_config = None
ctfidf_config = None
representation_config = None


topic_model = topicModel(train_docs=train_docs, test_docs=test_docs, embedding_config=embedding_config, clustering_config=clustering_config, vectorizer_config=vectorizer_config, ctfidf_config=ctfidf_config, representation_config=representation_config, save_dir=save_dir)
topic_model.run(load_embeddings=True)
topic_model.save_best_scores()