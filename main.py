import os
import warnings
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, Birch
from model import topicModel, CustomSpectralClustering, CustomGaussianMixture, customEnsemble
from cluster_ensemble import ClusterEnsemble
# from embedding_selection import topicModel

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

data_dir = os.path.join("data", "preprocessed")
train_file = "questions_list_unique_train.xlsx"
test_file = "questions_list_unique_test.xlsx"

train_docs = pd.read_excel(os.path.join(data_dir, train_file))["Follow - up Question"].values.tolist()
test_docs = pd.read_excel(os.path.join(data_dir, test_file))["Follow - up Question"].values.tolist()

# save_dir = "data"
save_dir = os.path.join("data","results","cluster_ensemble", "final", "v5")

save_dir_en = os.path.join("data","results","baseline")

# ## Add hyper-parameter combinations for embedding selection
# embedding_selection_config = {
#     "embedding_models": {"ST-MiniLM-L6": ["sentence_transformer", "all-MiniLM-L6-v2"],
#                          "ST-mnpnet-base": ["sentence_transformer","all-mpnet-base-v2"],
#                          "ST-MiniLM-L12": ["sentence_transformer","all-MiniLM-L12-v2"],
#                          "ST-distil-roberta": ["sentence_transformer","all-distilroberta-v1"],
#                          "ST-mpnet-qa": ["sentence_transformer","multi-qa-mpnet-base-dot-v1"],
#                          "ST-t5-base": ["sentence_transformer","sentence-t5-base"],
#                          "ST-LaBSE": ["sentence_transformer","LaBSE"],
#                          "USE": ["tensorflow","https://tfhub.dev/google/universal-sentence-encoder/4"]},
#     "umap_params_dict": {"n_neighbors":[5, 10, 15, 20, 25], "min_dist":[0.1, 0.25, 0.5, 0.75, 0.99], "n_components":[5, 10, 15, 20], "metric":["cosine","euclidean"], "low_memory":[False], "random_state":[42]},
#     "hdbscan_params_dict": {"min_cluster_size":[5, 10, 15, 20, 25, 30, 40, 50], "min_samples":[1, 5, 10, 15, 20], "cluster_selection_epsilon":[0.3, 0.5, 0.7, 1, 1.2, 2], "metric":["euclidean","manhattan"], "prediction_data":[True]}
# }


## Add UMAP configuration for each embedding model
embedding_config = {
    "ST-distil-roberta": {"type":"sentence_transformer", "url":"all-distilroberta-v1", "umap_params":{"n_neighbors":5, "min_dist":0.25, "n_components":5, "metric":"euclidean", "low_memory":False, "random_state":42}},
    "ST-MiniLM-L6": {"type":"sentence_transformer", "url":"all-MiniLM-L6-v2", "umap_params":{"n_neighbors":25, "min_dist":0.1, "n_components":15, "metric":"euclidean", "low_memory":False, "random_state":42}},
    "ST-mpnet-qa": {"type":"sentence_transformer", "url":"multi-qa-mpnet-base-dot-v1", "umap_params":{"n_neighbors":10, "min_dist":0.25, "n_components":10, "metric":"euclidean", "low_memory":False, "random_state":42}}
}

# Add hyper-parameter combinations - incomplete (test each algorithms)
# clustering_config = {
#     "hdbscan": {"constructor":HDBSCAN, "type":"density-based", "params_dict":{"min_cluster_size":[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], "min_samples":[1, 5, 10, 15, 20, 25, 30, 35], "cluster_selection_epsilon":[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 2], "metric":["euclidean","manhattan"], "prediction_data":[True]}},
    
#     "birch": {"constructor":Birch, "type":"birch", "params_dict":{"threshold":[0.25, 0.37, 0.5, 0.62, 0.75, 0.87, 1, 1.25, 1.5, 2], "branching_factor":[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "n_clusters":[None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}},
    
#     "kmeans": {"constructor":KMeans, "type":"k-means", "params_dict":{"n_clusters":[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "max_iter":[300, 400, 500, 600, 700], "n_init":[1, 10, 20, 30, 40, 50], "init":["k-means++", "random"], "algorithm":["lloyd","elkan"], "random_state":[42]}},
    
#     "spectral": {"constructor":CustomSpectralClustering, "type":"spectral", "params_dict":[{"n_clusters":[5, 6, 7, 8, 9, 10], "n_components":[3,5,7], "affinity":["nearest_neighbors"], "n_neighbors":[5, 10, 15, 20, 25], "assign_labels":["kmeans","discretize", "cluster_qr"], "n_jobs":[-1], "random_state":[42]},
#                                                                                            {"n_clusters":[5, 6, 7, 8, 9, 10], "n_components":[3,5,7], "affinity":["rbf", "cosine", "laplacian", "sigmoid"], "gamma":[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1], "assign_labels":["kmeans","discretize", "cluster_qr"], "random_state":[42]},
#                                                                                            {"n_clusters":[5, 6, 7, 8, 9, 10], "n_components":[3,5,7], "affinity":["polynomial"], "degree":[3,4,5,6,7], "assign_labels":["kmeans","discretize", "cluster_qr"], "random_state":[42]},
#                                                                                            ]},
    
#     "guassian": {"constructor":CustomGaussianMixture, "type":"guassian", "params_dict":[{"n_components":[5, 6, 7, 8, 9, 10], "covariance_type":["full", "tied", "diag", "spherical"], "tol":[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "n_init":[1, 10, 50], "init_params":["k-means++", "random_from_data"], "reg_covar":[1e-4], "random_state":[42]},
#                                                                                   {"n_components":[5, 6, 7, 8, 9, 10], "covariance_type":["full", "tied", "diag", "spherical"], "tol":[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "warm_start":[True], "init_params":["k-means++", "random_from_data"], "reg_covar":[1e-4], "random_state":[42]},
#                                                                                   ]},
# }

clustering_config = {
    "ensemble": {"constructor":customEnsemble, "type":"ensemble", "params_dict":{"n_neighbors":[5, 10, 15, 20, 25, 30, 35]}},
}

vectorizer_config = None
ctfidf_config = None
representation_config = None

ensemble_model = ClusterEnsemble(train_docs=train_docs, test_docs=test_docs, members_dir=save_dir_en, min_alpha_1 = 0.4, alpha_2=0.3, save_dir=os.path.join("data","results"))
# # ensemble_model.get_partial_membership()
ensemble_model.combine_partial_membership_matrix(ver=5, n_threshold=240, load=False)
print("\n\n")
total_clusters = 0
for key, matrix in ensemble_model.membership_matrices.items():
    total_clusters += matrix.shape[1]
    print(key + ": ", matrix.shape)
print("\nTotal Clusters: " + str(total_clusters) + "\n\n")
ensemble_model.consensus_fn(ver=5, load=False)


topic_model = topicModel(train_docs=train_docs, test_docs=test_docs, embedding_config=embedding_config, clustering_config=clustering_config, vectorizer_config=vectorizer_config, ctfidf_config=ctfidf_config, representation_config=representation_config, ver=5, save_dir=save_dir)
# # topic_model = topicModel(train_docs=train_docs, test_docs=test_docs, embedding_selection_config=embedding_selection_config, save_dir=save_dir)
# # topic_model.run_selection()
topic_model.run(load_embeddings=True)
topic_model.get_evaluation_scores(load_embeddings=True, gt_dir="data")
topic_model.save_best_scores()