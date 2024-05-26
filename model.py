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


## Spectral clustering
class CustomSpectralClustering:
    def __init__(self, n_clusters=3, **kwargs):
        self.model = SpectralClustering(n_clusters=n_clusters, **kwargs)
        self.classifier = KNeighborsClassifier(n_neighbors=10)

    def fit(self, X, y):
        self.labels_ = self.model.fit_predict(X)
        self.classifier.fit(X, self.labels_)
        return self

    def predict(self, X_new):
        return self.classifier.predict(X_new)

## Gaussian Mixture
class CustomGaussianMixture:
    def __init__(self, n_components=5, **kwargs):
        self.model = GaussianMixture(n_components=n_components, **kwargs)

    def fit(self, X, y):
        self.labels_ = self.model.fit_predict(X, y=y)
        return self

    def predict(self, X_new):
        return self.model.predict(X_new)

## Ensemble
class customEnsemble:
    def __init__(self, path="", n_neighbors=10):
        self.path = path
        self.mat = pd.read_csv(os.path.join(path, "ensemble_hard_clusters.csv")).sort_values(by="index").drop(["index"], axis=1)
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def fit(self, X, y):
        self.labels_ = self.mat.values.argmax(axis=1)
        print(X.shape, self.labels_.shape)
        self.classifier.fit(X, self.labels_)
        return self

    def predict(self, X_new):
        return self.classifier.predict(X_new)


## model definition
class topicModel():
    def __init__(self, train_docs:list, test_docs:list, embedding_config:dict, clustering_config:dict, vectorizer_config=None, ctfidf_config=None, representation_config=None, ver=0, save_dir=""):
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

        self.ver = ver
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
    def compute_embeddings(self, load_embeddings=False):

        if load_embeddings:
            for model_name in self.embedding_config.keys():
                embeddings = np.load(os.path.join(self.save_directory, model_name, "embeddings.npz"))
                with open(os.path.join(self.save_directory, model_name, "embedding_dict.json"), "r") as f:
                    umap_params = json.loads(f.read())["umap_params"]

                self.embedding_dict[model_name]["model"] = None
                self.embedding_dict[model_name]["train_embeddings"] = embeddings["train_embeddings"]
                self.embedding_dict[model_name]["test_embeddings"] = embeddings["test_embeddings"]
                self.embedding_dict[model_name]["umap_model"] = UMAP(**self.embedding_config[model_name]["umap_params"])

                if (umap_params == self.embedding_config[model_name]["umap_params"]) and ("random_state" in umap_params):
                    self.embedding_dict[model_name]["train_features"] = embeddings["train_features"]
                    self.embedding_dict[model_name]["test_features"] = embeddings["test_features"]
                else:
                    self.embedding_dict[model_name]["train_features"] = self.embedding_dict[model_name]["umap_model"].fit_transform(self.embedding_dict[model_name]["train_embeddings"])
                    self.embedding_dict[model_name]["test_features"] = self.embedding_dict[model_name]["umap_model"].transform(self.embedding_dict[model_name]["test_embeddings"])

        else:
            for model_name in self.embedding_config.keys():
                if self.embedding_config[model_name]["type"] == "tensorflow":
                    self.embedding_dict[model_name]["model"] = hub.load(self.embedding_config[model_name]["url"])
                    self.embedding_dict[model_name]["train_embeddings"] = self.embedding_dict[model_name]["model"](self.train_docs).numpy()
                    self.embedding_dict[model_name]["test_embeddings"] = self.embedding_dict[model_name]["model"](self.test_docs).numpy()
            

                elif self.embedding_config[model_name]["type"] == "sentence_transformer":
                    self.embedding_dict[model_name]["model"] = SentenceTransformer(self.embedding_config[model_name]["url"])
                    self.embedding_dict[model_name]["train_embeddings"] = self.embedding_dict[model_name]["model"].encode(self.train_docs)
                    self.embedding_dict[model_name]["test_embeddings"] = self.embedding_dict[model_name]["model"].encode(self.test_docs)
            
                self.embedding_dict[model_name]["umap_model"] = UMAP(**self.embedding_config[model_name]["umap_params"])
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
    def run(self, load_embeddings=False, verbose=False):
        
        print("[INFO] Creating directory heirarchy for storing data and results...")
        self.create_save_dir()
        print("[INFO] Directories created. Computing embeddings and features from the documents using embedding models...")
        self.compute_embeddings(load_embeddings=load_embeddings)
        print("[INFO] Embeddings Calculated and saved. Starting iterations of topic modelling...\n")

        empty_dimensionality_model = BaseDimensionalityReduction()

        vectorizer_model = self.vectorizer_config["constructor"](**self.vectorizer_config["params_dict"]) if self.vectorizer_config is not None else None
        ctfidf_model = self.ctfidf_config["constructor"](**self.ctfidf_config["params_dict"]) if self.ctfidf_config is not None else None
        representation_model = self.representation_config["constructor"](**self.representation_config["params_dict"]) if self.representation_config is not None else None

        for model_name in self.embedding_dict.keys():
            
            if model_name not in self.evaluation_dict.keys():
                self.evaluation_dict[model_name] = {}

            train_features = self.embedding_dict[model_name]["train_features"]
            test_features = self.embedding_dict[model_name]["test_features"]

            for cluster_model_name in self.clustering_config.keys():
                
                constructor = self.clustering_config[cluster_model_name]["constructor"]
                if "params_grid" not in self.clustering_config[cluster_model_name].keys():
                    params_grid = list(ParameterGrid(self.clustering_config[cluster_model_name]["params_dict"]))
                    self.clustering_config[cluster_model_name]["params_grid"] = params_grid
                
                for param_combo in tqdm(self.clustering_config[cluster_model_name]["params_grid"], desc = model_name + ", " + cluster_model_name):
                    # cluster_model = constructor(**param_combo) # normal clustering
                    cluster_model = constructor(**{**{"path": os.path.join("data", "results", "cluster_ensemble", "final_merge", "v"+str(self.ver), model_name)}, **param_combo}) # ensemble clustering 

                    topic_model, (train_labels, train_probs) = self.fit(self.train_docs, train_features, empty_dimensionality_model, cluster_model, vectorizer_model, ctfidf_model, representation_model, verbose=verbose)
                    save_name = ", ".join(str(key) + "=" + str(value) for (key, value) in param_combo.items())

                    save_path = os.path.join(self.save_directory, model_name, cluster_model_name, save_name)
                    topic_model.save(save_path, serialization="pickle", save_embedding_model=False)

                print("[INFO] " + model_name + ", " + cluster_model_name + ": Complete. Moving to next model...\n")
            
            print("[INFO] " + model_name + ": Completed.\n")
        
        print("All iterations complete. All the data was stored.")
                
    
    ## evaluate model
    def evaluate(self, docs, embeddings, topic_model, gt_labels=None):
        preds, probs = topic_model.transform(docs, embeddings)
        s_score, d_score, c_score = self.calculate_internal_scores(features=embeddings, labels=preds)
        external_scores = []
        for labels in gt_labels:
            external_scores.append(self.calculate_external_scores(gt=labels, pred=preds))
        external_scores = np.mean(external_scores, axis=0)
        return (s_score, d_score, c_score, external_scores[0], external_scores[1])

    ## calculate internal metrics for a model
    def calculate_internal_scores(self, features, labels):
        try:
            s_score, d_score, c_score = silhouette_score(X=features, labels=labels), davies_bouldin_score(X=features, labels=labels), calinski_harabasz_score(X=features, labels=labels)
        except:
            s_score, d_score, c_score = -1, float('inf'), 0
        return (s_score, d_score, c_score)

    ## calculate external metrics for a model
    def calculate_external_scores(self, gt, pred):
        try:
            ari_score, mi_score = adjusted_rand_score(gt, pred), adjusted_mutual_info_score(gt, pred)
        except:
            ari_score, mi_score = 0, 0
        return (ari_score, mi_score)

    ## calculate evaluation scores for all models
    def get_evaluation_scores(self, load_embeddings=False, gt_dir=""):
        
        print("\n\n[INFO] Beginning evaluation ...\n")

        gt_labels = self.load_gt_labels(gt_dir=gt_dir)

        for model_name in self.embedding_dict.keys():
            
            if model_name not in self.evaluation_dict.keys():
                self.evaluation_dict[model_name] = {}

            if ("train_features" not in self.embedding_dict[model_name].keys()) or ("test_features" not in self.embedding_dict[model_name].keys()):
                self.compute_embeddings(load_embeddings=load_embeddings)
            
            train_features = self.embedding_dict[model_name]["train_features"]
            test_features = self.embedding_dict[model_name]["test_features"]

            for cluster_model_name in self.clustering_config.keys():
                
                if cluster_model_name not in self.evaluation_dict[model_name].keys():
                    self.evaluation_dict[model_name][cluster_model_name] = {}

                num_cluster_count = {}
                cluster_model_path = os.path.join(self.save_directory, model_name, cluster_model_name)
                cluster_model_combos = os.listdir(cluster_model_path)
                
                for file_name in ["num_clusters.json","params_combo.json","scores.npy"]:
                    if file_name in cluster_model_combos:
                        cluster_model_combos.remove(file_name)

                for cluster_model_combination in tqdm(cluster_model_combos  , desc = model_name + ", " + cluster_model_name):
                    
                    topic_model = BERTopic.load(os.path.join(cluster_model_path, cluster_model_combination))

                    num_cluster_count[cluster_model_combination] = topic_model.get_topic_info().shape[0]

                    if topic_model.get_topic_info().shape[0] > 2:
                        train_scores = self.evaluate(self.train_docs, train_features, topic_model, gt_labels)
                        evaluation_scores = self.evaluate(self.test_docs, test_features, topic_model, gt_labels)
                    else:
                        train_scores, evaluation_scores = (-1, float('inf'), 0, 0, 0), (-1, float('inf'), 0, 0, 0)

                    self.evaluation_dict[model_name][cluster_model_name][cluster_model_combination] = np.array([train_scores, evaluation_scores])

                with open(os.path.join(self.save_directory, model_name, cluster_model_name, "params_combo.json"), "w") as f:
                    json.dump(list(self.evaluation_dict[model_name][cluster_model_name].keys()), f)
                
                with open(os.path.join(self.save_directory, model_name, cluster_model_name, "scores.npy"), "wb") as f:
                    np.save(f, np.array(list(self.evaluation_dict[model_name][cluster_model_name].values())))
                
                with open(os.path.join(self.save_directory, model_name, cluster_model_name, "num_clusters.json"), "w") as f:
                    json.dump(num_cluster_count, f)
                
                print("[INFO] " + model_name + ", " + cluster_model_name + ": Complete. Moving to next model...\n")

            print("[INFO] " + model_name + ": Completed.\n")
        
        print("Evaluation complete. All the scores were stored.")

    ## load gt-labels
    def load_gt_labels(self, gt_dir=""):
        gt_labels = []
        for file_path in glob.glob(os.path.join(gt_dir,"annotations", "*_labelled.xlsx")):
            df = pd.read_excel(file_path)
            gt_labels.append(df["labels"].tolist())
        return np.array(gt_labels)

    ## Save model
    def create_save_dir(self):
        
        load_embeddings = None

        for model_name in self.embedding_config.keys():
            for cluster_model_name in self.clustering_config.keys():
                os.makedirs(os.path.join(self.save_directory, model_name, cluster_model_name), exist_ok=True)
        
        return load_embeddings
    
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

    ## find best hyper-parameter combination for each clustering algo in each embedding space
    def save_best_scores(self):
        
        print("\n\n[INFO] Calculating best scores from all hyper-parameter combinations for all model types ...")

        def normalize(value, min_val, max_val):
            return ((value - min_val) / (max_val - min_val))

        best_scores = {}
        for model_name in self.embedding_dict.keys():
            best_scores[model_name] = {}
            for cluster_model_name in self.clustering_config.keys():
                
                with open(os.path.join(self.save_directory, model_name, cluster_model_name, "params_combo.json"), "r") as f:
                    params_combo = json.loads(f.read())
                
                scores = np.load(os.path.join(self.save_directory, model_name, cluster_model_name, "scores.npy"))

                ### best scores - incomplete (problematic combinations (score error) not totally excluded in best score calculation (c-score range -1))
                normalized_scores = scores.copy()
                normalized_scores[:,:, 1] = np.divide(1,np.add(1, normalized_scores[:,:, 1]))
                # normalized_scores[normalized_scores == np.divide(1,0)] = -np.divide(1,0)
                max_c_score, min_c_score = normalized_scores[:,:, 2].max(), normalized_scores[:,:, 2].min()
                # print(model_name, cluster_model_name, max_c_score, min_c_score)
                normalize_vfunc = np.vectorize(normalize)
                normalized_scores[:, :, 2] = normalize_vfunc(normalized_scores[:, :, 2], min_c_score, max_c_score)

                #### For cluster ensemble
                # normalized_scores = np.sum(normalized_scores, axis=2)
                best_score_index = np.argmax(normalized_scores[:, 1, 0])

                best_parameter_combo = params_combo[best_score_index]
                best_score = normalized_scores[best_score_index, 1]

                best_scores[model_name][cluster_model_name] = {best_parameter_combo:{"best_score":best_score, "train_scores": scores[best_score_index, 0].tolist(), "test_scores": scores[best_score_index, 1].tolist()}}
        
        with open(os.path.join(self.save_directory, "best_scores.json"), "w") as f:
            json.dump(best_scores, f)
        
        print("[INFO] Best models found and information was stored.")