import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

annotation_dir = os.path.join("data", "annotations")
annotation_files = os.listdir(annotation_dir)

for (file_name, label_col) in zip(glob.glob(annotation_dir + "/*.xlsx"), ["Cluster", "Code", "Cluster"]):

    df = pd.read_excel(file_name)
    gt_labels = LabelEncoder().fit_transform(df[label_col])
    
    df["labels"] = gt_labels

    df.to_excel(file_name[:-5] + "_labelled.xlsx")