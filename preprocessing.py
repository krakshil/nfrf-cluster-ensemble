## import packages
import os
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## load file
df = pd.read_excel("data/preprocessed/questions_list.xlsx")


## deduplication (within domain)
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df_unique = df.iloc[pd.DataFrame({"Follow - up Question":df["Follow - up Question"].apply(clean_text), "Base question info":df["Base question info"]}).drop_duplicates().index]
df_unique.to_excel("data/preprocessed/questions_list_unique.xlsx", index=False)


## Train-Test split
train_df, test_df = train_test_split(df_unique, test_size=0.1, random_state=42)
train_df.sort_values("Sr. No").to_excel("data/preprocessed/questions_list_unique_train.xlsx")
test_df.sort_values("Sr. No").to_excel("data/preprocessed/questions_list_unique_test.xlsx")