import pandas as pd
import numpy as np
import math

from sklearn.metrics import roc_auc_score

def printAllPandasColumns(max = None):
    pd.set_option('display.max_columns', None)

def split_by(validation_idx, df_raw):
    if(isinstance(df_raw, pd.DataFrame)):
        raw_valid = df_raw.iloc[validation_idx]
        raw_train = df_raw.loc[~df_raw.index.isin(raw_valid.index)]
    else:
        raw_valid = np.take(df_raw, validation_idx)
        raw_train = np.delete(df_raw, validation_idx)

    return raw_train, raw_valid

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

def aucFor(m, X_train, y_train):
    return(roc_auc_score(y_train, m.predict_proba(X_train)[:,1]))

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def getCategoricalColumns(df):
    # return( [ f for f in df.columns if df[f].dtype == 'object' ] )
    cat_vars = [col for col in df if df[col].dtype.name != 'float64' and df[col].dtype.name != 'float32' and len(df[col].unique()) < 150]
    for v in cat_vars: df[v] = df[v].astype('category').cat.as_ordered()
    return(cat_vars)
    
def getObjectColumns(df):
    return( [ f for f in df.columns if df[f].dtype == 'object' ] )

def getEmbeddingSizesFor(df_raw, categorical_cols):
     cat_sz = [(c, len(df_raw[c].cat.categories) + 1) for c in categorical_cols]
     return [(c, min(50, (c+1)//2)) for _, c in cat_sz]
