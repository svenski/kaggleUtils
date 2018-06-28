from fastai.structured import apply_cats
from fastai.structured import proc_df

def create_submission_based_on(model, df_raw, nas, path, df, filename = 'submission.csv', max_n_cat = None):
    df_test_raw = pd.read_csv(f'{path}test.csv')
    apply_cats(df_test_raw, df_raw)
    df_test_processed, _, nas = proc_df(df_test_raw, na_dict = nas, max_n_cat = max_n_cat)
    train_columns = df.columns.values
    df_test_final = df_test_processed[train_columns]

    y_test = model.predict(df_test_final)
    submission = pd.DataFrame({'Id':df_test_final.Id, 'SalePrice':np.exp(y_test)})
    submission.to_csv(filename, index = False)

def print_score(m, X_train, y_train, X_valid, y_valid):
    res = f'[ train rmse: {rmse(m.predict(X_train), y_train)}, val rmse: {rmse(m.predict(X_valid), y_valid)},\n\
train score: {m.score(X_train, y_train)}, val score: {m.score(X_valid, y_valid)}'
    if hasattr(m, 'oob_score_'): res +=f'\noob score: {m.oob_score_}'
    res +=']'
    print(res)

def print_score_auc(m, X_train, y_train, X_valid, y_valid):
    res = f'[ train auc: {aucFor(m, X_train, y_train)}, val auc: {aucFor(m, X_valid, y_valid)},\n train score: {m.score(X_train, y_train)}, val score: {m.score(X_valid, y_valid)}'
    if hasattr(m, 'oob_score_'): res +=f'\noob score: {m.oob_score_}'
    res +=']'
    print(res)

