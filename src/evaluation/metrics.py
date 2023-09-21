'''
Implement whatever competition metric comes up tomorrow (Thursday)
'''
from sklearn.metrics import balanced_accuracy_score

def competition_metric(df_pred, df_true):
    extended_cols = ['AnB', 'AnC', 'AnD', 'BnC', 'BnD', 'CnD',
                     'AnBnC', 'AnBnD', 'AnCnD', 'BnCnD', 'AnBnCnD']
    target_cols = ['A', 'B', 'C', 'D']

    # find the weights for each of the columns
    final_score = 0.
    for c in target_cols:
        w = 0.5 / len(target_cols)
        final_score += w * balanced_accuracy_score(df_true[c], df_pred[c])
    for c in extended_cols:
        w = 0.5 / len(extended_cols)
        final_score += w * balanced_accuracy_score(df_true[c], df_pred[c])
    return final_score

