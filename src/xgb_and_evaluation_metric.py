
import xgbooast as xgb 
import lightgbm as lgb 
from sklearn.metrics import accuracy_score
from  sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
import xgboost as xgb

"""# Model evaluation - Top10 accuracy """

def top10_accuracy_scorer(probas, target, label_dict):
    """A custom scorer that evaluates a model on whether the correct label is in 
    the top 10 most probable predictions.

    Args:
        estimator (sklearn estimator): The sklearn model that should be evaluated.
        X (numpy array): The validation data.
        y (numpy array): The ground truth labels.

    Returns:
        float: Accuracy of the model as defined by the proportion of predictions
               in which the correct label was in the top 10. Higher is better.
    """
    # predict the probabilities across all possible labels for rows in our training set
    # probas = estimator.predict_proba(X)
    
    # get the indices for top 10 predictions for each row; these are the last ten in each row
    # Note: We use argpartition, which is O(n), vs argsort, which uses the quicksort algorithm 
    # by default and is O(n^2) in the worst case. We can do this because we only need the top ten
    # partitioned, not in sorted order.
    # Documentation: https://numpy.org/doc/1.18/reference/generated/numpy.argpartition.html


    # index into the classes list using the top ten indices to get the class names
    # top10_preds = estimator.classes_[top10_idx]
    # top10_preds = [label_dict[label] for label  in top10_idx]
    # true_labels = np.array([label_dict[idx] for  idx in target])

    # check if y-true is in top 10 for each set of predictions
    
    top10_idx = np.argpartition(probas, -10, axis=1)[:, -10:]
    top10_preds = [label_dict[label] for label  in top10_idx]
    true_labels = np.array([label_dict[idx] for  idx in target])

    mask = top10_preds == y.reshape((y.size, 1))
    top_10_accuracy = mask.any(axis=1).mean()
 
    return top_10_accuracy
    
def runXGB(train_X, train_y, test_X, test_y = None, test_X2=None, seed_val=786, child= 0.3052, colsample= 0.80 ):
    param = {}
    param['objective'] = 'multi:softprob'
    param['num_class'] = 1314
    param['eta'] = 0.1 # EARLIER LEARNING RATE -- 0.102345
    param['max_depth'] = 6 # EARLIER MAX DEPTH ---- 6
    param['silent'] = 1
    param['eval_metric'] = "mlogloss"
    # param['max_delta_step'] = 9.462
    param['min_child_weight'] = child
    param['tree_method'] = 'gpu_hist'
    param['subsample'] = 0.90
    param['gamma'] = 1.086
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    param['reg_lambda'] = 1.74
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label = train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label = test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
        
    return pred_test_y, pred_test_y2, model


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "multiclass",
        "metric" : "multi_logloss", 
        "num_leaves" : 24,
        "num_classes":1314,
        "max_depth": 6,
        "lambda_l1":0.7051,
        "lambda_l2": 2.703,
        "boosting": 'gbdt',
        "min_child_weight" : 5.166,
        "learning_rate" : 0.1056932,
        "bagging_fraction" : 0.9173,
        "min_split_gain":0.01564,
        "feature_fraction" : 0.4275,
        "bagging_seed" : 2018,
        # 'device' : 'gpu',
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 5000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)

    return pred_test_y, model, pred_val_y

 
kf = StratifiedKFold(n_splits = 5)
cv_scores = []
pred_full_test_xgb = 0

pred_train = np.zeros((predictors.shape))
pred_test = np.zeros((predictors_test.shape))

for dev_index, val_index in kf.split(predictors, target):
    dev_X, val_X = predictors.iloc[dev_index], predictors.iloc[val_index]
    dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, predictors_test)
    pred_full_test_xgb = pred_full_test_xgb + pred_test_y
    pred_train[val_index] = pred_val_y

#     # label_val_y = np.argmax(pred_val_y, axis = 1)
#     print("ROC_AUC_FOR_THIS_ITERATION_IS :: -- >> ", f1_score(val_y, label_val_y, average = 'weighted'))
#     print("\n")
#     cv_scores.append(f1_score(val_y, label_val_y, average = 'weighted'))
# np.mean(cv_scores)


