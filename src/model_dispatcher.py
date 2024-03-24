from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
models = {
    'decision_tree' : tree.DecisionTreeClassifier(
        criterion = 'gini', 
        max_depth= 10, 
        max_features= None, 
        min_samples_leaf= 1,
        min_samples_split= 7
    ),
    'lr' : linear_model.LogisticRegression(),
    'rf' : ensemble.RandomForestClassifier(
        n_estimators= 300,
        min_samples_split=  10,
        min_samples_leaf=  1,
        max_features=  'auto',
        max_depth = None, 
        bootstrap =  True
    ) 

}