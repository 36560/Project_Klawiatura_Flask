from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pandas as pd
import numpy as np

#from cuml import RandomForestClassifier as cuRF


def convert_to_int(word):
    logins = {0: 0, 'Kinga': 1, 'Krystian': 2, 'Patryk': 3, 'Joanna' : 4}
    return logins[word]

# def convert_to_int(word):
#     logins = {0: 0, 'Kinga': 1, 'Krystian': 2, 'Patryk': 3}
#     return logins[word]

# Load data
data = pd.read_csv('key_entity2_plus.csv')
data.shape  # rows, cols

# Delete row with 0 values in time_to_next
listIndex = (data[data['time_to_next'] == 0].index.values)

print(listIndex)

for i in listIndex:
    data = data.drop(i)  # delete row

print(data.size)
print("Removing null values...")
data = data.dropna()
print(data.size)

# data
data.pop('id')
X = data.drop('login', axis=1)  # samples

y = data['login']  # Labels
y = y.apply(lambda x: convert_to_int(x))
y

print("Learning...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)  # Split to train and test data

rf = RandomForestClassifier()

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=2000, stop=3000, num=100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 100, num=100)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 7, 9, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=1000, cv=2, verbose=4, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)
print(rf_random.best_score_)
print(rf_random.best_estimator_)