import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from joblib import dump, load

#4 klasy
def convert_to_int(word):
    logins = {0: 0, 'Kinga': 1, 'Krystian': 2, 'Patryk': 3, 'Joanna': 4}
    return logins[word]

# Load data
data = pd.read_csv('key_entity2_plus.csv')   #key_entity2_sis
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

kf = KFold(n_splits=29, shuffle=True)
model = RandomForestClassifier(bootstrap=False, max_depth=49, min_samples_split=20, n_estimators=172)

#model = RandomForestClassifier(max_depth=36, min_samples_split=7, n_estimators=2545)
# model = RandomForestClassifier(n_estimators=1780, min_samples_split=25, min_samples_leaf=4, max_features='sqrt',  max_depth=29, bootstrap=True)
# model = RandomForestClassifier(n_estimators=114, bootstrap=True, max_features='sqrt', max_depth=39)
# model = RandomForestClassifier(max_depth=92, max_features='log2', min_samples_leaf=2, min_samples_split=15, n_estimators=396)
# model = RandomForestClassifier(max_depth=32, max_features='sqrt', min_samples_split=20, n_estimators=871)
# model = RandomForestClassifier(max_depth=32, max_features='sqrt', min_samples_split=20, n_estimators=871)
#model = RandomForestClassifier(n_estimators=114, bootstrap=True, max_features='sqrt', max_depth=39)
# model = RandomForestClassifier(max_depth=81, max_features='log2', min_samples_split=15, n_estimators=1626)
# model = RandomForestClassifier(max_depth=31, max_features='sqrt', min_samples_split=25, n_estimators=643)

for train_index, test_index in kf.split(X):
    model.fit(X.values[train_index], y.values[train_index])
    score = model.score(X.values[train_index], y.values[train_index])
    print("Score dla train: ", score)
    score = model.score(X.values[test_index], y.values[test_index])
    print("Score dla train: ", score)

dump(model, 'model.joblib')