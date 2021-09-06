from statistics import mean
import pandas as pd
from matplotlib import pyplot
from numba import jit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut

@jit
def kFolds():
    for k in folds:
        # define the test condition
        cv = KFold(n_splits=k, shuffle=True, random_state=1)
        # evaluate k value
        k_mean, k_min, k_max = evaluate_model(cv)
        # report performance
        print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
        # store mean accuracy
        means.append(k_mean)
        # store min and max relative to the mean
        mins.append(k_mean - k_min)
        maxs.append(k_max - k_mean)

@jit
def convert_to_int(word):
    logins = {0: 0, 'Kinga': 1, 'Krystian': 2, 'Patryk': 3}
    return logins[word]

def get_data():
    # Load data
    data = pd.read_csv('short_plus_all.csv')
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

    return X, y


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
@jit
def get_model():
    model = RandomForestClassifier(n_estimators=80, bootstrap=True, max_features='sqrt', max_depth=29, verbose=1)
    #model = RandomForestClassifier(n_estimators=80, min_samples_split=25, min_samples_leaf=6, max_features='log2', max_depth=93, bootstrap=True)
    return model
@jit
def evaluate_model(cv):
    # get the dataset
    X, y = get_data()
    # get the model
    model = get_model()
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()


ideal, _, _ = evaluate_model(LeaveOneOut())
print('Ideal: %.3f' % ideal)

# define folds to test
folds = range(2, 31)

# record mean and min/max of each set of results
means, mins, maxs = list(), list(), list()
# evaluate each k value
kFolds()

    # line plot of k mean values with min/max error bars
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
# plot the ideal case in a separate color
pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')
# show the plot
pyplot.show()
