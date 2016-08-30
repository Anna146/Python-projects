from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier, \
    RandomizedLasso, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

__author__ = 'Asus'

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold


train_df = pd.DataFrame.from_csv("train_ds_exp_three_clean", sep='\t', encoding='utf-8')
test_df = pd.DataFrame.from_csv("test_ds_exp_three_clean", sep='\t', encoding='utf-8')


#train_df_new = pd.DataFrame.from_csv("train_ds", sep='\t', encoding='utf-8')
'''
ogr = []
for c in train_df:
    if c not in train_df_new:
        ogr.append(c)
print(ogr)
'''
#train_df = train_df[ogr]
#test_df = test_df[ogr]


predictors = train_df.columns.tolist()
prediction = "uid"
predictors.remove(prediction)



from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

X_train = train_df[predictors][:]
y_train = train_df[prediction]

X_test = test_df[predictors][:]
y_test = test_df[prediction]


from sklearn import metrics

# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(metrics.precision_recall_fscore_support(y_test, pred, average='weighted'))
    print(pred[y_test.values == pred].size / float(len(pred)))
    return pred[y_test.values == pred].size / float(len(pred))
    '''
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))
    '''
'''
feat_selector = LinearSVC(C=0.4, penalty="l1", dual=False, tol=1e-3)
feat_selector.fit(X_train,y_train)
X_train = feat_selector.transform(X_train)
X_test = feat_selector.transform(X_test)
results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (BernoulliNB(alpha=.0001), "BernoulliNB"),
        (MultinomialNB(alpha=.000001), "MultinomialNB_noselection"),
        (RandomForestClassifier(n_estimators=50, n_jobs=-1), "Random forest"),
        (ExtraTreesClassifier(n_estimators=50, n_jobs=-1, min_samples_split = 1, max_features=None, class_weight="balanced"), "Extra forest")
         ):
    print(name)
    results.append(benchmark(clf))
exit(0)
'''

#clf = MultinomialNB(alpha=.000001)
clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, min_samples_split = 1, max_features=None, class_weight="balanced")

baseline = benchmark(clf)
exit(0)
'''
good_features = []

for f in ogr:
    pred1 = predictors
    pred1.append(f)

    X_train = train_df[pred1][:]
    y_train = train_df[prediction]

    X_test = test_df[pred1][:]
    y_test = test_df[prediction]
    if benchmark(clf) >= baseline:
        good_features.append(f)
    print(f)
print(good_features)

pred1 = predictors
pred1.extend(good_features)

X_train = train_df[pred1][:]
y_train = train_df[prediction]

X_test = test_df[pred1][:]
y_test = test_df[prediction]
res = benchmark(clf)
print("WITH GOOD FEATURES" + str(res))
print("BASELINE" + str(baseline))
'''
kk = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
max = 0.0
kmax = -1
reses = []
for kkk in kk:
    res = benchmark(Pipeline([
          #("feture selection", SelectKBest(k=kkk)),
          ('feature_selection', LinearSVC(C=kkk, penalty="l1", dual=False, tol=1e-3)),
          #('feature_selection', LogisticRegression(C=w, penalty="l1", dual=False, tol=1e-3)),
          #('classification', MultinomialNB(alpha=.000001))
          #('classification', RandomForestClassifier(n_estimators=100, n_jobs=-1))
          ('classification', ExtraTreesClassifier(n_estimators=20, n_jobs=-1, min_samples_split = 1, max_features=None, class_weight="balanced"))
        ]))
    reses.append(res)
    if res > max:
        max = res
        kmax = kkk

import matplotlib.pyplot as plt


plt.title('KBest feature selector')
plt.ylabel('Precision')
plt.xlabel('Number of Features')
#plt.xlim(32,212)                # try commenting this out...
plt.grid(True)

plt.plot(kk,reses)
plt.show()

exit(0)
benchmark(Pipeline([
      #("faeture selection", SelectKBest(k=kmax)),
      ('feature_selection', LinearSVC(C=0.4, penalty="l1", dual=False, tol=1e-3)),
      #('feature_selection', LogisticRegression(C=w, penalty="l1", dual=False, tol=1e-3)),
      #('classification', MultinomialNB(alpha=.000001))
      #('classification', RandomForestClassifier(n_estimators=100, n_jobs=-1))
      ('classification', ExtraTreesClassifier(n_estimators=100, n_jobs=-1, min_samples_split = 1, max_features=None, class_weight="balanced"))
    ]))
exit(0)
results = []

#grid = [32 - x for x in range(10)]
'''
for x in grid:
    ch2 = SelectKBest(f_classif, k=x)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("n_samples: %d, n_features: %d" % X_train.shape)
    benchmark(clf)
    print()
'''
'''
from sklearn.linear_model import RandomizedLasso
randomized_lasso = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_train = randomized_lasso.fit_transform(X_train, y_train)
X_test = randomized_lasso.transform(X_test)
print("n_samples: %d, n_features: %d" % X_train.shape)
benchmark(clf)
exit(0)
'''


for clf, name in (
        #(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        #(Perceptron(n_iter=50), "Perceptron"),
        #(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        #(BernoulliNB(alpha=.0001), "BernoulliNB"),
        (MultinomialNB(alpha=.000001), "MultinomialNB_noselection"),
        #(RandomForestClassifier(n_estimators=20, n_jobs=-1), "Random forest"),
        #(ExtraTreesClassifier(n_estimators=50, n_jobs=-1, min_samples_split = 1, max_features=None, class_weight="balanced"), "Extra forest")
         ):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))
exit(0)
'''
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

benchmark(bdt_real)
'''
grid = [1e-3]

for w in grid:
    results.append(benchmark(Pipeline([
      ('feature_selection', LinearSVC(C=0.4, penalty="l1", dual=False, tol=w)),
      #('feature_selection', LogisticRegression(C=w, penalty="l1", dual=False, tol=1e-3)),
      #('classification', MultinomialNB(alpha=.000001))
      ('classification', RandomForestClassifier(n_estimators=20, n_jobs=-1))
    ])))
exit(0)

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))
exit(0)

import numpy as np
import matplotlib.pyplot as plt
# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()