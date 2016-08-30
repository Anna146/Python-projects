from collections import defaultdict
from io import StringIO
from sklearn import tree, ensemble
import sklearn
from sklearn.cross_validation import train_test_split
import numpy
from sklearn.metrics import recall_score, precision_score

# number of equal items
def eval(l1, l2):
    sum = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            sum+=1
    return sum

#homomorphism: it is good when y[i]=y[j] => pred[i]=pred[j]
#where pred - classifier prediction, y - real labels
def count_err(y, pred):
    err = 0
    for i in range(len(y)):
        #print(str(pred[i]) + " "  + str(y[i]))
        for j in range(i+1,len(y)):
            if (y[i] == y[j]) and (pred[i] != pred[j]):
                err += 1
    return err

f = open('features_new.txt', encoding = 'utf-8')
X = []
y = []
labels = [] #to visualize nodes

for line in f:
    if line != "\n":
        list = line.strip().split('\t')
        if list[0] == "id":
            labels = list[1:]
        else:
            X.append([float(x) for x in list[1:]])
            y.append(int(list[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#clf = tree.DecisionTreeClassifier(min_samples_leaf = 10)

best = [2] #min numbers of values there should be in the list (min possible users per class)
for k in best:
    #clf = ensemble.RandomForestClassifier(min_samples_leaf = k)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    leaves = clf.predict(X_test)
    #check quality - amount of errors
    err = count_err(y_test, leaves)
    #err = precision_score(y_test, leaves)
    print(err)
    print("----")
print(len(y))

#export the tree
tree.export_graphviz(clf, out_file='tree3.dot', feature_names = labels) #produces dot file



# show the classes for each instance sorted by probabilities
'''
probs = clf.predict_proba(X_test)
scores = []

for k in best:
    sam = 0
    correct = [0 for x in range(len(probs))]
    for pr in probs:
        arr = pr.tolist()
        ranked = defaultdict()
        for i in range(len(arr)):
            ranked.update({(arr[i],i):i})
        answ = y_test[sam]
        ranked = [x[1] for x in sorted(ranked.items(), key=lambda x: x[0], reverse=True)]
        #print(ranked)
        #break
        for j in range(min(k, len(ranked))):
            if ranked[j] == answ:
                correct[sam] = 1
        sam+=1

    print(sam)
    #print(len(y_test))
    print(sum(correct))
'''
#print(sorted(arr, reverse = True))
#y_pred = clf.predict(X_test)
#print("Number of mislabeled points out of a total %d points : %d" % (len(y_test),(y_test != y_pred).sum()))
