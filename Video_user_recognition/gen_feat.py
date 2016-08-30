from datetime import datetime, time
import pprint
from sklearn import cluster
import operator
import numpy
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
import statistics
import sklearn
import math
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

import pymongo
import logging
from sklearn import tree, ensemble
import sklearn
from sklearn.cross_validation import train_test_split

__author__ = 'Asus'

def dt(u):
    return datetime.utcfromtimestamp(u)

# number of equal items
def eval(l1, l2):
    sum = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            sum+=1
    return sum

#homomorphism: it is good when y[i]=y[j] => pred[i]=pred[j]
#where pred - classifier prediction, y - real labels
def count_err(y, pred, X):
    err = 0
    for i in range(len(y)):
        #print(str(pred[i]) + " "  + str(y[i]))
        for j in range(i+1,len(y)):
            if (y[i] == y[j]) and (pred[i] != pred[j]):
                print(X[i])
                err += 1
    return err

#distribution of watches per hour for time
#frac - total time for USER (divide by it in the end)
#return 25 features
def dealWithTime(details, frac):
    #not to divide by 0
    frac = max(1,frac)
    mfhs = dict([(x,0) for x in range(0,25)])
    for det in details:
        if "time" in det:
            d = dt(det['time'])
            h = round(((((d.hour * 60) + d.minute) * 60) + d.second) / (60.0 * 60))
            mfhs[h] += 1
    #normalize by user
    res = [x/frac for x in mfhs.values()]
    return res

#distribution of watches per weekdays
#frac - total watches for USER (divide by it in the end)
#return 7 features
def dealWithDate(details, frac):
    #not to divide by 0
    frac = max(1,frac)
    wds = dict([(x,0) for x in range(0,7)])
    for det in details:
        if "time" in det:
            d = dt(det['time'])
            w = d.weekday()
            wds[w] += 1
    #normalize by user
    res = [x/frac for x in wds.values()]
    return res


def dealWithWeekend(details, frac):
    #not to divide by 0
    frac = max(1,frac)
    wds = dict([(x,0) for x in range(0,7)])
    for det in details:
        if "time" in det:
            d = dt(det['time'])
            w = d.weekday()
            wds[w] += 1
    #normalize by user
    res = [x/frac for x in wds.values()]
    if max(sum(res[0:4]),sum(res[5:6])) == sum(res[0:4]):
        return [0]
    else:
        return [1]
    #return res


#returns len(channels)*2 features
#calculates ratio duration/length and count for each channel
def VODs_time_ratio(details, v, names, timeF, viewF):
    # v - DictVectorizer
    # names of channels
    # timeF - total time of user
    # viewF - total views of user
    names_t = {} #time at each channel
    names_l = {} #lengths of the watch clips to calculate ratio watched/length
    names_c = {} #clicks (views) at each channel
    for n in names:
        names_t.setdefault(n,0)
        names_l.setdefault(n,0)
        names_c.setdefault(n,0)
    for det in details:
        if "channel" in det:
            if det["channel"] in names_t:
                #add view
                names_c[det["channel"]] += 1
                #add clipLen
                if "clipLen" in det:
                    names_l[det["channel"]] = max(names_l[det["channel"]], float(det["clipLen"]))
                #add watched time
                if "duration" in det:
                    names_t[det["channel"]] = max(names_t[det["channel"]], det["duration"])
    #normalize by user
    if timeF != 0:
        for k in names_t:
            names_t[k] = min(1, names_t[k]/timeF)
        for k in names_l:
            names_l[k] = min(1, names_l[k]/timeF)
    #normalize by user
    if viewF != 0:
        for k in names_c:
            names_c[k] = min(1, names_c[k]/viewF)
    ret = []
    ret.extend(v.transform(names_c)[0])
    ret.extend(v.transform(names_t)[0])
    return ret

#morning noon evening night and their mapping to the hours
daytime = {"m":(6,11),"nn":(12,17),"ev":(18,23),"ni":(0,5)}

#to push into query
attr_names = {"os":"$os", "deviceKind":"$deviceKind", "VODTimeCum":"$VODTimeCum",
              "VODDetails":"$VODDetails", "RemindmeCount":"$RemindmeCount", "ShareitemCount":"$ShareitemCount",
              "FavoritedCount":"$FavoritedCount", "WatchlaterCount":"$WatchlaterCount",
             "ItemselectedCount":"$ItemselectedCount", "ItemselectedDetails":"$ItemselectedDetails"}

#dict to contain the counts for the channels
channels = {}

devices = {"Tablet":1, "Phone":2, 'undefined':0}
os = {"Android":1, "iPhone OS":2}
counts = ["RemindmeCount", "ShareitemCount", "FavoritedCount", "WatchlaterCount"]

#array for printing attr names
labels = []

mongo = pymongo.MongoClient()

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
cursor = mongo.get_database('7tv')

devs = {}
dev_cnt = 0

query = [
    {"$unwind":
         "$LoginUserIDs"},
    {"$group":{
        "_id": "$LoginUserIDs",
        "count" : {"$sum":1},
        #"VODDetails":{"$push":"$VODDetails"},
        "os":{"$push":"$os"},
        "deviceKind":{"$push":"$deviceKind"},
        "VODDetails":{"$push":"$VODDetails"},
        "VODTimeCum":{"$push":"$VODTimeCum"},
        "RemindmeCount":{"$push":"$RemindmeCount"},
        "ShareitemCount":{"$push":"$ShareitemCount"},
        "FavoritedCount":{"$push":"$FavoritedCount"},
        "WatchlaterCount":{"$push":"$WatchlaterCount"},
        #"info":{"$push":attr_names},
        "dev":{"$push":"$distinctID"}
        #"VOD":{"$push":"$VODDetails"}
    }},
    {"$match":{
        "VODDetails": {'$exists': 'true' },
        #"info.lastSeen": {'$exists': 'true' },
        #"info.FavoritedDetails" : {'$exists': 'true' },
        "count" : {"$gt":1},
        #"lastSeen" : {"$gt":min_time}
    }},
    {"$limit": 1000}
]


spn = 0
#***************************************************************************************************
if 0:
  samples = cursor.get_collection("withLogin").aggregate(query, allowDiskUse=True)
  for ijk in range(1,3):
    devs = {}
    dev_cnt = 0
    users = {}
    user_cnt = 0
    for d in samples:
        spn += 1
        if spn >= 500*ijk:
            break
        #get user id
        if isinstance(d["_id"], list):
            users.update({d["_id"][0]:user_cnt})
        else:
            users.update({d["_id"]:user_cnt})

        user_cnt += 1
        viewsCum = 0 #views of user
        timeCum = 0 #time of watches
        if "VODTimeCum" in d:
                for j in d["VODTimeCum"]:
                    timeCum += j
        if "VODDetails" in d:
            for j in d["VODDetails"]:
                viewsCum += len(j)

        # for all devices filling in the device's information
        for i in range(len(d["dev"])):
            devs.update({d["dev"][i]:{"user":d["_id"]}})
            if len(d["os"]) > i:
                devs[d["dev"][i]].update({"os":d["os"][i]})
            if len(d["deviceKind"]) > i:
                devs[d["dev"][i]].update({"deviceKind":d["deviceKind"][i]})
            if len(d["VODDetails"]) > i:
                devs[d["dev"][i]].update({"VODDetails":d["VODDetails"][i]})
            devs[d["dev"][i]].update({"timeFrac":timeCum})
            devs[d["dev"][i]].update({"viewFrac":viewsCum})
            dev_cnt += 1
        if (ijk == 1):
            # filling in the array with channels' names
            if "VODDetails" in d and len(d["VODDetails"]) != 0:
                for entry in d["VODDetails"][0]:
                    if "channel" in entry:
                        channels.setdefault(entry["channel"],1)
    print('exit loop')

    v = DictVectorizer(sparse=False)
    v.fit_transform(channels)

    f = open("features_new_" + str(ijk) + ".txt", "w", encoding = "UTF-8")

    #creating an array of labels for visualizing the tree
    labels = ["id"]

    #labels.extend([x + "_clicks" for x in channels])
    #labels.extend([x + "_time" for x in channels])
    #labels.extend([str(x) + "h" for x in range(0,25)])
    labels.extend(['mon','tue','wed','thu','fri','sat','sun'])
    labels.extend(['holiday'])
    '''
    for x in counts:
        labels.append(x + "_present")
        labels.append(x + "_count")
    '''
    labels.extend(["os", "deviceKind"])
    #labels.extend(["deviceKind"])
    f.writelines(["%s\t" % x  for x in labels])
    f.write("\n")

    for d in devs:
        #adding user id
        if isinstance(devs[d]["user"], list):
            resultingVector = [users[devs[d]["user"][0]]]
        else:
            resultingVector = [users[devs[d]["user"]]]

        nu = 0
        if "VODDetails" in devs[d]:
            #adding channel distributions
            #resultingVector += VODs_time_ratio(devs[d]["VODDetails"],v,channels,devs[d]["timeFrac"],devs[d]["viewFrac"])
            #hours of day distribution
            #resultingVector += dealWithTime(devs[d]["VODDetails"],devs[d]["viewFrac"])
            resultingVector += dealWithWeekend(devs[d]["VODDetails"],devs[d]["viewFrac"])
            #weekdays distribution
            resultingVector += dealWithDate(devs[d]["VODDetails"],devs[d]["viewFrac"])
        else:
            resultingVector += [0.0 for x in range(7+25+1)]# + 2*len(channels))]
            nu = 1
        '''
        #adding presence and amount of FavoritedCount, WatchLaterCount etc
        for attr in counts:
            if attr in devs[d]:
                resultingVector += [1] + devs[d][attr]
            else:
                resultingVector += [0,0]
        '''
        #adding os and devKind

        if "os" in devs[d]:
            resultingVector += [os[devs[d]["os"]]]
        else:
            resultingVector += [0]

        if "deviceKind" in devs[d]:
            resultingVector += [devices[devs[d]["deviceKind"]]]
        else:
            resultingVector += [0]
        if not(nu):
            f.writelines(["%s\t" % x  for x in resultingVector])
            f.write("\n")
        #print(len(resultingVector))

    f.close()

#***************************************************************************

f = open('features_new_1.txt', encoding = 'utf-8')
f1 = open('features_new_2.txt', encoding = 'utf-8')
X_train = []
y_train = []
labels = [] #to visualize nodes
X_test = []
y_test = []

for line in f:
    if line != "\n":
        list = line.strip().split('\t')
        if list[0] == "id":
            labels = list[1:]
        else:
            X_train.append([float(x) for x in list[1:]])
            y_train.append(int(list[0]))


for line in f1:
    if line != "\n":
        list = line.strip().split('\t')
        if list[0] == "id":
            labels = list[1:]
        else:
            X_test.append([float(x) for x in list[1:]])
            y_test.append(int(list[0]))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#clf = tree.DecisionTreeClassifier(min_samples_leaf = 10)

#from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
hi = SelectKBest(chi2)
X_train = hi.fit_transform(X_train, y_train)
X_test = hi.transform(X_test)

best = [1,2,3,5,10,30] #min numbers of values there should be in the list (min possible users per class)
for k in best:
    #clf = tree.DecisionTreeClassifier(min_samples_leaf = k)
    clf = tree.DecisionTreeClassifier(min_samples_leaf = k)
    clf = clf.fit(X_train, y_train)
    leaves = clf.predict(X_test)
    #check quality - amount of errors
    err = count_err(y_test, leaves, X_test)
    #err = precision_score(y_test, leaves)
    print(err)
    print("----")
print(len(y_test))

#export the tree
tree.export_graphviz(clf, out_file='tree1.dot', feature_names = labels) #produces dot file

