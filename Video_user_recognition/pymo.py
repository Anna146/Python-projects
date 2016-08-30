from datetime import datetime
import pprint
from sklearn import cluster
import operator
from sklearn.feature_extraction import FeatureHasher, DictVectorizer

__author__ = 'Asus'

def dt(u):
    return datetime.utcfromtimestamp(u)


def dealWithDetails(details):
    res = [0 for x in range(4)] #time,duration,channel,clipID
    mfhs = dict()
    chs = dict()
    clps = dict()
    durCum = 0
    for det in details:
        if "time" in det:
            d = dt(det['time'])
            h = round(((((d.hour * 60) + d.minute) * 60) + d.second) / (60.0 * 60))
            if (h in mfhs):
                mfhs[h] += 1
            else:
                mfhs[h] = 1
        else:
            res[0] = -1
        if "duration" in det:
            durCum += det["duration"]
        else:
            res[1] = -1
        if "channel" in det:
            ch = det["channel"]
            if (ch in chs):
                chs[ch] += 1
            else:
                chs[ch] = 1
        else:
            res[2] = -1
        if "clipID" in det:
            cl = det["clipID"]
            if (cl in clps):
                clps[cl] += 1
            else:
                clps[cl] = 1
        else:
            res[3] = -1
    if res[0] != -1:
        srt = sorted(mfhs.items(), key=operator.itemgetter(1))
        res[0] = srt[-1][0]
    if res[1] != -1:
        res[1] = durCum / len(details)
    if res[2] != -1:
        srt = sorted(chs.items(), key=operator.itemgetter(1))
        res[2] = srt[-1][0]
    if res[3] != -1:
        srt = sorted(clps.items(), key=operator.itemgetter(1))
        srt = sorted(clps.items(), key=lambda x: x[1])
        res[3] = srt[-1][0]
    return res


import pymongo
mongo = pymongo.MongoClient()
min_time = 1450137600
max_time = 1451606400
tresh_low = 20
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
cursor = mongo.get_database('7tv')
att_names = ["deviceKind", ]
#print(cursor.collection_names())
#print(cursor.withLogin.find().next())
#cursor2 = cursor.get_collection("withLogin").aggregate([{"$unwind":"$LoginUserIDs"}, {"$group":{"_id":"$LoginUserIDs", "os":{"$push":"$os"}, "VODDetails":{"$push":"$VODDetails"}, "installDate":{"$push":"$installDate"}, "deviceKind":{"$push":"$deviceKind"}, "VODTimeCum":{"$push":"$VODTimeCum"}}}],allowDiskUse=True)
query = [
    {"$unwind":
         "$LoginUserIDs"},
    {"$group":{
        "_id": "$LoginUserIDs",
        "count" : {"$sum":1},
        "compF":{"$push":"$VODTimeCum"}
        #"VOD":{"$push":"$VODDetails"}
    }},
    {"$match":{
        "compF": {"$ne": []},
        "RemindmeCount": {"$ne": 0},
        "count" : {"$gt":1},
    }}
]
top = 0
for d in cursor.get_collection("withLogin").aggregate(query, allowDiskUse=True):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(d)
    top += 1
    if top > 20:
        break
#cursor2 = cursor.get_collection("withLogin").aggregate([{"$unwind":"$LoginUserIDs"}, {"$group":{"_id":"$LoginUserIDs", "VODDetails":{"$push":"$VODDetails"}}}],allowDiskUse=True)
#while (cursor2.next()):
#   print(cursor2.next())
'''
i = 0
y = []
big = []
#k_means = cluster.KMeans(n_clusters=5000)
d = cursor2.next()
fh = DictVectorizer()
try:
    while 1:
        print(d)
        #break
        feat = []
        feat.append(d["os"][0]) if "os" in d else feat.append(-1)
        feat.append(d["InstallDate"][0]) if "InstallDate" in d else feat.append(-1)
        feat.append(d["deviceKind"][0]) if "deviceKind" in d else feat.append(-1)
        feat.append(d["lastSeen"][0]) if "lastSeen" in d else feat.append(-1)
        feat.append(d["VODTimeCum"][0]) if "VODTimeCum" in d else feat.append(-1)
        feat.extend(dealWithDetails(d["VODDetails"][0])) if "VODDetails" in d else feat.extend([-1 for x in range(4)])
        feat.append(d["LoginFirst"][0]) if "LoginFirst" in d else feat.append(-1)
        feat.append(d["LoginLast"][0]) if "LoginLast" in d else feat.append(-1)
        feat.append(d["RemindmeCount"][0]) if "RemindmeCount" in d else feat.append(-1)
        feat.append(d["ShareitemCount"][0]) if "ShareitemCount" in d else feat.append(-1)
        feat.append(d["FavoritedCount"][0]) if "FavoritedCount" in d else feat.append(-1)
        feat.append(d["WatchlaterCount"][0]) if "WatchLaterCount" in d else feat.append(-1)
        feat.extend(dealWithDetails(d["WatchlaterDetails"][0])) if "WatchLaterDetails" in d else feat.extend([-1 for x in range(4)])
        feat.append(d["TrialUserIDs"][1]) if "TrialUserIDs" in d and len(d["TrialUserIDs"]) > 1 else feat.append(-1)
        feat.append(d["TrialUserIDs"][0]) if "TrialUserIDs" in d else feat.append(-1)
        feat.append(d["SubscrUserIDs"][1]) if "SubscrUserIDs" in d else feat.append(-1)
        feat.append(d["SubscrUserIDs"][0]) if "SubscrUserIDs" in d else feat.append(-1)
        feat.append(d["Pl"][0]) if "Pl" in d else feat.append(-1)
        feat.append(d["PlFirst"][0]) if "PlFirst" in d else feat.append(-1)
        feat.append(d["PlLast"][0]) if "PlLast" in d else feat.append(-1)
        feat.append(d["TralFirst"][0]) if "TrialFirst" in d else feat.append(-1)
        feat.append(d["TrialLast"][0]) if "TrialLast" in d else feat.append(-1)
        feat.append(d["SubscrFirst"][0]) if "SubscrFirst" in d else feat.append(-1)
        feat.append(d["SubscrLast"][0]) if "SubscrLast" in d else feat.append(-1)
        #feat.append(d.get('key', [-1])[0])
        feat.append(d["ItemselectedCount"][0]) if "ItemselectedCount" in d else feat.append(-1)
        feat.extend(dealWithDetails(d["RemindmeDetails"][0])) if "RemindmeDetails" in d else feat.extend([-1 for x in range(4)])
        feat.extend(dealWithDetails(d["ShareitemDetails"][0])) if "ShareitemDetails" in d else feat.extend([-1 for x in range(4)])
        feat.extend(dealWithDetails(d["FavoritedDetails"][0])) if "FavoritedDetails" in d else feat.extend([-1 for x in range(4)])
        feat.extend(dealWithDetails(d["ItemSelectedDetails"][0])) if "ItemSelectedDetails" in d else feat.extend([-1 for x in range(4)])
        d = cursor2.next()
        #feat = [feat]
        #break
        big.append(feat)
except:
    print(y)
#res = fh.fit_transform(big)
#k_means.fit(big)
print("finished")
    '''



'''
from collections import Counter
distinct_ids_per_user = Counter()
for d in cursor.peopleTural.find():
    login_user_ids = d['lastSeen']
    for user_id in login_user_ids:
        distinct_ids_per_user[user_id] += 1
print(len(distinct_ids_per_user))
users_with_two_or_more_devices = 0
for k, v in distinct_ids_per_user.iteritems():
    if v >= 2:
        users_with_two_or_more_devices += 1
print("Percentage of users with 2 or more devices is %.2f%%" % \
    (100.0 * users_with_two_or_more_devices / len(distinct_ids_per_user)))
'''
