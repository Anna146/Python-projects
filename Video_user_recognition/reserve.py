
# coding: utf-8

# In[12]:

import numpy as np


# In[13]:
from pandas.util.testing import DataFrame

import pymongo
mongo = pymongo.MongoClient()
cursor = mongo.get_database('7tv')


# In[14]:
'''
for x in cursor.withLogin.find({"LoginUserIDs": {"$ne": None}})[:10]:
    print(x['LoginUserIDs'])
'''

# In[15]:

devices_per_user = 2
pipeline = [
    {"$match": {
#        "$or": [{"$gt": ["$VODV", 0]}, {"$gt": ["$ItemselectedCount", 0]}]
        "$or": [{"VODV": {"$gt": 0}}, {"ItemselectedCount": {"$gt": 0}}],
    }},
    {"$unwind": "$LoginUserIDs"},
    {"$group": {
        "_id": "$LoginUserIDs",
        "count": {"$sum": 1}
    }},
    {"$match": {"count": {"$gte": devices_per_user}}}
]


# In[11]:

proper_user_ids = {}

i = 0

for x in cursor.withLogin.aggregate(pipeline):
    proper_user_ids[x["_id"][0]] = x["count"]
    
print(len(proper_user_ids))


# In[69]:

compression_levels = ("channel", "showID", "clipID")

def compress_vods(vods):
    compressed_data = []
    for v in sorted(vods, key=lambda x: x['time']):
        cur_dict = {
            "channel": v['channel'],
            "showID": v.get('showID'),
            "clipID": v.get('clipID'),
            "start_time": int(v["time"] - v["duration"]),
            "end_time": v["time"],
            "duration": v['duration']
        }
        
        def _is_the_same_sequence(d1, d2):       
            for lv in compression_levels:
                if d1[lv] != d2[lv]:
                    return False
            return True
    
        if len(compressed_data) > 0:
            prev_dict = compressed_data[-1]
            if _is_the_same_sequence(prev_dict, cur_dict):
                compressed_data[-1] = cur_dict
            else:
                compressed_data.append(cur_dict)
        else:
            compressed_data.append(cur_dict)
    return compressed_data


# In[70]:

def counter_to_percent(cnt):
    size = float(sum(cnt.values()))
    unit_cnt = {}
    for k, v in cnt.items():
        unit_cnt[k] = v / size
    return unit_cnt


# In[71]:

from collections import Counter
import datetime

DAYS_PER_WEEK = 7
HOURS_PER_DAY = 24

class Device(object):
    def __init__(self):
        self.uid = None
        self.time_per_channel = Counter()
        self.favs_per_channel = Counter()
        self.prefered_hours = Counter()
        self.prefered_days_of_week = Counter()
        self.os = None
        self.kind = None
        self.views_per_week = 0
        self.views = 0
        self.avg_dur = 0
    
    @classmethod
    def create(cls, uid, d):
        device = cls()
        
        # common
        device.uid = uid
        device.kind = d.get("deviceKind")
        device.os = d.get("os")
        
        # favs_per_channel
        for rec in d.get('FavoritedDetails', []):
            channel = rec['channel']
            action = rec['actionType']
            if action == 'add':
                device.favs_per_channel[channel] += 1
            elif action == 'delete' and device.favs_per_channel[channel] > 0:
                device.favs_per_channel[channel] -= 1
                if device.favs_per_channel[channel] == 0:
                    del device.favs_per_channel[channel]

        start_act = datetime.max
        end_act = datetime.min

        #device.favs_per_channel = counter_to_percent(device.favs_per_channel)
                
        for rec in compress_vods(d.get('VODDetails', [])):
            channel = rec['channel']
            duration = float(rec['duration'])
            device.avg_dur += duration
            device.views += 1
            
            # time_per_channel
            if duration > 0:
                device.time_per_channel[channel] += duration
            
            start_dt = datetime.datetime.fromtimestamp(rec['start_time'])
            end_dt = datetime.datetime.fromtimestamp(rec['end_time'])
            
            start_day = start_dt.weekday()
            end_day = end_dt.weekday()

            if start_dt < start_act:
                start_act = start_dt
            if end_dt > end_act:
                end_act = end_dt
            
            # prefered_days_of_week
            def _get_day_index(day):
                #return day
                return "w" if day < 5 else "h"
            
            if start_day == end_day:
                device.prefered_days_of_week[_get_day_index(start_day)] += 1
            elif start_day < end_day:
                for d in range(start_day, end_day + 1):
                    device.prefered_days_of_week[_get_day_index(d)] += 1
            else:
                for d in range(start_day, end_day + DAYS_PER_WEEK):
                    device.prefered_days_of_week[_get_day_index(d % DAYS_PER_WEEK)] += 1
                    
            # prefered_hours
            def _get_hours_index(hour):
                #return hour
                r = [4, 10, 15, 19, 24]
                for i, hl in enumerate(r):
                    if hour < hl:
                        return i
            
            if start_dt.hour == end_dt.hour:
                device.prefered_hours[_get_hours_index(start_dt.hour)] += 1
            elif start_dt.hour < end_dt.hour:
                for h in range(start_dt.hour, end_dt.hour + 1):
                    device.prefered_hours[_get_hours_index(h)] += 1
            else:
                for h in range(start_dt.hour, end_dt.hour + HOURS_PER_DAY):
                    device.prefered_hours[_get_hours_index(h % HOURS_PER_DAY)] += 1
                    
        #device.time_per_channel = counter_to_percent(device.time_per_channel)
        #device.prefered_days_of_week = counter_to_percent(device.prefered_days_of_week)
        #device.prefered_hours = counter_to_percent(device.prefered_hours)
        device.views_per_week = ((end_act - start_act).days) / 7 / device.views
        device.avg_dur /= device.views
        return device
            
    def to_dict(self):
        res = {
            u"uid": self.uid,
            u"os": self.os,
            u"kind": self.kind,
        }
        
        for k, v in self.favs_per_channel.items():
            res[u"fv:%s" % k] = v
        for k, v in self.time_per_channel.items():
            res[u"tc:%s" % k] = v
        for k, v in self.prefered_hours.items():
            res[u"h:%s" % k] = v
        for k, v in self.prefered_days_of_week.items():
            res[u"d:%s" % k] = v
            
        return res 

class Video(object):
    def __init__(self, clip, show, chan):
        self.clipID = clip
        self.showID = show
        self.views_cnt = 0
        self.on_channel = [chan]
        self.on_channel_cnt = 1

    def add_view(self, chan):
        self.views_cnt += 1
        if not(chan in self.on_channel):
            self.on_channel.append(chan)
            self.on_channel_cnt = len(self.on_channel)


class User(object):
    def __init__(self, id):
        self.uid = id
        self.devCount = 0
        self.devices = []
        self.oss = Counter()
        self.kinds = Counter()
        self.time_per_channel = Counter()
        self.favs_per_channel = Counter()
        self.prefered_hours = Counter()
        self.prefered_days_of_week = Counter()

    def add_device(self, Device):
        self.devices.append(Device)
        self.devCount += 1
        self.kinds[Device.kind] += 1
        self.oss[Device.os] += 1

        for itm,cnt in Device.favs_per_channel.items():
            self.favs_per_channel[itm] += cnt
        for itm,cnt in Device.prefered_days_of_week.items():
            self.prefered_days_of_week[itm] += cnt
        for itm, cnt in Device.prefered_hours.items():
            self.prefered_hours[itm] += cnt
        for itm,cnt in Device.time_per_channel.items():
            self.time_per_channel[itm] += cnt

    def count_stats(self):
        self.time_per_channel = counter_to_percent(self.time_per_channel)
        self.prefered_days_of_week = counter_to_percent(self.prefered_days_of_week)
        self.prefered_hours = counter_to_percent(self.prefered_hours)
        self.favs_per_channel = counter_to_percent(self.favs_per_channel)

    def to_dict(self):
        self.count_stats()
        res = {
            u"uid": self.uid,
            #u"os": self.os,
            #u"kind": self.kind,
        }
        res[u"os"] = max(self.oss.items(), key=lambda x: x[1])[0]
        res[u"kind"] = max(self.kinds.items(), key=lambda x: x[1])[0]
        for k, v in self.favs_per_channel.items():
            res[u"fv:%s" % k] = v
        for k, v in self.time_per_channel.items():
            res[u"tc:%s" % k] = v
        for k, v in self.prefered_hours.items():
            res[u"h:%s" % k] = v
        for k, v in self.prefered_days_of_week.items():
            res[u"d:%s" % k] = v

        return res


# ## Preparing data

# In[72]:

pipeline = [
    {"$match": {
        "$or": [{"VODV": {"$gt": 0}}, {"ItemselectedCount": {"$gt": 0}}],
    }},
    {"$unwind": "$LoginUserIDs"},
    #{"$match": {"LoginUserIDs": {"$in": proper_user_ids.keys()}}}
]

j = 0

import pprint

user_ids = {}
users_and_objects = {}
users_dict = {}

for data in cursor.withLogin.aggregate(pipeline):
    #print(data["LoginUserIDs"][0])
    iid = data["LoginUserIDs"][0]
    if iid in proper_user_ids.keys():
        if not(iid in users_dict):
            uid = user_ids.setdefault(data['LoginUserIDs'][0], len(user_ids))
            #uid = data['LoginUserIDs']
            usr = User(uid)
            users_dict.setdefault(iid, usr)
        else:
            uid = users_dict[iid].uid
        obj = Device.create(uid, data)
        users_dict[iid].add_device(obj)
    j += 1

print(len(users_dict))

'''
for data in cursor.withLogin.aggregate(pipeline):
    #print(data["LoginUserIDs"][0])
    if data["LoginUserIDs"][0] in proper_user_ids.keys():
        uid = user_ids.setdefault(data['LoginUserIDs'][0], len(user_ids))
        #uid = data['LoginUserIDs']
        obj = Device.create(uid, data)
        users_and_objects.setdefault(uid, []).append(obj)
    pprint.pprint(data)
    break
    j += 1
    if j > 5000:
        break
'''

#print(len(users_and_objects))


# ### Training and testing

# In[73]:

testing = []
training = []
import numpy
import pprint

users = []
Y = []
X = []
probability = 0.5

'''
fields = []

for uid, objs in users_and_objects.items():
    for obj in objs:
        fields.extend(list(obj.to_dict().keys()))
    fields = list(filter(lambda x: x!= "uid", fields))

print(fields)

for uid, objs in users_and_objects.items():
    #take users with >2 devices
    if len(objs) < 2:
        continue
    user = {}
    user_res = {}
    vec = []
    for field in fields:
        user.setdefault(field,[])
        for obj in objs:
            dobj = obj.to_dict()
            if field in list(dobj.keys()):
                user.setdefault(field,[]).append(dobj[field])
    for name, arr in user.items():
        #print(arr)
        if len(arr) == 0:
            arr.append(0)
        avg = 0.0 #numpy.mean(arr)
        vec.append(avg)
        if name == u"os" or name == u"kind":
            user_res[name] = arr[0]
        else:
            user_res[name + "_avg"] = float(avg)
        #device from which the longest of this channel was watched
        #kind = numpy.max(np.array(arr).argmax())
        #kind_num = 0 if objs[kind].to_dict()["kind"] == "undefined" else 1 if objs[kind].to_dict()["kind"] == u'Phone' else 2
        #user_res[name + "_kind"] = kind_num
    user_res["uid"] = uid
    X.append(vec)
    Y.append(uid)
    #pprint.pprint(user_res)
    #break
    users.append(user_res)

import sklearn.metrics.pairwise

X = numpy.matrix(X)
dsts = sklearn.metrics.pairwise.pairwise_distances(X)
'''
'''
max_val = [0,1,0]

from scipy.spatial import distance

for i in range(len(dsts)):
    for j in range(i+1,len(dsts[i])):
        cell = dsts[i][j]
        if cell > max_val[2]:
            max_val = [i,j,cell]
#print(max_val)
i = 0
for f in users[max_val[0]].keys():
    print(str(f) + ": " + " " +  str(list(users[max_val[0]].values())[i]) + " " + str(list(users[max_val[1]].values())[i]))
    #print(str(f) + ": " + " " + str(distance.euclidean(list(users[max_val[0]].values())[i],list(users[max_val[1]].values())[i])))
    i += 1
'''
for uid, usr in users_dict.items():
    #do I copy or create a link here?
    objs = usr.devices
    if np.random.rand() >= 0.5:
        test_obj = objs.pop(np.random.randint(len(objs)))
        testing.append(test_obj.to_dict())
    #the error is here
    training.append(usr.to_dict())
'''
for uid, usr in users_dict.items():
    #do I copy or create a link here?
    objs = usr.devices
    if np.random.rand() >= 0.5:
        testing.append(usr.to_dict())

    for obj in objs:
        training.append(obj.to_dict())

        '''
'''

for us in users:
    if np.random.rand() >= 0.5:
        testing.append(us)
    else:
        training.append(us)
'''
#print(training)
        
print(len(training), len(testing))

'''
# In[74]:

# The usual preamble
#get_ipython().magic('matplotlib inline')
'''
import pandas as pd
import matplotlib.pyplot as plt


# In[75]:

train_df = pd.DataFrame.from_dict(training)
#print(train_df.shape)
#print(train_df.head())


# In[76]:

test_df = pd.DataFrame.from_dict(testing)
#print test_df.shape
#print(test_df.head())


# In[77]:

common_columns = list(set(train_df.columns).intersection(test_df.columns))
train_df = train_df[common_columns]
test_df = test_df[common_columns]

#print(common_columns)

# In[78]:

# prepare training data
train_df.loc[train_df["os"] == u'Android', "os"] = 0
train_df.loc[train_df["os"] == u'iPhone OS', "os"] = 1

train_df.loc[train_df["kind"] == u'undefined', "kind"] = 0
train_df.loc[train_df["kind"] == u'Phone', "kind"] = 1
train_df.loc[train_df["kind"] == u'Tablet', "kind"] = 2

train_df = train_df.fillna(0)
#print(train_df.dtypes)
#train_df1 = train_df.groupby(['uid']).sum()
'''
from scipy import spatial
similarity = {}

for index, i in train_df1.iterrows():
    for index2, j in train_df1.iterrows():
        if index != index2:
            similarity[str(index) + " " + str(index2)] = spatial.distance.cosine(i.values, j.values)


diff = sorted(similarity.items(), key=lambda x: x[1])
least_comm = diff[-1][0].split(" ")
most_comm = diff[0][0].split(" ")
print(train_df[train_df["uid"] == int(least_comm[0])])
'''
# In[79]:

# prepare testing data

test_df.loc[test_df["os"] == u'Android', "os"] = 0
test_df[test_df["os"] == u'Android'] = 0
test_df.loc[test_df["os"] == u'iPhone OS', "os"] = 1

test_df.loc[test_df["kind"] == u'undefined', "kind"] = 0
test_df.loc[test_df["kind"] == u'Phone', "kind"] = 1
test_df.loc[test_df["kind"] == u'Tablet', "kind"] = 2
test_df.head()

test_df = test_df.fillna(0)

# In[81]:

print(train_df.head())
print(test_df.head())

from sklearn.feature_selection import SelectKBest, f_classif, chi2

#train_df = pd.DataFrame.from_dict(users)

predictors = train_df.columns.tolist()
prediction = "uid"
predictors.remove(prediction)

# Perform feature selection
selector = SelectKBest(f_classif, k=1)
selector.fit(train_df[predictors], train_df[prediction])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
print(scores)
# Plot the scores
plt.figure(figsize=(20,10))
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# ## Applying classification

# In[82]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[143]:

good_features = [
    u'tc:sat1', 
    u'fv:pro7', 
    u'fv:prosieben',
    u'tc:prosieben', 
    u'fv:sat1', 
    u'fv:prosiebenmaxx', 
    #u'd:h', 
    u'fv:sat1gold', 
    u'h:4', 
    u'h:1', 
    u'h:0', 
    u'h:3', 
    u'h:2', 
    u'tc:sixx', 
    #u'd:w', 
    u'tc:kabeleins', 
    u'tc:pro7', 
    u'fv:sixx', 
    u'kind', 
    #u'tc:prosiebenmaxx', 
    u'fv:kabeleins', 
    u'tc:sat1gold', 
    u'tc:kabel1', 
    u'fv:kabel1',
    #u'os'
]

# good_features = list(train_df.columns.values)
# good_features.remove('os')
# good_features.remove('uid')

train_X = train_df[predictors][:]
train_y = train_df[prediction]

test_X = test_df[predictors][:]
test_y = test_df[prediction]


# In[147]:
from sklearn import metrics

#clf = RandomForestClassifier(n_estimators=200, random_state=1)
clf = DecisionTreeClassifier(random_state=1)
clf.fit(train_X, train_y)
predicted_y = clf.predict(test_X)
print(metrics.precision_recall_fscore_support(test_y, predicted_y, average='weighted'))
print(predicted_y[test_y.values == predicted_y].size / float(len(predicted_y)))


# In[148]:
'''
from sklearn import tree
tree.export_graphviz(clf, feature_names=train_X.columns)
'''

# ## Idea with clustering

# In[149]:
'''
dataset = []
for x in users_and_objects.values():
    for v in x:
        dataset.append(v.to_dict())
ds_df = pd.DataFrame(dataset)

ds_df.loc[ds_df["os"] == u'Android', "os"] = 0
ds_df.loc[ds_df["os"] == u'iPhone OS', "os"] = 1

ds_df.loc[ds_df["kind"] == u'undefined', "kind"] = 0
ds_df.loc[ds_df["kind"] == u'Phone', "kind"] = 1
ds_df.loc[ds_df["kind"] == u'Tablet', "kind"] = 2

ds_df = ds_df.fillna(0)

print ds_df.shape
ds_df.head()


# In[150]:

X = ds_df[predictors]


# In[162]:

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5).fit(X)


# In[163]:

y = db.labels_
len(set(y)) - (1 if -1 in y else 0)


# In[164]:

ds_df["klass"] = y


# In[165]:

from sklearn.cross_validation import cross_val_score
clf = RandomForestClassifier(n_estimators=10, random_state=1)
scores = cross_val_score(clf, X, y)
print scores
print scores.mean()


# In[166]:

ds_df[ds_df.klass == 5][["uid", "klass"]].sort_values("uid")


'''