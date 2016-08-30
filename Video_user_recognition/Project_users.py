# coding: utf-8
import numpy as np
from pandas.util.testing import DataFrame

import pymongo
mongo = pymongo.MongoClient()
cursor = mongo.get_database('7tv')

devices_per_user = 3
pipeline = [
    {"$match": {
        "$or": [{"VODV": {"$gt": 0}}, {"ItemselectedCount": {"$gt": 0}}],
    }},
    {"$unwind": "$LoginUserIDs"},
    {"$group": {
        "_id": "$LoginUserIDs",
        "count": {"$sum": 1}
    }},
    {"$match": {"count": {"$gte": devices_per_user}}}
]

proper_user_ids = {}

for x in cursor.withLogin.aggregate(pipeline):
    proper_user_ids[x["_id"][0]] = x["count"]
    
print(len(proper_user_ids))

#_______________________

devices_per_user = 2
pipeline2 = [
    {"$match": {
        "$or": [{"VODV": {"$gt": 0}}, {"ItemselectedCount": {"$gt": 0}}],
    }},
    {"$unwind": "$LoginUserIDs"},
    {"$group": {
        "_id": "$LoginUserIDs",
        "count": {"$sum": 1}
    }},
    {"$match": {"count": {"$gte": 2}}}
]

proper_user_ids_two = {}

for x in cursor.withLogin.aggregate(pipeline2):
    proper_user_ids_two[x["_id"][0]] = x["count"]

print(len(proper_user_ids_two))

#__________________________________

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

def counter_to_percent(cnt):
    size = float(sum(cnt.values()))
    unit_cnt = {}
    for k, v in cnt.items():
        unit_cnt[k] = v / size
    return unit_cnt

from collections import Counter
import datetime

DAYS_PER_WEEK = 7
HOURS_PER_DAY = 24

#to find populer videos
#td - popularity by device, by user
class Video(object):
    def __init__(self, clip, show, chan, dur):
        self.clipID = clip
        self.showID = show
        self.views_cnt = 1
        #chanels this clip appeared on
        self.on_channel = [chan]
        #count of channels this clip appeared on
        self.on_channel_cnt = 1
        #duration
        self.dur = dur

    def add_view(self, chan):
        self.views_cnt += 1
        if not(chan in self.on_channel):
            self.on_channel.append(chan)
            self.on_channel_cnt = len(self.on_channel)

#min minutes to distinguish sessions
diff_between_sessions = 10

#session - a solid time period when user was watching non-interrupted
class Session(object):
    def __init__(self,st,et,ch):
        self.start_t = st
        self.end_t = et
        self.video_cnt = 1
        self.duration = st - et
        self.chanels = set()
        self.chanels.add(ch)

    def add_view(self, new_end,ch):
        self.video_cnt += 1
        self.end_t = new_end
        self.duration = self.start_t - new_end
        self.chanels.add(ch)

    #@staticmethod
    def check_end_session(self,new_start):
        if (new_start - self.end_t).seconds/60 > diff_between_sessions:
            return True
        return False

videos = {}

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
        self.views = 0 #number of distinct records in VODDetails
        self.avg_dur = 0 #of one view
        self.sessions = []  #array of sessions to count statistics
        self.sessions_per_day = 0
        self.views_per_session = 0
        self.chann_numb = 0 #number of watched channels
        self.video_len = [] #the lengths of watch videos for statistics count
        self.avg_len = 0
        self.watch_percent = [] #watched percent of the total length of the video
        self.avg_percent = 0
        self.rewatch = 0 #count of the times the user watched the same video multiple times

        self.fav_num = 0 #number of liked channels
        self.add_num = 0 #adds to FavoritedDetails
        self.del_num = 0 #deletes from FavoriteDetails
        self.distinct_fav_shows = 0 #number of fav shows
        self.fav_days = Counter() #days of week with most favs
        self.favs_per_day = 0 #favorited per_day
        self.session_duration = 0 #average session duration
        self.session_count = 0 #number of sessions
        self.time_between_sess = 0 #time between sessions

        #----experiment---- device distribution
        self.freq_dev_tab = {}
        self.freq_dev_phone = {}
        #self.pure_dev = {} - if user watched only from this kind of device

    @classmethod
    def create(cls, uid, d):
        device = cls()

        # common
        device.uid = uid
        device.kind = d.get("deviceKind")
        device.os = d.get("os")

        shows = set()
        # favs_per_channel
        for rec in d.get('FavoritedDetails', []):
            shows.add(rec["showID"])
            channel = rec['channel']
            action = rec['actionType']
            fav_day = datetime.datetime.fromtimestamp(rec['time']).weekday()
            device.fav_days[fav_day] += 1
            if action == 'add':
                device.favs_per_channel[channel] += 1
                device.add_num += 1
                device.favs_per_day += 1
            elif action == 'delete' and device.favs_per_channel[channel] > 0:
                device.del_num += 1
                device.favs_per_channel[channel] -= 1
                if device.favs_per_channel[channel] == 0:
                    del device.favs_per_channel[channel]

        device.distinct_fav_shows = len(shows)
        from datetime import date

        #start and end of VODDetaiils
        start_act = datetime.datetime.combine(date.max, datetime.datetime.min.time())
        end_act = datetime.datetime.combine(date.min, datetime.datetime.min.time())

        for rec in compress_vods(d.get('VODDetails', [])):
            channel = rec['channel']
            duration = float(rec['duration'])
            device.avg_dur += duration
            device.views += 1
            clips = set() #to store what he has already seen

            #-------experiment distribution
            if device.kind != None:
                if device.kind == "Tablet":
                    device.freq_dev_tab[rec["channel"]] = 1
                else:
                    device.freq_dev_phone[rec["channel"]] = 1
            #----------------

            #to count the popularity of videos
            if not(rec["clipID"] in videos.keys()):
                video = Video(rec["clipID"], rec["showID"], rec["channel"], rec["duration"])
                videos.setdefault(rec["clipID"], video)
            else:
                videos[rec["clipID"]].add_view(rec["channel"])

            #for videos watched multiple times
            if rec["clipID"] in clips:
                device.rewatch += 1
            else:
                clips.add(rec["clipID"])

            # time_per_channel
            if duration > 0:
                device.time_per_channel[channel] += duration
            
            start_dt = datetime.datetime.fromtimestamp(rec['start_time'])
            end_dt = datetime.datetime.fromtimestamp(rec['end_time'])
            if "clipLen" in rec:
                device.video_len.append(rec['clipLen']) #to count the average len later

            #check if a new session has started and create it or extend the current one
            if not(len(device.sessions) == 0):
                if device.sessions[len(device.sessions) - 1].check_end_session(start_dt):
                    device.time_between_sess += (device.sessions[len(device.sessions) - 1].end_t - start_dt).seconds
                    device.sessions.append(Session(start_dt, end_dt, rec["channel"]))
                    device.session_duration += device.sessions[len(device.sessions) - 1].duration.seconds
                else:
                    device.sessions[len(device.sessions) - 1].add_view(end_dt, rec["channel"])
            else:
                device.sessions.append(Session(start_dt, end_dt, rec["channel"]))

            #to count the average watched percent of the total video length
            if rec["duration"] != float(0) and "clipLen" in rec:
                device.watch_percent.append(rec["duration"]/rec["clipLen"])

            start_day = start_dt.weekday()
            end_day = end_dt.weekday()

            #change borders of VODDetails period
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

        if device.views != 0:
            device.views_per_week = ((end_act - start_act).days) / 7 / device.views
            device.avg_dur /= device.views
            device.views_per_session = len(device.sessions) / device.views
        if device.favs_per_day != 0:
            device.favs_per_day = (end_act - start_act).seconds / device.favs_per_day
        if len(device.video_len) != 0:
            device.avg_len = sum(device.video_len) / len(device.video_len)
        if len(device.watch_percent):
            device.avg_percent = sum(device.watch_percent) / len(device.watch_percent)
        if len(device.sessions) != 0:
            device.session_duration /= len(device.sessions)
            device.session_count = len(device.sessions)
            device.time_between_sess /= len(device.sessions)
        return device

    def count_stats(self):
        self.time_per_channel = counter_to_percent(self.time_per_channel)
        self.prefered_days_of_week = counter_to_percent(self.prefered_days_of_week)
        self.prefered_hours = counter_to_percent(self.prefered_hours)
        self.favs_per_channel = counter_to_percent(self.favs_per_channel)
        self.fav_days = counter_to_percent(self.fav_days)
        self.chan_num = len(self.favs_per_channel)


    def to_dict(self):
        self.count_stats()

        res = {
            u"uid": self.uid,
            u"os": self.os,
            u"kind": self.kind,

            u"views_per_week": self.views_per_week,
            u"avg_dur": self.avg_dur,
            u"sessions_per_day": self.sessions_per_day,
            u"views_per_session": self.views_per_session,
            u"avg_len": self.avg_len,
            u"avg_percent": self.avg_percent,
            u"rewatch": self.rewatch,
            }
        '''
            u"fav_num" : self.fav_num,
            u"add_num" : self.add_num,
            u"del_num" : self.del_num,
            u"distinct_fav_shows" : self.distinct_fav_shows,
            u"favs_per_day" : self.favs_per_day,
            u"session_duration" : self.session_duration,
            u"session_count" : self.session_count,
            u"time_between_sess" : self.time_between_sess}
        '''
        for k, v in self.favs_per_channel.items():
            res[u"fv:%s" % k] = v
        for k, v in self.time_per_channel.items():
            res[u"tc:%s" % k] = v
        for k, v in self.prefered_hours.items():
            res[u"h:%s" % k] = v
        for k, v in self.prefered_days_of_week.items():
            res[u"d:%s" % k] = v
        '''
        for k, v in self.fav_days.items():
            res[u"fav_d:%s" % k] = v
        '''

        #----experiment
        for k, v in self.freq_dev_tab.items():
            res[u"tab:%s" % k] = v
        for k, v in self.freq_dev_phone.items():
            res[u"ph:%s" % k] = v
        #----------------

        return res

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

        self.views_per_week = 0
        self.views = 0 #number of distinct records in VODDetails
        self.avg_dur = 0 #of one view
        self.sessions = []  #array of sessions to count statistics
        self.sessions_per_day = 0
        self.chann_numb = 0
        self.views_per_session = 0
        self.video_len = [] #the lengths of watch videos for statistics count
        self.avg_len = 0
        self.watch_percent = [] #watched percent of the total length of the video
        self.avg_percent = 0
        self.rewatch = 0 #count of the times the user watched the same video multiple times
        self.prefered_days_of_week = Counter()

        self.fav_num = 0 #number of liked channels
        self.add_num = 0 #adds to FavoritedDetails
        self.del_num = 0 #deletes from FavoriteDetails
        self.distinct_fav_shows = 0 #number of fav shows
        self.fav_days = Counter() #days of week with most favs
        self.favs_per_day = 0 #favorited per_day
        self.session_duration = 0 #average session duration
        self.session_count = 0 #number of sessions
        self.time_between_sess = 0 #time between sessions

        #----experiment---- device distribution
        self.freq_dev_tab = {}
        self.freq_dev_phone = {}


    def add_device(self, Device):
        self.devices.append(Device)
        self.devCount += 1
        self.kinds[Device.kind] += 1
        self.oss[Device.os] += 1
        self.views_per_week += Device.views_per_week
        self.views += Device.views
        self.avg_dur += Device.avg_dur
        self.sessions_per_day += Device.sessions_per_day
        self.views_per_session += Device.views_per_session
        self.avg_len += Device.avg_len
        self.avg_dur += Device.avg_dur
        self.avg_percent += Device.avg_percent
        self.fav_num += Device.fav_num
        self.add_num += Device.add_num
        self.del_num += Device.del_num
        self.distinct_fav_shows += Device.distinct_fav_shows
        self.favs_per_day += Device.favs_per_day
        self.session_count += Device.session_count
        self.session_duration += Device.session_duration
        self.time_between_sess += Device.time_between_sess

        #--------experiment
        for k,v in Device.freq_dev_tab.items():
            self.freq_dev_tab[k] = self.freq_dev_tab.get(k,1) + 1
        for k,v in Device.freq_dev_phone.items():
            self.freq_dev_phone[k] = self.freq_dev_phone.get(k,1) + 1

        #shouldn't I divide these counts by device number?
        for itm,cnt in Device.favs_per_channel.items():
            self.favs_per_channel[itm] += cnt
        for itm,cnt in Device.prefered_days_of_week.items():
            self.prefered_days_of_week[itm] += cnt
        for itm, cnt in Device.prefered_hours.items():
            self.prefered_hours[itm] += cnt
        for itm,cnt in Device.time_per_channel.items():
            self.time_per_channel[itm] += cnt
        for itm,cnt in Device.fav_days.items():
            self.fav_days[itm] += cnt

    def count_stats(self):
        self.time_per_channel = counter_to_percent(self.time_per_channel)
        self.prefered_days_of_week = counter_to_percent(self.prefered_days_of_week)
        self.prefered_hours = counter_to_percent(self.prefered_hours)
        self.favs_per_channel = counter_to_percent(self.favs_per_channel)
        self.chann_numb = len(self.favs_per_channel)

        #-------experiment
        for k,v in self.freq_dev_tab.items():
            if not(k in self.freq_dev_phone.keys()) or v > self.freq_dev_phone[k]:
                self.freq_dev_tab[k] = 1
                self.freq_dev_phone[k] = 0
            else:
                self.freq_dev_tab[k] = 1
                self.freq_dev_phone[k] = 0
        for k,v in self.freq_dev_phone.items():
            if not(k in self.freq_dev_tab.keys()):
                self.freq_dev_tab[k] = 0
                self.freq_dev_phone[k] = 1
        #----------

        if self.devCount != 0:
            self.views_per_week /= self.devCount
            self.views /= self.devCount
            self.avg_dur /= self.devCount
            self.sessions_per_day /= self.devCount
            self.views_per_session /= self.devCount
            self.avg_len /= self.devCount
            self.avg_dur /= self.devCount
            self.avg_percent /= self.devCount
            self.fav_num /= self.devCount
            self.add_num /= self.devCount
            self.del_num /= self.devCount
            self.distinct_fav_shows /= self.devCount
            self.favs_per_day /= self.devCount
            self.session_duration /= self.devCount
            self.session_count /= self.devCount
            self.time_between_sess /= self.devCount

    def to_dict(self):
        self.count_stats()
        res = {
            u"views_per_week": self.views_per_week,
            u"avg_dur": self.avg_dur,
            u"sessions_per_day": self.sessions_per_day,
            u"views_per_session": self.views_per_session,
            u"avg_len": self.avg_len,
            u"avg_percent": self.avg_percent,
            u"rewatch": self.rewatch,
            u"number_of_ch": self.chann_numb,
            u"uid": self.uid,
            #u"os": self.os,
            #u"kind": self.kind,

            u"views_per_week": self.views_per_week,
            u"avg_dur": self.avg_dur,
            u"sessions_per_day": self.sessions_per_day,
            u"views_per_session": self.views_per_session,
            u"avg_len": self.avg_len,
            u"avg_percent": self.avg_percent,
            u"rewatch": self.rewatch
            }
        '''
            u"fav_num" : self.fav_num,
            u"add_num" : self.add_num,
            u"del_num" : self.del_num,
            u"distinct_fav_shows" : self.distinct_fav_shows,
            u"favs_per_day" : self.favs_per_day,
            u"session_duration" : self.session_duration,
            u"session_count" : self.session_count,
            u"time_between_sess" : self.time_between_sess
        }
        '''
        #maybe assign -1 if it is a user?
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
        '''
        for k, v in self.fav_days.items():
            res[u"fav_d:%s" % k] = v
        '''
        #----experiment
        for k, v in self.freq_dev_tab.items():
            res[u"tab:%s" % k] = v
        for k, v in self.freq_dev_phone.items():
            res[u"ph:%s" % k] = v
        #----------------

        return res

# ## Preparing data
pipeline = [
    {"$match": {
        "$or": [{"VODV": {"$gt": 0}}, {"ItemselectedCount": {"$gt": 0}}],
    }},
    {"$unwind": "$LoginUserIDs"},
    {"$match": {"LoginUserIDs": {"$in": list(proper_user_ids_two.keys())}}}
]

j = 0

import pprint

user_ids = {}
users_and_objects = {}
users_dict = {}
videos = {}

# ### Training and testing

testing = []
training = []
import numpy
import pprint
dev_cnt = 0

for data in cursor.withLogin.aggregate(pipeline):
    iid = data["LoginUserIDs"][0]
    if iid in proper_user_ids.keys():
        if not(iid in users_dict):
            uid = user_ids.setdefault(data['LoginUserIDs'][0], len(user_ids))
            usr = User(uid)
            users_dict.setdefault(iid, usr)
        else:
            uid = users_dict[iid].uid
        obj = Device.create(uid, data)
        dev_cnt += 1
        #if iid in proper_user_ids.keys():
        if np.random.rand() >= 0.1:
            users_dict[iid].add_device(obj)
        else:
            testing.append(obj.to_dict())
        #else:
        #    users_dict[iid].add_device(obj)
    j += 1

print("!!!!!!!!!!!!!!!!!!" + str(dev_cnt))

print(len(users_dict))




users = []
Y = []
X = []
probability = 0.5

for uid, usr in users_dict.items():
    #do I copy or create a link here?
    #objs = usr.devices
    #if np.random.rand() >= 0.5:
    #    test_obj = objs.pop(np.random.randint(len(objs)))
    #    testing.append(test_obj.to_dict())
    #the error is here

    #for obj in objs:
    #    training.append(obj.to_dict())
    if usr.devCount != 0:
        training.append(usr.to_dict())

print(len(training), len(testing))


import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.DataFrame.from_dict(training)
#print(train_df.shape)
#print(train_df.head())

test_df = pd.DataFrame.from_dict(testing)
#print test_df.shape
#print(test_df.head())

common_columns = list(set(train_df.columns).intersection(test_df.columns))
train_df = train_df[common_columns]
test_df = test_df[common_columns]

#print(common_columns)

# prepare training data
train_df.loc[train_df["os"] == u'Android', "os"] = 0
train_df.loc[train_df["os"] == u'iPhone OS', "os"] = 1

train_df.loc[train_df["kind"] == u'undefined', "kind"] = 0
train_df.loc[train_df["kind"] == u'Phone', "kind"] = 1
train_df.loc[train_df["kind"] == u'Tablet', "kind"] = 2

train_df = train_df.fillna(0)
#print(train_df.dtypes)

# prepare testing data

test_df.loc[test_df["os"] == u'Android', "os"] = 0
test_df[test_df["os"] == u'Android'] = 0
test_df.loc[test_df["os"] == u'iPhone OS', "os"] = 1

test_df.loc[test_df["kind"] == u'undefined', "kind"] = 0
test_df.loc[test_df["kind"] == u'Phone', "kind"] = 1
test_df.loc[test_df["kind"] == u'Tablet', "kind"] = 2
test_df = test_df.fillna(0)

print(train_df.head())
print(test_df.head())

from sklearn.feature_selection import SelectKBest, f_classif, chi2

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
    #u'fv:kabel1',
    u'uid',
    #u'os'
]

#predictors = train_df.columns.tolist()
predictors = good_features
prediction = "uid"
#predictors.remove(prediction)
'''
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
'''
print(len(proper_user_ids))
print(len(train_df))
print(len(test_df))
train_df.to_csv("train_ds_exp_three_clean", sep='\t', encoding='utf-8')
test_df.to_csv("test_ds_exp_three_clean", sep='\t', encoding='utf-8')
exit(0)
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

print(train_df[predictors].shape)
lsvc = LinearSVC(C=0.28, penalty="l1", dual=False).fit(train_df[predictors], train_df[prediction])
model = SelectFromModel(lsvc, prefit=True)
train_new = model.transform(train_df[predictors])
print(train_new.shape)

test_new = model.transform(test_df[predictors])

# ## Applying classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# good_features = list(train_df.columns.values)
# good_features.remove('os')
# good_features.remove('uid')

train_X = train_df[predictors][:]
train_y = train_df[prediction]

test_X = test_df[predictors][:]
test_y = test_df[prediction]

from sklearn import metrics

#clf = RandomForestClassifier(n_estimators=200, random_state=1)
clf = DecisionTreeClassifier(random_state=1)
clf.fit(train_X, train_y)

model = SelectFromModel(clf, prefit=True)
#X_new = model.transform(train_df[predictors])
#print(X_new.shape)
predicted_y = clf.predict(test_X)
print(metrics.precision_recall_fscore_support(test_y, predicted_y, average='weighted'))
print(predicted_y[test_y.values == predicted_y].size / float(len(predicted_y)))

predicted_y = clf.predict(test_X)
print("---------------------------AFTER SELECTION-----------------------")
train_X = train_new
test_X = test_new

clf.fit(train_X, train_y)
predicted_y = clf.predict(test_X)

print(metrics.precision_recall_fscore_support(test_y, predicted_y, average='weighted'))
print(predicted_y[test_y.values == predicted_y].size / float(len(predicted_y)))