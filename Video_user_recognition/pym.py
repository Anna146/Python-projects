__author__ = 'Asus'

import pymongo
mongo = pymongo.MongoClient()

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
cursor = mongo.get_database('7tv')
print(cursor.collection_names())
print(cursor.withLogin.find().count())

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

