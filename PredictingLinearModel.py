# Predictions for the Linear Regression Model:

from __future__ import print_function
from __future__ import division
from datetime import datetime, timedelta, date
from pyspark import SparkConf, SparkContext

from pyspark.mllib.regression import LabeledPoint, LinearRegressionModel

import numpy as np
import sys

# Module Constants
event_date = '%d/%m/%y'
APP_NAME = 'sharaf_predicting_linear'


# FUNCTIONS for Predicting

def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()


def extract_features(record, cat_len, mappings):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[0:len(mappings)]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i += 1
        step += len(m)
    return cat_vec


def extract_label(record):
    return float(record[-1])


def format_date(x, zone, group, mappings, cat_len):
    return extract_features((x.year, x.month, (x.day - 1) // 7 + 1, x.day, x.weekday(), zone, group), cat_len, mappings)


def format_show(x, zone, group):
    return x.year, x.month, (x.day - 1) // 7 + 1, x.day, x.weekday(), zone, group


def get_group_zone(to_test, group_or_zone, features_cr):
    if (to_test == 'all') & (group_or_zone == 'zone'):
        return np.unique(features_cr.map(lambda x: x[5]).collect())
    elif (to_test == 'all') & (group_or_zone == 'group'):
        return np.unique(features_cr.map(lambda x: x[6]).collect())
    elif (to_test != 'all'):
        return [to_test]


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days + 1)):
        yield start_date + timedelta(n)


# Functions for calculating the errors:
def squared_error(actual, pred):
    return (pred - actual) ** 2


def abs_error(actual, pred):
    return np.abs(pred - actual)


def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1)) ** 2


def main(sc):
    features_cr = sc.pickleFile('/tmp/features_saved')
    linear_model = LinearRegressionModel.load(sc, "/tmp/linear_model")
    # Getting the features ready for predicting
    numberFeatures = len(features_cr.first()) - 1
    mappings = [get_mapping(features_cr, i) for i in range(0, numberFeatures)]

    # Month Dictionary
    dictio_month = {}
    for i in range(12):
        dictio_month[i + 1] = i
    mappings[1] = dictio_month

    cat_len = sum(map(len, mappings))




    if len(sys.argv) == 2:
        dateZoneGroup = str(sys.argv[1]).split(',')
        zones = get_group_zone(dateZoneGroup[5], 'zone', features_cr)
        groups = get_group_zone(dateZoneGroup[6], 'group', features_cr)
        year_feat = int(dateZoneGroup[0])
        month_feat = int(dateZoneGroup[1])
        day_feat = int(dateZoneGroup[3])
        startDate = date(year_feat, month_feat, day_feat)
        endDate = date(year_feat, month_feat, day_feat)
        print (startDate)
    else:
        startDate = datetime.strptime(str(sys.argv[1]), event_date).date()
        endDate = datetime.strptime(str(sys.argv[2]), event_date).date()
        zones = get_group_zone(str(sys.argv[3]), 'zone', features_cr)
        groups = get_group_zone(str(sys.argv[4]), 'group', features_cr)

    dateRangeWF = list(date_range(startDate, endDate))

    # Making predictions:

    feat_vs_pred = list()
    featPredShow = list()
    for g in groups:
        for z in zones:
            for dR in dateRangeWF:
                featureLine = format_date(dR, z, g, mappings, cat_len)
                #print(featureLine)
                featureShow = format_show(dR, z, g)
                predLine = linear_model.predict(featureLine)
                feat_vs_pred.append(list(featureLine) + [predLine])
                featPredShow.append(list(featureShow) + [predLine])

    scFeaturesPred = sc.parallelize(featPredShow)
    [print(j) for j in scFeaturesPred.collect()]
    # In order to Export:
    # scFeaturesPred.coalesce(1, True).saveAsTextFile('/tmp/results_linear')


if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setMaster("local[*]")
    conf = conf.setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')
    # Execute Main functionality
    main(sc)
    sc.stop()
