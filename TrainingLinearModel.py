# Linear Regression Model:

from __future__ import division
from datetime import datetime, timedelta, date
from pyspark import SparkConf, SparkContext

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
import numpy as np

# Module Constants
datetime_event = '%m/%d/%y %H:%M'
datetime_event2 = '%m/%d/%y %H:%M:%S'
APP_NAME = 'sharaf_training_linear'


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
        i = i + 1
        step = step + len(m)
    return cat_vec


def extract_label(record):
    return float(record[-1])


def main(sc):
    # Loading the features:
    features_cr = sc.pickleFile('/tmp/features_saved')
    print (features_cr.first())

    # Getting the features ready for training
    numberFeatures = len(features_cr.first()) - 1
    mappings = [get_mapping(features_cr, i) for i in range(0, numberFeatures)]

    # Working with the Mapping:
    # Month:
    dictio_month = {}
    for i in range(12):
        dictio_month[i + 1] = i
    mappings[1] = dictio_month
    # Year: ?

    cat_len = sum(map(len, mappings))
    data = features_cr.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, cat_len, mappings)))
    print (features_cr.first())
    # Regression:
    linear_model = LinearRegressionWithSGD.train(data, iterations=100, step=0.25, intercept=False)



    linear_model.save(sc, '/tmp/linear_model')
    print 'OK model'


if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setMaster("local[*]")
    conf = conf.setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')
    # Execute Main functionality
    main(sc)
    sc.stop()
