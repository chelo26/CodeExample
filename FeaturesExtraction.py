# Decision Tree Model:

from __future__ import division
from datetime import datetime, timedelta, date
from pyspark import SparkConf, SparkContext

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
import os, tempfile
import numpy as np
import sys
import pandas as pd

# Module Constants
datetime_event = '%m/%d/%y %H:%M'
datetime_event2 = '%m/%d/%y %H:%M:%S'
APP_NAME = 'sharaf_features_extraction'


# Functions for Extracting features

# Parsing Function: Gets Datetime, status, group,zone, brand, department

def get_data(path_data):
    return sc.textFile(path_data)


def take_out_header(data):
    header =data.first()
    return data.filter(lambda x: x!=header)


def parse(row):
    row_converted = row
    row_converted[0] = str(row[0])
    if len(row[1]) == 14:  # --->> Difference between H:M:S and H:M
        row_converted[1] = datetime.strptime(row[1], datetime_event)
    else:
        row_converted[1] = datetime.strptime(row[1], datetime_event2)
    if row[4] == 'Sold':
        row_converted[2] = 1
    else:
        row_converted[2] = 0
    row_converted[3] = str(row[5])
    row_converted[4] = str(row[7])
    row_converted[5] = str(row[11])
    row_converted[6] = str(row[13])
    row_converted[7] = 1
    return row_converted[:8]


# Getting the first and last date, the first zone, the last group, brand, dep, status
def summarize_serial(x, y):
    first_date = min(x[0], y[0])
    if first_date == x[0]:
        first_line = x
        last_line = y
    elif first_date == y[0]:
        first_line = y
        last_line = x
    status = last_line[6]
    zone = first_line[2]
    group = last_line[3]
    brand = last_line[4]
    dep = last_line[5]
    blinks_number = first_line[6] + last_line[6]
    result = [first_date, last_line[0], zone, group, brand, dep, blinks_number, status]
    return result


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days + 1)):
        yield start_date + timedelta(n)


def explode(tag_row):
    start_date = tag_row[1][0].date()
    end_date = tag_row[1][1].date()
    total_dates = list(daterange(start_date, end_date))
    result = list()
    for step in total_dates:
        row = []
        row.append(tag_row[0])
        row.append(step)
        row.append(tag_row[1][2])
        row.append(tag_row[1][3])
        row.append(tag_row[1][7])
        result.append(row)
    return result


def main(sc):
    # Main functionality
    # Extracting the features:

    path1 = "/Users/Chelo/Documents/Work/Consulting/Mojix/data_sharaf/sharaf_clean_1_aug_nov_v2.csv"
    path2 = "/Users/Chelo/Documents/Work/Consulting/Mojix/data_sharaf/sharaf_clean_2_nov_jan_v2 2.csv"

    # Importing the CSVs as a SC
    raw_rdd1 = get_data(path1)
    raw_rdd2 = get_data(path2)
    # Taking out header:
    # Taking out header:
    rdd1WH = take_out_header(raw_rdd1)
    rdd2WH = take_out_header(raw_rdd2)
    # We split and parse
    raw_data = rdd1WH.union(rdd2WH)
    rd = raw_data.map(lambda x: x.split(','))
    rd_split = rd.map(parse)
    fields = raw_rdd1.first().split(',')

    # Others:
    dict_fields = {}
    for i in range(len(fields)):
        dict_fields[i] = str(fields[i])

    # If we want to get everything by tag:

    rd_general = rd_split.map(lambda x: (x[0], (x[1], x[1], x[3], x[4], x[5], x[6], x[7], x[2])))
    rd_by_tag = rd_general.reduceByKey(summarize_serial)

    # Creating dictionary "Thing: group"
    dictio_tg = rd_by_tag.map(lambda x: (x[0], x[1][3])).collectAsMap()

    # Counting number of things present per zone
    flat_tag = rd_by_tag.flatMap(explode)
    zoneGroup = flat_tag.map(lambda x: ((x[1], x[2], x[3]), 1)).reduceByKey(lambda x, y: x + y)

    # Counting the number of things sold by group by zone
    rd_sold = rd_by_tag.filter(lambda x: x[1][-1] == 1)
    tagZoneGroup_sold = rd_sold.map(
        lambda line: ((line[0], line[1][1].date(), line[1][2], dictio_tg[line[0]]), line[1][7])).reduceByKey(
        lambda x, y: x + y)
    zoneGroup_sold = tagZoneGroup_sold.map(lambda x: ((x[0][1], x[0][2], x[0][3]), 1)).reduceByKey(lambda x, y: x + y)

    # Gettting the Conversion Rate per zone per group
    jointZoneGroup = zoneGroup_sold.join(zoneGroup)
    daily_cr = jointZoneGroup.map(lambda x: ((x[0][0], x[0][1], x[0][2]), float(x[1][0] / x[1][1])))

    features_cr = daily_cr.map(
        lambda x: (x[0][0].year,x[0][0].month, (x[0][0].day - 1) // 7 + 1, x[0][0].day, x[0][0].weekday(), x[0][1], x[0][2], x[1]))


    features_cr.coalesce(1, True).saveAsPickleFile('/tmp/features_saved')
    print 'OK features'

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setMaster("local[*]")
    conf = conf.setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')
    # Execute Main functionality
    main(sc)
    sc.stop()
