from __future__ import division
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from  operator import add

from datetime import datetime, timedelta

APP_NAME = 'ExtractorMongoDB'


def parse(row):
    serialNumber = str(row.value.serialNumber)
    timestamp = row.time
    if row.value.zone is not None:
        zone = str(row.value.zone.value.name)
    else:
        zone = None
    if row.value.sGroup is not None:
        group = str(row.value.sGroup.value)
    else:
        group = None  # -> No group???
    if row.value.status is not None:
        if str(row.value.status.value) == 'Sold':
            status = int(1)
        else:
            status = int(0)
    else:
        status = int(0)
        # if row.value.itemcode is not None:
        #   itemcode=str(row.itemcode.value)
        # else:
        #   itemcode=None
    return [serialNumber, timestamp, zone, group, status]


def summarize_serial(x, y):
    # Assign to first_date the first blink row:
    last_date = max(x[0], y[0])
    if last_date == x[0]:
        first_line = y
        last_line = x
    elif last_date == y[0]:
        first_line = x
        last_line = y
    status = last_line[3]
    zone = first_line[1]
    group = last_line[2]
    # brand = last_line[4]
    # dep = last_line[5]
    # blinks_number = first_line[6] + last_line[6]
    # result = [first_date, last_line[0], zone, group, brand, dep, blinks_number, status]
    result = [last_date, zone, group, status]
    return result


def extracting_time(x):
    year = x[0].year
    month = x[0].month
    WM = (x[0].day - 1) // 7 + 1
    DM = x[0].day
    DW = x[0].weekday()
    Zone = x[1]
    Sold = x[2]
    return [year, month, WM, DM, DW, Zone, Sold]


def main(sqlContext):
    #
    sqlContext.sql(
        "CREATE TEMPORARY TABLE thingsTable USING com.stratio.datasource.mongodb OPTIONS (host 'localhost:27017', "
        "database 'riot_main', collection 'thingSnapshotsSH', splitKey 'time', splitKeyType 'isoDate', "
        "splitKeyMin '2016-01-01T00:00:00.000Z', splitKeyMax '2016-01-03T00:00:00.000Z' )")

    # Extract all blinks from the Database:
    raw_data = sqlContext.sql(
        "SELECT * FROM thingsTable WHERE value.groupId=3 AND value.status.value='Sold'")  # WHERE value.status.value = 'Sold'")
    print ('Number of blinks: ' + str(raw_data.count()))
    #print ('Document Example:    ' + str(raw_data.first()))

    # Parse the rows and extract as (serial, initial_date, end_date, zone, group, status(binary))

    SerialDateZoneGroupStatusRdd = raw_data.map(parse).map(
        lambda x: (x[0], (x[1], x[2], x[3], x[4]))).reduceByKey(summarize_serial)

    print ('Number of Things Sold: ' + str(SerialDateZoneGroupStatusRdd.count()))
    print ('Thing Sold Example :    ' + str(SerialDateZoneGroupStatusRdd.first()))

    # Get the number of objects sold per Zone per day:

    DateZoneStatusRdd = SerialDateZoneGroupStatusRdd.map(
        lambda x: ((x[1][0].date(), x[1][1]), x[1][3])).reduceByKey(add).map(lambda x: (x[0][0], x[0][1], x[1]))

    print ('Number of Zones Sold :' + str(DateZoneStatusRdd.count()))
    print ('Zone Example :    ' + str(DateZoneStatusRdd.first()))

    # Get the number of objects sold per Zone per Group per day:

    DateZoneGroupStatusRdd = SerialDateZoneGroupStatusRdd.reduceByKey(summarize_serial).map(
        lambda x: ((x[1][0].date(), x[1][1], x[1][2]), x[1][3])).reduceByKey(add).map(
        lambda x: (x[0][0], x[0][1], x[0][2],
                   x[1]))
    print ('Number of things Sold per group per zone: ' + str(DateZoneGroupStatusRdd.count()))
    print ('Zone Group Example :    ' + str(DateZoneGroupStatusRdd.first()))

    # Extracting Features to RDDs:
    ZoneData = DateZoneStatusRdd.map(extracting_time)
    print ('Number of registers : ' + str(ZoneData.count()))
    print ('ZoneData Example :    ' + str(ZoneData.first()))

    #ZoneData.coalesce(1, True).saveAsPickleFile('/tmp/ZoneData_1D')

    #ZoneData.coalesce(1, True).saveAsTextFile('/tmp/ZoneDataText_2d_v2')


if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setMaster("local[*]")
    conf = conf.setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')
    # Execute Main functionality
    sqlContext = SQLContext(sc)
    main(sqlContext)
    sc.stop()
