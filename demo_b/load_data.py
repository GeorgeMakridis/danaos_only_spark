from pyspark.sql.functions import col, when, rand, lit
import gc

from demo_b.Configs import schema_vds, schema_mes

useless_cols = ['vessel_code', 'foTotVolume', 'scavAirInLetPress', 'scavengeAirPressure', 'scavAirFireDetTempNo11',
                'scavAirFireDetTempNo12', 'coolerCWinTemp', 'cylLoTemp', 'hfoViscocityHighLow', 'hpsBearingTemp',
                'cylExhGasOutTempNo11',
                'cylExhGasOutTempNo12', 'cylJCFWOutTempNo11', 'cylJCFWOutTempNo12', 'cylPistonCOOutTempNo11',
                'cylPistonCOOutTempNo12', 'stw', 'stw_long', 'stw_trans',
                'tcExhGasInTempNo4', 'tcExhGasOutTempNo4', 'tcLOInLETPressNo4', 'tcLOOutLETTempNo4',
                'tcRPMNo4', 'orderRPMBridgeLeverer', 'scavengeAirPressure', 'coolingWOutLETTempNo4',
                'foConsumption', 'foID']


def load_vessel_data_of_specific_vessel(vessel_id=None, db_table="vds", sc=None):
    train_vds = sc.read.format("minioSelectParquet").schema(schema_vds).load(
        "s3://danaos/" + str(vessel_id) + db_table + ".parquet")
    train_vds = train_vds.dropDuplicates(['datetime'])
    train_vds = train_vds[train_vds['datetime'] > '2016-1-31 23:00:00']
    train_vds = train_vds[train_vds['datetime'] < '2018-3-31 23:00:00']
    return train_vds


def load_main_engine_data_of_specific_vessel(vessel_id=None, sc=None, db_table='mes'):
    train = sc.read.format("minioSelectParquet").schema(schema_mes).load(
        "s3://danaos/" + str(vessel_id) + db_table + ".parquet")
    train = train.dropDuplicates(['datetime'])
    train = train[train['datetime'] > '2016-1-31 23:00:00']
    train = train[train['datetime'] < '2018-3-31 23:00:00']
    return train


def load_train_data(model=None, spark=None):
    """

    :param model:
    :param spark:
    :return:
    """
    if model == 'xgb':
        train_vds = load_vessel_data_of_specific_vessel(7, sc=spark)
        train = load_main_engine_data_of_specific_vessel(7, sc=spark)
        train_vds_2 = load_vessel_data_of_specific_vessel(8, sc=spark)
        train_2 = load_main_engine_data_of_specific_vessel(8, sc=spark)
        train_vds_3 = load_vessel_data_of_specific_vessel(6, sc=spark)
        train_3 = load_main_engine_data_of_specific_vessel(6, sc=spark)

        train_vds = train_vds.drop(*['rpm', 'power'])
        train_vds_2 = train_vds_2.drop(*['rpm', 'power'])
        train_vds_3 = train_vds_3.drop(*['rpm', 'power'])

        # print('length :' + str(len(train.columns)))
        train = train.join(train_vds, ['datetime'])
        train_2 = train_2.join(train_vds_2, ['datetime'])
        train_3 = train_3.join(train_vds_3, ['datetime'])
        train = train.union(train_2)
        train = train.union(train_3)
        del train_2, train_vds, train_vds_2, train_3, train_vds_3
        gc.collect()

        train = train.drop(*useless_cols)
        return train


def load_test_data(spark):
    """

    :param spark:
    :return:
    """

    test_vds = load_vessel_data_of_specific_vessel(5, sc=spark)
    test = load_main_engine_data_of_specific_vessel(5, sc=spark)

    test_vds = test_vds.drop(*['rpm', 'power'])
    test = test.join(test_vds, ['datetime'])
    del test_vds
    gc.collect()

    test = test.drop(*useless_cols)
    return test
