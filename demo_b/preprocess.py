

def preprocess_train(train, model=None, spark=None):
    """
    The preprocess function for the training
    :param train: data in pyspark.Dataframe
    :param model: string, the name of the used model
    :param spark: saprk context
    :return: the preprocessed Dataframe
    """
    train = train.fillna(0)
    return train


def preprocess_test(test, model=None):
    """
    The preprocess function for the predic
    :param train: data in pyspark.Dataframe
    :param model: string, the name of the used model
    :return: the preprocessed Dataframe
    """
    test = test.fillna(0)
    return test
