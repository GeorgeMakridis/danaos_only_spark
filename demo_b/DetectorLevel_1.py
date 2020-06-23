import demo_b.preprocess as preprocess
import demo_b.load_data as load_data
from demo_b.DetectorWithRegression import DetectorWithRegression


class DetectorLevel_1(object):

    def __init__(self, param, sc):
        self.param = param
        self.test_ini = None
        self.train_ini = None
        self.sc = sc

    def run_xgb_prediction(self, message):
        """
         Handles the training process.
        
        :param message: 
        :return: 
        """
        print(message)
        if self.test_ini is None:
            self.test_ini = load_data.load_test_data(self.sc)
        test = preprocess.preprocess_test(self.test_ini, 'xgb')
        xgb_detector = DetectorWithRegression(test, self.sc)
        xgb_detector.predict()

    def run_xgb_training(self, message):
        """
        Handles the prediction process. 
        
        :param message: 
        :return: 
        """
        print(message)
        if self.train_ini is None:
            self.train_ini = load_data.load_train_data('xgb', spark=self.sc)
        train = preprocess.preprocess_train(self.train_ini, 'xgb')
        xgb_detector = DetectorWithRegression(train, self.sc)
        xgb_detector.fit()
