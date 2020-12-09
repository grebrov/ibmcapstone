#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from modelcapstone import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-0_1.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        all_data, all_models = model_load(test=True)
        model = all_models['united_kingdom']
        
        self.assertTrue('predict' in dir(model))
    def test_03_predict(self):
        """
        test the predict function input
        """
   
        ## specify test values
        # model_predict(country,year,month,day,all_models=None,test=False):
        result = model_predict('united_kingdom','2018','01','05',all_models=False, test=True)
        y_pred = result['y_pred']
        print("prediciton; ", y_pred)
        self.assertTrue(y_pred >= 0.0)
           
### Run the tests
if __name__ == '__main__':
    unittest.main()
