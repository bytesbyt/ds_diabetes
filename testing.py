import unittest
import numpy as np
import pickle
from diabetes_model import model
from diabetes_model import scaler

class TestDiabetesPrediction(unittest.TestCase):
    def test_diabetes_prediction(self):
        # Test case 1
        input_data_1 = np.array([1,85,66,29,0,26.6,0.351,31])
        input_data_1 = input_data_1.reshape(1, -1)

        # Apply scaling transformation
        scaled_data_1 = scaler.transform(input_data_1)

        # Make prediction
        result_1 = model.predict(scaled_data_1)
        print(f"Test case 1 result is {result_1[0]}") 


         # test validation
        self.assertEqual(result_1[0], 0)

        # Test case 2
        input_data_2 = np.array([6,148,72,35,0,33.6,0.627,50])
        input_data_2 = input_data_2.reshape(1, -1)

        # Apply scaling transformation
        scaled_data_2 = scaler.transform(input_data_2)

        # Make prediction
        result_2 = model.predict(scaled_data_2)
        print(f"Test case 2 result is {result_2[0]}")

            # test validation
        self.assertEqual(result_2[0], 1)

if __name__ == '__main__':
    unittest.main()
