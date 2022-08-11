import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Logger.Logging import LogClass
import os


class GetData:

    def __init__(self):
        self.training_file = '../DataSet/StoreSalesPredictionTrain.csv'
        self.testing_file = '../DataSet/StoreSalesPredictionTest.csv'
        self.folder = '../LogFiles/'
        self.file_name = "Loading_Data.txt"

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj = LogClass(self.folder, self.file_name)

    def load_trainig_data(self):

        """
                    Method: load_data
                    Description: This method is used to load the dataset
                    Parameters: None
                    Return: Gave the DataFrame of a data.

        """

        try:
            self.log_obj.log("INFO", "loading the dataset into pandas dataframe")
            self.dataframe = pd.read_csv(self.training_file)
            self.log_obj.log("INFO", 'Raw Train data got loaded in dataframe')

            return self.dataframe

        except Exception as e:
            self.log_obj.log("ERROR", "Failed while loading dataset.Error is:" + str(e))


    def load_testing_data(self):

        """
                    Method: load_data
                    Description: This method is used to load the test dataset
                    Parameters: None
                    Return: Gave the DataFrame of a data.

        """

        try:
            self.log_obj.log("INFO", "loading the dataset into pandas dataframe")
            self.dataframe = pd.read_csv(self.testing_file)
            self.log_obj.log("INFO", 'Test data got loaded in dataframe')

            return self.dataframe

        except Exception as e:
            self.log_obj.log("ERROR", "Failed while loading dataset.Error is:" + str(e))

