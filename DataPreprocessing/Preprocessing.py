import warnings
warnings.filterwarnings('ignore')
from Logger.Logging import LogClass
from datetime import datetime
from DataValidation.DataValidation import Validation
import os



class Preprocessing:
    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Preprocessing.txt'
        self.validation = Validation()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj = LogClass(self.folder, self.filename)

    def remove_training_duplicates(self):

        """
                    Method: remove_training_duplicates
                    Description: This method is used to remove duplicates  values if present in the dataset
                    Parameters: None
                    Return: DataFrame after removing the duplicate values


        """

        try:
            self.log_obj.log("INFO", "Removing Duplicates from Dataframe")
            TrainData = self.validation.training_finding_null()

            TrainData.drop_duplicates(keep='first', inplace=True)
            self.log_obj.log("INFO", "Duplicate values have been dropped")
            return TrainData

        except Exception as e:
            self.log_obj.log("ERROR", "Exception is:" + '\t' + str(e))

    def remove_testing_duplicates(self):

        """
                    Method: remove_testing_duplicates
                    Description: This method is used to remove duplicates  values if present in the dataset
                    Parameters: None
                    Return: DataFrame after removing the duplicate values


        """

        try:
            self.log_obj.log("INFO", "Removing Duplicates from Dataframe")
            Testdata = self.validation.testing_finding_null()
            Testdata.drop_duplicates(keep='first', inplace=True)
            self.log_obj.log("INFO", "Duplicate values have been dropped")
            return Testdata

        except Exception as e:
            self.log_obj.log("ERROR", "Exception is:" + '\t' + str(e))


    def trainig_column_cleaner(self):
        """
                            Method: training_column_cleaner
                            Description: This method is used to remove clean the respective column.
                            Parameters: None
                            Return: DataFrame after cleaning the column


        """

        try:
            self.log_obj.log('INFOR', 'Cleaning the Item_Fat_Content column')
            TrainData = self.remove_training_duplicates()
            TrainData['Item_Fat_Content'] = TrainData['Item_Fat_Content'].str.replace('LF', 'Low Fat')
            TrainData['Item_Fat_Content'] = TrainData['Item_Fat_Content'].str.replace('low fat', 'Low Fat')
            TrainData['Item_Fat_Content'] = TrainData['Item_Fat_Content'].str.replace('reg', 'Regular')

            TrainData['Age_Outlet'] = datetime.now().year - TrainData['Outlet_Establishment_Year']
            TrainData.drop(columns=['Outlet_Identifier', 'Item_Identifier','Outlet_Establishment_Year'], inplace=True)

        except Exception as e:
            self.log_obj.log('Error', 'Exception is'+ '\t' + str(e))

        else:
            return TrainData


    def testing_column_cleaner(self):
        """
                            Method: testing_column_cleaner
                            Description: This method is used to remove clean the respective column.
                            Parameters: None
                            Return: DataFrame after cleaning the column


        """

        try:
            self.log_obj.log('INFOR', 'Cleaning the Item_Fat_Content column')
            TestData = self.remove_testing_duplicates()
            TestData['Item_Fat_Content'] = TestData['Item_Fat_Content'].str.replace('LF', 'Low Fat')
            TestData['Item_Fat_Content'] = TestData['Item_Fat_Content'].str.replace('low fat', 'Low Fat')
            TestData['Item_Fat_Content'] = TestData['Item_Fat_Content'].str.replace('reg', 'Regular')

            TestData['Age_Outlet'] = datetime.now().year - TestData['Outlet_Establishment_Year']
            TestData.drop(columns=['Outlet_Identifier', 'Item_Identifier','Outlet_Establishment_Year'], inplace=True)

        except Exception as e:
            self.log_obj.log('Error', 'Exception is'+ '\t' + str(e))

        else:
            return TestData





