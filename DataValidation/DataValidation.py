import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Logger.Logging import LogClass
from DataLoader.DataGetter import GetData
from DataFeatureClassification.FeatureClassification import Features
import os

class Validation:

    def __init__(self):
        self.folder='../LogFiles/'
        self.filename='Raw_Data_Validation.txt'
        self.df_object = GetData()
        self.features = Features()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj = LogClass(self.folder,self.filename)

    def training_data_stdzero(self):

        """
                    Method: training_data_stdzero
                    Description: This method is used to check if standard deviation is Zero in the training dataset
                    Parameters: None
                    Return: DataFrame after removing zero standard deviation columns

        """

        try:
            self.log_obj.log("INFO","Checking if the standard deviation is zero or not for numerical features ")
            dataframe = self.df_object.load_trainig_data()
            numerical_features = self.features.training_numerical_features()

            for feature in numerical_features:
                if dataframe[feature].std()== 0:
                    self.log_obj.log("INFO", "If there is a Zero std deviation columns are present then remove those columns")
                    dataframe.drop(columns=feature,axis=1,inplace=True)
            self.log_obj.log("INFO","Zero std deviation columns are removed")
            return dataframe

        except Exception as e:
            self.log_obj.log("ERROR","Occured Exception is :"+ "\t" + str(e))

    def testing_data_stdzero(self):

        """
                    Method: testing_data_stdzero
                    Description: This method is used to check if standard deviation is Zero in the training dataset
                    Parameters: None
                    Return: DataFrame after removing zero standard deviation columns

        """

        try:
            self.log_obj.log("INFO","Checking if the standard deviation is zero or not for numerical features ")
            dataframe = self.df_object.load_testing_data()
            numerical_features = self.features.testing_numerical_features()

            for feature in numerical_features:
                if dataframe[feature].std()== 0:
                    self.log_obj.log("INFO", "If there is a Zero std deviation columns are present then remove those columns")
                    dataframe.drop(columns=feature,axis=1,inplace=True)
            self.log_obj.log("INFO","Zero std deviation columns are removed")
            return dataframe

        except Exception as e:
            self.log_obj.log("ERROR","Occured Exception is :"+ "\t" + str(e))


    def traing_data_whole_missing(self):

        """
                    Method: traning_data_whole_missing
                    Description: This method is used to check if entire column has missing values
                    Parameters: None
                    Return: DataFrame after removing columns with entire missing values


        """

        try:
            self.log_obj.log("INFO","checking if there is column with whole missing values or not")
            data=self.training_data_stdzero()

            for i in data.columns:
                if data[i].isnull().sum()==len(data[i]):
                    self.log_obj.log("INFO", "If a column having whole missing value then drop it.")
                    data.drop(columns=i,axis=1,inplace=True)
                else:
                    pass

            self.log_obj.log("INFO","Whole missing value row is dropped")

            return data

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))

    def testing_data_whole_missing(self):

        """
                    Method: testing_data_whole_missing
                    Description: This method is used to check if entire column has missing values
                    Parameters: None
                    Return: DataFrame after removing columns with entire missing values


        """

        try:
            self.log_obj.log("INFO","checking if there is column with whole missing values or not")
            data=self.testing_data_stdzero()

            for i in data.columns:
                if data[i].isnull().sum()==len(data[i]):
                    self.log_obj.log("INFO", "If a column having whole missing value then drop it.")
                    data.drop(columns=i,axis=1,inplace=True)
                else:
                    pass

            self.log_obj.log("INFO","Whole missing value row is dropped")

            return data

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))


    def training_finding_null(self):
        """
                    Method: training_finding_null
                    Description: This method is used to check if there are any null values and replacing them with nan
                    Parameters: None
                    Return: DataFrame after checking null values

         """


        try:
            self.log_obj.log("INFO","Replacing null type values with NAN value")
            dataFrame = self.traing_data_whole_missing()
            categorical = self.features.training_categorical_features()
            numerical = self.features.training_numerical_features()

            for feature in numerical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)
            self.log_obj.log("INFO", "Replace null type values with NAN in numerical column")


            for feature in categorical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)
            self.log_obj.log("INFO", "Replace null type values with NAN in categorical column")


            self.log_obj.log("INFO","Replaced all null type values with NAN values")
            return dataFrame

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))


    def testing_finding_null(self):
        """
                    Method: testing_finding_null
                    Description: This method is used to check if there are any null values and replacing them with nan
                    Parameters: None
                    Return: DataFrame after checking null values

         """


        try:
            self.log_obj.log("INFO","Replacing null type values with NAN value")
            dataFrame = self.testing_data_whole_missing()
            categorical = self.features.testing_categorical_features()
            numerical = self.features.testing_categorical_features()

            for feature in numerical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)
            self.log_obj.log("INFO", "Replace null type values with NAN in numerical column")


            for feature in categorical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)
            self.log_obj.log("INFO", "Replace null type values with NAN in categorical column")


            self.log_obj.log("INFO","Replaced all null type values with NAN values")
            return dataFrame

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))


