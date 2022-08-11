import warnings
warnings.filterwarnings('ignore')
from Logger.Logging import LogClass
import os
from DataLoader.DataGetter import GetData

class Features:

    def __init__(self):
        self.folder = '../LogFiles/'
        self.file_name = "RawData_Feature_Classification.txt"
        self.df_object = GetData()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj = LogClass(self.folder,self.file_name)

    def training_numerical_features(self):

        """
                       Method: traning_numerical_features
                       Description:This method is used to get all the numerical features of the given training dataset
                       Parameters:None
                       Return:Numerical variables of the dataset


        """

        try:
            dataframe = self.df_object.load_trainig_data()
            numerical = [i for i in dataframe.columns if
                         dataframe[i].dtypes == 'int64' or dataframe[i].dtypes == 'float64']
            self.log_obj.log("INFO", f"Numerical features in dataset are: {numerical}")
            return numerical

        except Exception as e:
            self.log_obj.log("ERROR", "Exception caused.Error is:" + str(e))

    def training_categorical_features(self):

        """
                        Method: traning_categorical_features
                        Description: This method is used to get all the categorical features of the given training dataset
                        Parameters: None
                        Return: categorical variables of the dataset


        """

        try:
            dataframe = self.df_object.load_trainig_data()
            categorical = [i for i in dataframe.columns if dataframe[i].dtypes == 'O']
            self.log_obj.log("INFO", f"Categorical features are:{categorical}")
            return categorical

        except Exception as e:
            self.log_obj.log("ERROR", "Exception caused.Error is:" + str(e))


    def testing_numerical_features(self):

        """
                       Method: testing_numerical_features
                       Description:This method is used to get all the numerical features of the given testing dataset
                       Parameters:None
                       Return:Numerical variables of the dataset


        """

        try:
            dataframe = self.df_object.load_testing_data()
            numerical = [i for i in dataframe.columns if
                         dataframe[i].dtypes == 'int64' or dataframe[i].dtypes == 'float64']
            self.log_obj.log("INFO", f"Numerical features in dataset are: {numerical}")
            return numerical

        except Exception as e:
            self.log_obj.log("ERROR", "Exception caused.Error is:" + str(e))

    def testing_categorical_features(self):

        """
                        Method: traning_categorical_features
                        Description: This method is used to get all the categorical features of the given testing dataset
                        Parameters: None
                        Return: categorical variables of the dataset


        """

        try:
            dataframe = self.df_object.load_testing_data()
            categorical = [i for i in dataframe.columns if dataframe[i].dtypes == 'O']
            self.log_obj.log("INFO", f"Categorical features are:{categorical}")
            return categorical

        except Exception as e:
            self.log_obj.log("ERROR", "Exception caused.Error is:" + str(e))



