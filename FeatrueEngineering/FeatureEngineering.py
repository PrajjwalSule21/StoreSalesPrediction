import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from DataPreprocessing.Preprocessing import Preprocessing
from Logger.Logging import LogClass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



class FeatureEngineering:
    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Feature_Engineering.txt'
        self.preprocessing = Preprocessing()
        self.scaler = StandardScaler()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj = LogClass(self.folder, self.filename)


    def training_categorical_encoder(self):
        try:
            self.log_obj.log("INFO", "Training Data encoding has been started")
            data = self.preprocessing.trainig_column_cleaner()

            ## temporary impute
            self.log_obj.log("INFO", "Impute the Missing values on temporary basis")
            data['Item_Weight'].fillna(0.000, inplace=True)
            data['Outlet_Size'].fillna('Missing', inplace=True)

            X = data.drop(columns=['Item_Outlet_Sales'])
            y = data['Item_Outlet_Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.log_obj.log("INFO", "Initialize the transformer with OneHotEncoding for training data")
            transformer = ColumnTransformer(transformers=[
                ('Ordinalencoder', OrdinalEncoder(
                    categories=[['Low Fat', 'Regular'], ['Small', 'Medium', 'High', 'Missing'],
                                ['Tier 1', 'Tier 2', 'Tier 3'],
                                ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']]),
                 ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']),
                ('Ordinalencoder2', OrdinalEncoder(categories=[
                    ['Others', 'Seafood', 'Breakfast', 'Starchy Foods', 'Hard Drinks', 'Breads', 'Meat', 'Soft Drinks',
                     'Health and Hygiene', 'Baking Goods', 'Canned', 'Dairy', 'Frozen Foods', 'Household',
                     'Snack Foods', 'Fruits and Vegetables']]), ['Item_Type'])
            ], remainder='passthrough')

            # Encode all the feature with the help of transformer
            self.log_obj.log("INFO", "Apply the training data transformer on X_train and X_test")
            X_train_encoder = transformer.fit_transform(X_train)
            X_test_encoder = transformer.transform(X_test)


            #make the train and test array seperately
            self.log_obj.log("INFO", "Making the Train and Test Array of training data seperately ")
            newtrainArray_encoder = np.column_stack((X_train_encoder, y_train))
            newtestArray_encoder = np.column_stack((X_test_encoder, y_test))

            self.log_obj.log("INFO", "Join those train and test array into one complete array ")
            new = np.vstack((newtrainArray_encoder, newtestArray_encoder))
            self.log_obj.log("INFO", "Make a dataframe with that complete array of encoded training data")

            data = pd.DataFrame(new, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type', 'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Age_Outlet', 'Item_Outlet_Sales'])
            self.log_obj.log('INFO', 'Training Data Enconding is completed')

            return data


        except Exception as e:
            self.log_obj.log('Error', "Exception is:" + '\t' + str(e))


    def testing_categorical_encoder(self):
        try:
            self.log_obj.log("INFO", "Testing Categorical Encoding has been started")
            data = self.preprocessing.testing_column_cleaner()

            ## temporary impute
            data['Item_Weight'].fillna(0.000, inplace=True)
            data['Outlet_Size'].fillna('Missing', inplace=True)

            X = data.drop(columns=['Item_Outlet_Sales'])
            y = data['Item_Outlet_Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.log_obj.log("INFO", "Initialize the transformer with OneHotEncoding and OrdinalEncoding")
            transformer = ColumnTransformer(transformers=[
                ('OrdinalEncoder', OrdinalEncoder(
                    categories=[['Low Fat', 'Regular'], ['Small', 'Medium', 'High', 'Missing'],
                                ['Tier 1', 'Tier 2', 'Tier 3'],
                                ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],
                                ['Others', 'Seafood', 'Breakfast', 'Starchy Foods', 'Hard Drinks', 'Breads', 'Meat',
                                 'Soft Drinks', 'Health and Hygiene', 'Baking Goods', 'Canned', 'Dairy', 'Frozen Foods',
                                 'Household', 'Snack Foods', 'Fruits and Vegetables']]),
                 ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type'])
            ], remainder='passthrough')

            # Encode all the feature with the help of transformer
            self.log_obj.log("INFO", "Encode all the feature with the help of transformer")
            X_train_new = transformer.fit_transform(X_train)
            X_test_new = transformer.transform(X_test)

            # make the train and test array seperately
            self.log_obj.log("INFO", "Making the Train and Test Array seperately ")
            newtrainArray = np.column_stack((X_train_new, y_train))
            newtestArray = np.column_stack((X_test_new, y_test))

            self.log_obj.log("INFO", "Join those train and test array into one complete array ")
            new = np.vstack((newtrainArray, newtestArray))
            self.log_obj.log("INFO", "Make a dataframe with that complete array")
            data = pd.DataFrame(new, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type', 'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Age_Outlet', 'Item_Outlet_Sales'])

            self.log_obj.log('INFO', 'Testing data encoding is complete')

            return data


        except Exception as e:
            self.log_obj.log('Error', "Exception is:" + '\t' + str(e))


    def training_missing_value_imputation(self):
        try:
            self.log_obj.log('INFO', 'Training Data Missing value imputation is start')
            data = self.training_categorical_encoder()

            self.log_obj.log('INFO', 'Remove those temporary imputers from Item_Weight and Outlet_Size')
            data['Item_Weight'] = data['Item_Weight'].replace(0.000, np.nan)
            data['Outlet_Size'] = data['Outlet_Size'].replace(3.0, np.nan)

            self.log_obj.log('INFO', 'Initalize KNN tranformer for missing value imputation')
            transformer = ColumnTransformer(transformers=[
                ('KnnImputer', KNNImputer(n_neighbors=5, weights='distance'), ['Item_Weight']),
                ('SimpleImputer', SimpleImputer(strategy='most_frequent'), ['Outlet_Size'])
            ], remainder='passthrough')

            X = data.drop(columns=['Item_Outlet_Sales'])
            y = data['Item_Outlet_Sales']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.log_obj.log('INFO', 'Apply the KNN tranformer on X_train and X_test')
            X_train_new = transformer.fit_transform(X_train)
            X_test_new = transformer.transform(X_test)

            newtrainArray = np.column_stack((X_train_new, y_train))
            newtestArray = np.column_stack((X_test_new, y_test))

            new = np.vstack((newtrainArray, newtestArray))

            self.log_obj.log('INFO', 'Make a new data frame for training data')
            data = pd.DataFrame(new, columns=[ 'Item_Weight', 'Outlet_Size', 'Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type', 'Item_Visibility', 'Item_MRP', 'Age_Outlet', 'Item_Outlet_Sales'])

            self.log_obj.log('INFO', 'Training Data Missing value imputation is complete')

            return data

        except Exception as e:
            self.log_obj.log('ERROR', "Exception is:" + '\t' + str(e))

    def testing_missing_value_imputation(self):
        try:
            self.log_obj.log('INFO', 'Testing Data Missing value imputation is start')
            data = self.testing_categorical_encoder()

            self.log_obj.log('INFO', 'Remove those temporary imputers from Item_Weight and Outlet_Size')
            data['Item_Weight'] = data['Item_Weight'].replace(0.000, np.nan)
            data['Outlet_Size'] = data['Outlet_Size'].replace(3.0, np.nan)

            self.log_obj.log('INFO', 'Initalize KNN tranformer for missing value imputation')
            transformer = ColumnTransformer(transformers=[
                ('KnnImputer', KNNImputer(n_neighbors=5, weights='distance'), ['Item_Weight']),
                ('SimpleImputer', SimpleImputer(strategy='most_frequent'), ['Outlet_Size'])
            ], remainder='passthrough')

            X = data.drop(columns=['Item_Outlet_Sales'])
            y = data['Item_Outlet_Sales']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.log_obj.log('INFO', 'Apply the KNN tranformer on X_train and X_test')
            X_train_new = transformer.fit_transform(X_train)
            X_test_new = transformer.transform(X_test)

            newtrainArray = np.column_stack((X_train_new, y_train))
            newtestArray = np.column_stack((X_test_new, y_test))

            new = np.vstack((newtrainArray, newtestArray))

            self.log_obj.log('INFO', 'Make a new data frame for test Data')
            data = pd.DataFrame(new, columns=[ 'Item_Weight', 'Outlet_Size', 'Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type', 'Item_Visibility', 'Item_MRP', 'Age_Outlet', 'Item_Outlet_Sales'])

            self.log_obj.log('INFO', 'Testing Data Missing value imputation is complete')
            return data

        except Exception as e:
            self.log_obj.log('ERROR', "Exception is:" + '\t' + str(e))



    def training_outlier_treatment(self):
        try:
            self.log_obj.log('INFO', 'Outlier treatment of Item_Visibility column')
            data = self.training_missing_value_imputation()

            self.log_obj.log('INFO', 'Getting the 25% and 75% value')
            percentile25 = data['Item_Visibility'].quantile(0.25)
            percentile75 = data['Item_Visibility'].quantile(0.75)

            self.log_obj.log('INFO', 'Get the IQR Value')
            IQR = percentile75 - percentile25

            self.log_obj.log('INFO', 'Set the Upper and Lower limit')
            upper_limit = percentile75 + 1.5 * IQR
            lower_limit = percentile25 - 1.5 * IQR
            self.log_obj.log('INFO', 'Outlier values are quit high so, we use Capping instead of Trimming')

            self.log_obj.log('INFO', 'Start the Capping Method')

            data['Item_Visibility'] = np.where(
                data['Item_Visibility'] > upper_limit,
                upper_limit,
                np.where(
                    data['Item_Visibility'] < lower_limit,
                    lower_limit,
                    data['Item_Visibility']
                )
            )

            self.log_obj.log('INFO', 'Training Data Outlier handling is complete')
            return data

        except Exception as e:
            self.log_obj.log('ERROR', "Exception is:" + '\t' + str(e))

    def testing_outlier_treatment(self):
        try:
            self.log_obj.log('INFO', 'Outlier treatment of Item_Visibility column in testing data')
            data = self.training_missing_value_imputation()

            self.log_obj.log('INFO', 'Getting the 25% and 75% value')
            percentile25 = data['Item_Visibility'].quantile(0.25)
            percentile75 = data['Item_Visibility'].quantile(0.75)

            self.log_obj.log('INFO', 'Get the IQR Value')
            IQR = percentile75 - percentile25

            self.log_obj.log('INFO', 'Set the Upper and Lower limit')
            upper_limit = percentile75 + 1.5 * IQR
            lower_limit = percentile25 - 1.5 * IQR
            self.log_obj.log('INFO', 'Outlier values are quit high so, we use Capping instead of Trimming')

            self.log_obj.log('INFO', 'Start the Capping Method')

            data['Item_Visibility'] = np.where(
                data['Item_Visibility'] > upper_limit,
                upper_limit,
                np.where(
                    data['Item_Visibility'] < lower_limit,
                    lower_limit,
                    data['Item_Visibility']
                )
            )

            self.log_obj.log('INFO', 'Testing Data Outlier handling is complete')

            return data

        except Exception as e:
            self.log_obj.log('ERROR', "Exception is:" + '\t' + str(e))


    def training_feature_selection(self):
        try:
            self.log_obj.log('INFO', 'Training Feature selection is start')
            data = self.training_outlier_treatment()

            self.log_obj.log('INFO', 'Select the important features and remove the unimportant features')
            data.drop(columns=['Outlet_Location_Type', 'Item_Visibility', 'Age_Outlet'], inplace=True)
            self.log_obj.log('INFO',"Remove ['Outlet_Location_Type', 'Item_Visibility', 'Age_Outlet'] because these columns are not contributing much")

            self.log_obj.log('INFO', 'Get the dataframe of best features')
            self.log_obj.log('INFO', 'Training Feature Selection is done')
            return data



        except Exception as e:
            self.log_obj.log('ERROR', "Exception is:" + '\t' + str(e))

    def testing_feature_selection(self):
        try:
            self.log_obj.log('INFO', 'Testing Feature selection is start')
            data = self.training_outlier_treatment()

            self.log_obj.log('INFO', 'Select the important features and remove the unimportant features')
            data.drop(columns=['Outlet_Location_Type', 'Item_Visibility', 'Age_Outlet'], inplace=True)
            self.log_obj.log('INFO',
                             "Remove ['Outlet_Location_Type', 'Item_Visibility', 'Age_Outlet'] because these columns are not contributing much")

            self.log_obj.log('INFO', 'Get the dataframe of best features')
            self.log_obj.log('INFO', 'Testing Feature Selection is done')
            return data

        except Exception as e:
            self.log_obj.log('ERROR', "Exception is:" + '\t' + str(e))












