from Logger.Logging import LogClass
from FeatrueEngineering.FeatureEngineering import FeatureEngineering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
from math import sqrt
import os
import warnings
warnings.filterwarnings('ignore')

class ModelBuilder:
    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Model_Building.txt'
        self.featureengineering = FeatureEngineering()
        self.randomforest = RandomForestRegressor()
        self.xgb = XGBRegressor()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj = LogClass(self.folder, self.filename)

    def random_forest(self):


        try:
            self.log_obj.log("INFO", "Entered into best_params_class for random_forest Regressor")
            data = self.featureengineering.training_feature_selection()
            X = data.drop(columns=['Item_Outlet_Sales'])
            y = data['Item_Outlet_Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            self.param_grid = {'criterion': ['squared_error'], 'max_depth': [890], 'max_features': ['auto'],


                               'min_samples_leaf': [2, 4, 6, 8, 10],


                               'min_samples_split': [0, 1, 2, 3, 4, 5],


                               'n_estimators': [400, 500, 600, 700, 800]

                               }
            self.grid = GridSearchCV(estimator=self.randomforest, param_grid=self.param_grid, cv=10, n_jobs=-1, verbose=2)
            self.grid.fit(X_train, y_train)

            self.log_obj.log("INFO", "Grid_Search cv is performed ")
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.n_estimators = self.grid.best_params_['n_estimators']

            self.log_obj.log("INFO", "Random forest modelling fitting has started")

            self.rfr = RandomForestRegressor(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features,
                                              min_samples_leaf=self.min_samples_leaf,
                                              min_samples_split=self.min_samples_split)
            self.rfr.fit(X_train, y_train)

            self.log_obj.log("INFO", f"Best random forest parameters are {self.grid.best_params_}")
            self.log_obj.log("INFO", "Random Forest Modelling completed")

            return self.rfr

        except Exception as e:
            self.log_obj.log("ERROR", "Exception occured at Random forest modelling, Exception is :" + '\n' + str(e))

    def xg_boost(self):

        try:
            self.log_obj.log("INFO", "Entered into best_params for xg boost class")
            data = self.featureengineering.training_feature_selection()
            X = data.drop(columns=['Item_Outlet_Sales'])
            y = data['Item_Outlet_Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            self.param_grid_xg = {

                        'n_estimators': [100, 200, 300, 400, 500],


                        'learning_rate': [0.1, 0.3, 0.01, 0.001, 0.0001],


                        'max_depth': [10, 120, 230, 340, 400]
            }

            self.grid_xg = GridSearchCV(XGBRegressor(objective='reg:squarederror'), self.param_grid_xg, verbose=3, cv=10)
            self.grid_xg.fit(X_train, y_train)

            self.learning_rate = self.grid_xg.best_params_['learning_rate']
            self.n_estimators = self.grid_xg.best_params_['n_estimators']
            self.max_depth = self.grid_xg.best_params_['max_depth']

            self.log_obj.log('INFO', "Xg_boost modelling has started")
            self.xgb = XGBRegressor(learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                     max_depth=self.max_depth)

            self.xgb.fit(X_train, y_train)
            self.log_obj.log("INFO", f"Best xg_boost parameters are :{self.grid_xg.best_params_}")
            self.log_obj.log("INFO", "XGboost modelling has been completed")

            return self.xgb

        except Exception as e:
            self.log_obj.log("Error", "Occured Exception is:" + '\n' + str(e))

    def get_best_model(self):

        try:
            self.log_obj.log("INFO", "Enter get_best_model method in ModelBuilder class ")
            data = self.featureengineering.training_feature_selection()
            X = data.drop(columns=['Item_Outlet_Sales'])
            y = data['Item_Outlet_Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            self.xgboost = self.xg_boost()
            self.log_obj.log('INFO', 'Get the prediction of XGbRegressor Model')
            self.prediction_xgboost = self.xgboost.predict(X_test)

            self.log_obj.log('INFO', 'Get the Root Mean Squared Error of XGbRegressor Model')
            self.xgboost_score = sqrt(mean_squared_error(y_test, self.prediction_xgboost))
            self.log_obj.log("INFO", f"Root Mean Squared Error of Xgboost model is {self.xgboost_score}")


            self.random_forest = self.random_forest()
            self.log_obj.log('INFO', 'Get the prediction of RandomForestRegressor Model')
            self.prediction_random_forest = self.random_forest.predict(X_test)

            self.log_obj.log('INFO', 'Get the Root Mean Squared Error of RandomForestRegressor Model')
            self.random_forest_score = sqrt(mean_squared_error(y_test, self.prediction_random_forest))
            self.log_obj.log("INFO", f"Root Mean Squared Error of random forest is {self.random_forest_score}")

            self.log_obj.log("INFO", "Selection of best model ")
            self.log_obj.log('INFO', "Whomever the RMSE is low we will select that model")
            if (self.random_forest_score > self.xgboost_score):
                self.best_model_name = 'xgboost_regressor'
                self.best_model = self.xgboost
                self.best_score = self.xgboost_score
            else:
                self.best_model_name = 'Random_forest_regressor'
                self.best_model = self.random_forest
                self.best_score = self.random_forest_score

            self.log_obj.log("INFO", "Best model selection is done")
            self.log_obj.log("INFO", "Saving model as pickle file")

            path = '../BestModel/'

            if not os.path.isdir(path):
                os.mkdir(path)

            with open(path + self.best_model_name + '.pkl', 'wb') as file:
                pickle.dump(self.best_model, file)

            self.log_obj.log("INFO", f"Best model saved in file ,Best model is:{self.best_model_name}")

            return self.best_model_name

        except Exception as e:
            self.log_obj.log("Error", "Exception occured at get_best_model  is:" + str(e))


model = ModelBuilder()
model.get_best_model()