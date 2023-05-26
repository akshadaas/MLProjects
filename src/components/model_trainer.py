import os,sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initial_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and testing input data")
            X_train, X_test, y_train, y_test = train_arr[:,:-1],  test_arr[:,:-1], train_arr[:,-1], test_arr[:,-1]
            models = {

                "LinearRegression" : LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNN" : KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "Random_Forest_Regressor": RandomForestRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor()

            }

            model_report:dict = evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test, models=models)
            #print(model_report)
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model")
            logging.info("Best model is found on dataset")

            save_obj(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_sc = r2_score(y_test,predicted)
            print("Best model is ",best_model_name,r2_sc)
            return r2_sc
        
        except Exception as e:
            raise CustomException(e,sys)