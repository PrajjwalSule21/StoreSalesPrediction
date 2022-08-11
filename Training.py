from FeatrueEngineering.FeatureEngineering import FeatureEngineering
from ModelBuilding.ModelBuilding import ModelBuilder


def training():
    try:
        fe = FeatureEngineering()
        fe.training_outlier_treatment()
        fe.testing_outlier_treatment()

        training_obj = ModelBuilder()
        training_obj.get_best_model() # find out the best model
    except Exception as e:
        return "Error"+str(e)


if __name__ == '__main__':
    training()














