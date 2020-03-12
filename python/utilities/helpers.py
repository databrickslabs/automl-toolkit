class Helpers:

    @staticmethod
    def check_model_family(model_family: str):
        supported_models = ["RandomForest","XGBoost", "LogisticRegresesion","Trees","GBT","LinearRegression",
                                "MLPC", "SVM"]
        if model_family not in supported_models:
            raise Exception("Your model family but be within any of the following supported model types:",
                            supported_models)

    @staticmethod
    def check_prediction_type(prediction_type:str):
        supported_prediction_types = ['regressor', 'classifier']
        if prediction_type not in supported_prediction_types:
            raise Exception("Prediction type is not supported - it must be one of the following",
                            supported_prediction_types)