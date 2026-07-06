import os
import sys

import pandas as pd

# ==========================================================
# Custom Modules
# ==========================================================

from src.exception import CustomException
from src.logger import logger

from src.utils import load_object


# ==========================================================
# Prediction Pipeline
# ==========================================================

class PredictPipeline:
    """
    This class is responsible for

    1. Loading the trained model
    2. Loading the preprocessing object
    3. Transforming new data
    4. Returning prediction
    """

    def __init__(self):
        pass

    # ======================================================
    # Predict Method
    # ======================================================

    def predict(self, features):

        try:

            logger.info("=" * 80)
            logger.info("Prediction Pipeline Started")
            logger.info("=" * 80)

            # --------------------------------------------------
            # Load Model
            # --------------------------------------------------

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            logger.info(f"Loading Model : {model_path}")

            model = load_object(model_path)

            # --------------------------------------------------
            # Load Preprocessor
            # --------------------------------------------------

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            logger.info(
                f"Loading Preprocessor : {preprocessor_path}"
            )

            preprocessor = load_object(
                preprocessor_path
            )

            # --------------------------------------------------
            # Transform Input Features
            # --------------------------------------------------

            logger.info(
                "Applying Data Transformation..."
            )

            transformed_features = preprocessor.transform(
                features
            )

            # --------------------------------------------------
            # Predict
            # --------------------------------------------------

            prediction = model.predict(
                transformed_features
            )

            logger.info(
                f"Prediction Completed : {prediction[0]}"
            )

            return prediction

        except Exception as e:

            logger.exception(
                "Prediction Pipeline Failed."
            )

            raise CustomException(e, sys)

# ==========================================================
# Custom Data Class
# ==========================================================

class CustomData:
    """
    Collects user input
    and converts it into
    a Pandas DataFrame.
    """

    def __init__(

        self,

        gender,

        race_ethnicity,

        parental_level_of_education,

        lunch,

        test_preparation_course,

        reading_score,

        writing_score

    ):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = (
            parental_level_of_education
        )

        self.lunch = lunch

        self.test_preparation_course = (
            test_preparation_course
        )

        self.reading_score = reading_score

        self.writing_score = writing_score

    # ======================================================
    # Convert to DataFrame
    # ======================================================

    def get_data_as_dataframe(self):

        try:

            custom_data_input = {

                "gender": [self.gender],

                "race_ethnicity": [
                    self.race_ethnicity
                ],

                "parental_level_of_education": [
                    self.parental_level_of_education
                ],

                "lunch": [
                    self.lunch
                ],

                "test_preparation_course": [
                    self.test_preparation_course
                ],

                "reading_score": [
                    self.reading_score
                ],

                "writing_score": [
                    self.writing_score
                ]

            }

            dataframe = pd.DataFrame(
                custom_data_input
            )

            logger.info(
                "Custom DataFrame Created Successfully."
            )

            return dataframe

        except Exception as e:

            logger.exception(
                "Unable to Create DataFrame."
            )

            raise CustomException(e, sys)