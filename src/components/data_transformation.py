import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Scikit-Learn Preprocessing Libraries
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Custom Modules
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


# ==========================================================
# Configuration Class
# ==========================================================

@dataclass
class DataTransformationConfig:
    """
    Configuration class for Data Transformation.
    Stores the location where the preprocessing
    object will be saved.
    """

    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


# ==========================================================
# Data Transformation Class
# ==========================================================

class DataTransformation:
    """
    Responsible for preprocessing the dataset.

    Tasks:
    -------
    1. Handle Missing Values
    2. Encode Categorical Features
    3. Scale Numerical Features
    4. Save the Preprocessor Object
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ======================================================
    # Create Preprocessing Pipeline
    # ======================================================

    def get_data_transformer_object(self):
        """
        Creates preprocessing pipelines for
        numerical and categorical columns.

        Returns
        -------
        ColumnTransformer
            Complete preprocessing pipeline.
        """

        try:

            logger.info("=" * 60)
            logger.info("Creating Data Transformation Pipeline")
            logger.info("=" * 60)

            # ---------------------------------------------
            # Numerical Features
            # ---------------------------------------------

            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            # ---------------------------------------------
            # Categorical Features
            # ---------------------------------------------

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            logger.info(f"Numerical Columns: {numerical_columns}")
            logger.info(f"Categorical Columns: {categorical_columns}")

            # ---------------------------------------------
            # Numerical Pipeline
            # ---------------------------------------------

            numerical_pipeline = Pipeline(

                steps=[

                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )

                ]

            )

            logger.info("Numerical Pipeline Created.")

            # ---------------------------------------------
            # Categorical Pipeline
            # ---------------------------------------------

            categorical_pipeline = Pipeline(

                steps=[

                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),

                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    ),

                    (
                        "scaler",
                        StandardScaler(with_mean=False)
                    )

                ]

            )

            logger.info("Categorical Pipeline Created.")

            # ---------------------------------------------
            # Combine Both Pipelines
            # ---------------------------------------------

            preprocessor = ColumnTransformer(

                [

                    (
                        "numerical_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),

                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )

                ]

            )

            logger.info("Preprocessor Object Created Successfully.")

            return preprocessor

        except Exception as e:
            logger.exception("Error while creating preprocessing pipeline.")
            raise CustomException(e, sys)

    # ======================================================
    # Apply Transformation
    # ======================================================

    def initiate_data_transformation(
            self,
            train_path,
            test_path
    ):
        """
        Applies preprocessing on both
        training and testing datasets.

        Parameters
        ----------
        train_path : str
            Path of training dataset.

        test_path : str
            Path of testing dataset.

        Returns
        -------
        train_array
        test_array
        preprocessor.pkl path
        """

        try:

            logger.info("=" * 60)
            logger.info("Starting Data Transformation")
            logger.info("=" * 60)

            # ---------------------------------------------
            # Read Datasets
            # ---------------------------------------------

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Train Shape : {train_df.shape}")
            logger.info(f"Test Shape  : {test_df.shape}")

            # ---------------------------------------------
            # Target Column
            # ---------------------------------------------

            target_column = "math_score"

            # ---------------------------------------------
            # Split Features & Target
            # ---------------------------------------------

            X_train = train_df.drop(columns=[target_column])

            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])

            y_test = test_df[target_column]

            logger.info("Separated Features and Target.")

            # ---------------------------------------------
            # Create Preprocessor
            # ---------------------------------------------

            preprocessing_object = self.get_data_transformer_object()

            # ---------------------------------------------
            # Fit on Train
            # ---------------------------------------------

            X_train_transformed = preprocessing_object.fit_transform(
                X_train
            )

            logger.info("Training Data Transformed.")

            # ---------------------------------------------
            # Transform Test
            # ---------------------------------------------

            X_test_transformed = preprocessing_object.transform(
                X_test
            )

            logger.info("Testing Data Transformed.")

            # ---------------------------------------------
            # Merge Features & Target
            # ---------------------------------------------

            train_array = np.c_[

                X_train_transformed,

                np.array(y_train)

            ]

            test_array = np.c_[

                X_test_transformed,

                np.array(y_test)

            ]

            # ---------------------------------------------
            # Save Preprocessor
            # ---------------------------------------------

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,

                obj=preprocessing_object

            )

            logger.info(
                f"Preprocessor saved at: "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            logger.info("=" * 60)
            logger.info("Data Transformation Completed")
            logger.info("=" * 60)

            return (

                train_array,

                test_array,

                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:

            logger.exception("Error during Data Transformation.")

            raise CustomException(e, sys)


# ==========================================================
# Testing the Component
# ==========================================================

if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        transformer.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    print("Train Array Shape :", train_arr.shape)
    print("Test Array Shape  :", test_arr.shape)
    print("Preprocessor      :", preprocessor_path)