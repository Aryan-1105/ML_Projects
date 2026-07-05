import os
import sys
import json
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logger


# ==========================================================
# Function to Save Python Objects
# ==========================================================

def save_object(file_path, obj):
    """
    Saves any Python object as a pickle (.pkl) file.

    This function is mainly used to save:
    - Trained Machine Learning Models
    - Preprocessing Pipelines
    - Encoders
    - Scalers

    Parameters
    ----------
    file_path : str
        Destination path where the object will be stored.

    obj : object
        Any Python object to be saved.
    """

    try:

        # Get directory path
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        logger.info(f"Saving object at: {file_path}")

        # Open file in binary write mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info("Object saved successfully.")

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================================
# Function to Load Saved Objects
# ==========================================================

def load_object(file_path):
    """
    Loads a saved pickle (.pkl) object.

    Parameters
    ----------
    file_path : str
        Path of the pickle file.

    Returns
    -------
    object
        Loaded Python object.
    """

    try:

        logger.info(f"Loading object from: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logger.info("Object loaded successfully.")

        return obj

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================================
# Evaluate Multiple Machine Learning Models
# ==========================================================

def evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        models,
        params
):
    """
    Train multiple regression models,
    perform hyperparameter tuning,
    evaluate their performance,
    and return the trained models.

    Returns
    -------
    dict
        Dictionary containing

        {
            model_name:
            {
                "model": trained_model,
                "train_score": float,
                "test_score": float
            }
        }
    """

    try:

        report = {}

        logger.info("=" * 80)
        logger.info("Model Evaluation Started")
        logger.info("=" * 80)

        for model_name, model in models.items():

            logger.info(f"Training {model_name}")

            param_grid = params.get(model_name, {})

            # --------------------------------------------------
            # Hyperparameter Tuning
            # --------------------------------------------------

            if param_grid:

                search = RandomizedSearchCV(

                    estimator=model,

                    param_distributions=param_grid,

                    n_iter=5,

                    cv=3,

                    scoring="r2",

                    random_state=42,

                    n_jobs=-1

                )

                search.fit(X_train, y_train)

                model = search.best_estimator_

                logger.info(
                    f"Best Parameters : {search.best_params_}"
                )

            else:

                model.fit(X_train, y_train)

            # --------------------------------------------------
            # Predictions
            # --------------------------------------------------

            train_prediction = model.predict(X_train)

            test_prediction = model.predict(X_test)

            # --------------------------------------------------
            # Scores
            # --------------------------------------------------

            train_score = r2_score(
                y_train,
                train_prediction
            )

            test_score = r2_score(
                y_test,
                test_prediction
            )

            logger.info(
                f"{model_name}"
                f" | Train R² : {train_score:.4f}"
                f" | Test R² : {test_score:.4f}"
            )

            report[model_name] = {

                "model": model,

                "train_score": train_score,

                "test_score": test_score

            }

        # --------------------------------------------------
        # Save Model Performance Report
        # --------------------------------------------------

        json_report = {}

        for model_name, values in report.items():

            json_report[model_name] = {

                "Train R2": round(
                    values["train_score"],
                    4
                ),

                "Test R2": round(
                    values["test_score"],
                    4
                )

            }

        os.makedirs("artifacts", exist_ok=True)

        with open(
            "artifacts/model_report.json",
            "w"
        ) as json_file:

            json.dump(
                json_report,
                json_file,
                indent=4
            )

        logger.info(
            "Model Performance Report Saved."
        )

        logger.info("=" * 80)
        logger.info("Model Evaluation Completed")
        logger.info("=" * 80)

        return report

    except Exception as e:

        raise CustomException(e, sys)