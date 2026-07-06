"""
==========================================================
Train Pipeline

This module is responsible for executing the complete
Machine Learning Training Pipeline.

Workflow
--------
1. Data Ingestion
2. Data Transformation
3. Model Training
==========================================================
"""

import sys

from src.logger import logger
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# ==========================================================
# Train Pipeline Class
# ==========================================================

class TrainPipeline:
    """
    Executes the complete ML training pipeline.
    """

    def __init__(self):
        pass

    # ======================================================
    # Run Complete Pipeline
    # ======================================================

    def run_pipeline(self):

        try:

            logger.info("=" * 80)
            logger.info("STARTING COMPLETE TRAINING PIPELINE")
            logger.info("=" * 80)

            # --------------------------------------------------
            # Step 1 : Data Ingestion
            # --------------------------------------------------

            logger.info("STEP 1 : DATA INGESTION")

            ingestion = DataIngestion()

            train_path, test_path = (
                ingestion.initiate_data_ingestion()
            )

            logger.info("Data Ingestion Completed Successfully.")

            # --------------------------------------------------
            # Step 2 : Data Transformation
            # --------------------------------------------------

            logger.info("STEP 2 : DATA TRANSFORMATION")

            transformation = DataTransformation()

            train_array, test_array, preprocessor_path = (

                transformation.initiate_data_transformation(

                    train_path,

                    test_path

                )

            )

            logger.info("Data Transformation Completed Successfully.")

            # --------------------------------------------------
            # Step 3 : Model Training
            # --------------------------------------------------

            logger.info("STEP 3 : MODEL TRAINING")

            trainer = ModelTrainer()

            best_model_name, final_r2_score = (

                trainer.initiate_model_trainer(

                    train_array,

                    test_array

                )

            )

            logger.info("Model Training Completed Successfully.")

            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE FINISHED")
            logger.info("=" * 80)

            return {

                "best_model": best_model_name,

                "r2_score": final_r2_score,

                "preprocessor_path": preprocessor_path

            }

        except Exception as e:

            logger.exception("Training Pipeline Failed.")

            raise CustomException(e, sys)


# ==========================================================
# Run Pipeline
# ==========================================================

if __name__ == "__main__":

    pipeline = TrainPipeline()

    result = pipeline.run_pipeline()

    print("\n")
    print("=" * 80)
    print("TRAINING PIPELINE SUMMARY")
    print("=" * 80)

    print(f"Best Model        : {result['best_model']}")
    print(f"Final Test R²     : {result['r2_score']:.4f}")
    print(f"Preprocessor Path : {result['preprocessor_path']}")

    print("=" * 80)