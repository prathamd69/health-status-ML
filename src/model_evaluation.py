import pandas as pd
import os
import logging
import numpy as np
import json
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from dvclive.live import Live

# Logging Setup and Configuration

logDir = 'logs'
os.makedirs(logDir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

logPath = os.path.join(logDir,'model_evaluation.log')
fileHandler = logging.FileHandler(logPath)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

def loadData(datapath : str) -> pd.DataFrame:
    """Load a processed CSV file into a DataFrame.

    Args:
        datapath: Path to a CSV file containing processed data.

    Returns:
        A pandas DataFrame loaded from the given file.
    """
    try:
        df = pd.read_csv(datapath)
        logger.debug("Raw data loaded from %s", datapath)
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found at {datapath}")
        raise

    except Exception as e:
        logger.error("Unexpected error loading raw data from %s : %s", datapath, e)
        raise

def load_model(modelPath: str):
    """Load the trained model from a file."""
    try:
        with open(modelPath, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', modelPath)
        return model
    
    except FileNotFoundError:
        logger.error('File not found: %s', modelPath)
        raise

    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def evaluate_model(model, dataPath : str) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        test_final = loadData(dataPath)
        logger.debug("Train and test data loaded successfully.")

        X_test = test_final.drop(columns=['target'])
        y_test = test_final['target']
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        y_pred_proba = model.predict_proba(X_test)

        auc = roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class='ovr',
            average='weighted'
        )

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
        }

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics_dict['accuracy'])
            live.log_metric('precision', metrics_dict['precision'])
            live.log_metric('recall', metrics_dict['recall'])
            live.log_metric('f1', metrics_dict['f1'])
            live.log_metric('auc', metrics_dict['auc'])
            
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def saveMetrics(metrics: dict, metricsPath: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(metricsPath), exist_ok=True)

        with open(metricsPath, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', metricsPath)

    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        logisticmodel = load_model('./models/logisticregressionModel.pkl')
        testPath = './data/final/test.csv'

        metrics = evaluate_model(logisticmodel, dataPath=testPath)

        saveMetrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()