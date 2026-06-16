import pandas as pd
import logging
import os
import pickle
from sklearn.linear_model import LogisticRegression

# Logging Setup and Configuration

logDir = 'logs'
os.makedirs(logDir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

logPath = os.path.join(logDir,'model_training.log')
fileHandler = logging.FileHandler(logPath)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)


def loadRawData(datapath : str) -> pd.DataFrame:
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

def model_building(trainpath : str, max_iter: int) -> LogisticRegression:
    try:
        train_final = loadRawData(trainpath)
        logger.debug("Train and test data loaded successfully.")

        X_train = train_final.drop(columns=['target'])
        y_train = train_final['target']

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug("Data split into training and validation sets successfully. " 
        "Training samples: %d. Training started.", X_train.shape[0])
        
        lr = LogisticRegression(
            max_iter=max_iter,
            random_state=42
        )

        lr.fit(X_train, y_train)
        logger.info('Model training completed')

        return lr

    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise

    except Exception as e:
        logger.error("Unexpected error during model building: %s", e)
        raise

def save_model(model, filePath: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        
        with open(filePath, 'wb') as file:
            pickle.dump(model, file)
        logger.info('Model saved to %s', filePath)

    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise

    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        trainpath = './data/final/train.csv'
        max_iter = 100

        lrmodel = model_building(trainpath, max_iter)
        
        modelDir = 'models'
        os.makedirs(modelDir, exist_ok=True)
        modelPath = os.path.join(modelDir,'logisticregressionModel.pkl')
 
        save_model(lrmodel, modelPath)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()