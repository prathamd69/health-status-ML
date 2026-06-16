import pandas as pd
import os
import logging
import yaml
import requests
from io import StringIO
from sklearn.model_selection import train_test_split

# Logging Setup and Configuration

logDir = 'logs'
os.makedirs(logDir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

logPath = os.path.join(logDir,'data_ingestion.log')
fileHandler = logging.FileHandler(logPath)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

def loadParams(paramsPath: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(paramsPath, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', paramsPath)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', paramsPath)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def loadData(dataURL : str) -> pd.DataFrame:
    """
    loadData takes a dataset URL as an input and returns the panda DataFrame of the dataset.
    Exceptions such as invalid URL, connection timeout, failed parsing are handles and logged.
    """

    try:
        response = requests.get(dataURL, timeout=10)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        logger.debug('Data loaded from %s', dataURL)
        return df
    
    except requests.exceptions.HTTPError:
        logger.error("The web page could not be found or is private.")
        raise

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect. Check your internet connection.")
        raise

    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the data: %s', e)
        raise

def saveData(data : pd.DataFrame, datadir : str, testSplit : float) -> None:
    """Split raw data and save it to disk.

    Args:
        data: DataFrame containing the full dataset.
        datadir: Directory path where train/test CSV files should be written.
        testSplit: Fraction of the data to reserve for testing.
    """

    try:
        os.makedirs(datadir,exist_ok=True)

        train_data, test_data = train_test_split(data, test_size=testSplit, random_state=24)

        traindataPath = os.path.join(datadir,'train.csv')
        testdataPath = os.path.join(datadir,'test.csv')

        train_data.to_csv(traindataPath, index=False)
        test_data.to_csv(testdataPath, index=False)
        logger.debug("Training data saved to %s", traindataPath)
        logger.debug("Testing data saved to %s", testdataPath)
    
    except Exception as e:
        logger.error("Unexpected error while saving raw data : %s", e)
        raise
    

def main():
    """Run the ingestion pipeline to download and save raw training and test data."""

    try:
        params = loadParams(paramsPath='params.yaml')
        dataURL = params['data_ingestion']['dataURL']
        testSplit = params['data_ingestion']['testSplit']

        df = loadData(dataURL=dataURL)
        
        savedir = 'data/raw'

        saveData(data=df, datadir=savedir, testSplit=testSplit)

    except Exception as e:
        logger.error("Unexpected error, data ingestion failed : %s",e)
        raise


if __name__ == "__main__":
    main()