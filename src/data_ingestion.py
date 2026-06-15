import pandas as pd
import os
import logging
import yaml
import requests
from io import StringIO

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

def saveRawData(data : pd.DataFrame) -> None:
    """Saves raw data."""

    try:
        rawdir = 'rawdata'
        os.makedirs(rawdir,exist_ok=True)

        rawdataPath = os.path.join(rawdir,'data.csv')
        data.to_csv(rawdataPath, index=False)
        logger.debug("Raw data saved to %s", rawdataPath)
    
    except Exception as e:
        logger.error("Unexpected error while saving raw data : %s", e)
        raise
    

def main():
    try:
        dataURL = 'https://raw.githubusercontent.com/prathamd69/datasets/refs/heads/main/healthdataset.csv'
        df = loadData(dataURL=dataURL)
        print(df.head())
        saveRawData(df)

    except Exception as e:
        logger.error("Unexpected error, data ingestion failed : %s",e)
        raise


if __name__ == "__main__":
    main()