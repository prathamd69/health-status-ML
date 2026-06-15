import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Logging Setup and Configuration

logDir = 'logs'
os.makedirs(logDir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

logPath = os.path.join(logDir,'data_preprocessing.log')
fileHandler = logging.FileHandler(logPath)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

def loadRawData(datapath : str) -> pd.DataFrame:
    """"""
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

def preprocessing(data : pd.DataFrame) -> pd.DataFrame:
    try:
        data = data.drop(columns=['Person_ID'])

        # Reducing Daily Calorie Intake and Requirement columns into one Daily Calorie Index
        data['daily_calorie_index'] = data['Daily_Calorie_Consumed'] - data['Daily_Calorie_Requirement']
        data = data.drop(columns=['Daily_Calorie_Consumed', 'Daily_Calorie_Requirement'])
        
        return data
    except Exception as e:
        logger.error("Unexpected Error while data preprocessing : %s", e)
        raise

def main():
    try:
        datapath = 'rawdata/data.csv'
        df = loadRawData(datapath)
        print(df.head())

    except Exception as e:
        logger.error("Data Pre-Processing failed : %s",e)

if __name__ == "__main__":
    main()