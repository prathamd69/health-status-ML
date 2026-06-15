import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

# Setting up encoder classes
labelenc = LabelEncoder()
onehotenc = OneHotEncoder()

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

def preprocessing(rawdata : pd.DataFrame, labelenc : LabelEncoder, onehotenc : OneHotEncoder) -> pd.DataFrame:
    try:
        logger.debug("Starting preprocessing pipeline. Initial shape: %s", rawdata.shape)
        data = rawdata.copy()

        if 'Gender' in data.columns:
            data['Gender'] = pd.Series(
                labelenc.transform(data['Gender']),
                index=data.index,
            )
            logger.debug("Encoded 'Gender' column using LabelEncoder.")
        
        else:
            logger.warning("'Gender' column not found in input data. Skipping LabelEncoder.")

        # 2. One-Hot Encode Diet_Type safely
        if 'Diet_Type' in data.columns:
            diet_encoded = onehotenc.transform(data[['Diet_Type']])
            
            if hasattr(diet_encoded, "toarray"):
                diet_encoded_array = diet_encoded.toarray() # type:ignore
            else:
                diet_encoded_array = diet_encoded
                
            diet_encoded_df = pd.DataFrame(
                diet_encoded_array, # type:ignore
                columns=onehotenc.get_feature_names_out(['Diet_Type']),
                index=data.index
            )   # type:ignore
            data = pd.concat([data, diet_encoded_df], axis=1)
            logger.debug("Successfully added one-hot encoded columns: %s", list(diet_encoded_df.columns))
        
        else:
            logger.warning("'Diet_Type' column not found in input data. Skipping OneHotEncoder.")

        activityMapping = {
            'Sedentary': 0, 'Lightly Active': 1, 'Moderately Active': 2, 
            'Very Active': 3, 'Athlete': 4
        }
        if 'Activity_Level' in data.columns:
            data['Activity_Level'] = data['Activity_Level'].map(activityMapping)
            logger.debug("Mapped 'Activity_Level' values to ordinal integers.")
        
        else:
            logger.warning("'Activity_Level' column not found in input data.")

        healthMapping = {
            'underweight': 0, 'fit': 1, 'overweight': 2, 'obese': 3
        }
        if 'Health_Status' in data.columns:
            data['target'] = data['Health_Status'].map(healthMapping)
            logger.debug("Mapping 'Health_Status' values to ordinal integers.")

        else:
            logger.warning("'Health_Status' column not found in input data.")

        # Drop columns safely (errors='ignore' prevents crashes if columns are missing)
        columns_to_drop = ['Person_ID', 'Height_cm', 'Weight_kg', 'Health_Status', 'Diet_Type']
        existing_drops = [col for col in columns_to_drop if col in data.columns]
        logger.debug("Dropping raw/redundant columns: %s", existing_drops)
        
        data = data.drop(columns=columns_to_drop, errors='ignore')
        
        logger.info("Preprocessing complete successfully. Final shape: %s", data.shape)
        return data

    except Exception as e:
        logger.error("Unexpected Error while data preprocessing: %s", e, exc_info=True)
        raise

def saveData(datadir : str, train_data : pd.DataFrame, test_data : pd.DataFrame) -> None:
    """Saves Pre-Processed data."""

    try:
        os.makedirs(datadir,exist_ok=True)

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
    try:
        rawtrainPath = './data/raw/train.csv'
        rawtestPath = './data/raw/test.csv'

        rawTrain = loadRawData(rawtrainPath)
        rawTest = loadRawData(rawtestPath)

        onehotenc.fit(rawTrain[['Diet_Type']])
        logger.debug("One hot encoding fitted on training data.")
        labelenc.fit(rawTrain[['Gender']])
        logger.debug("Label encoding fitted on training data.")
        
        processedTrain = preprocessing(rawdata=rawTrain, labelenc=labelenc, onehotenc=onehotenc)
        processedTest = preprocessing(rawdata=rawTest, labelenc=labelenc, onehotenc=onehotenc)

        savedir = 'data/processed'
        
        saveData(datadir=savedir, train_data=processedTrain, test_data=processedTest)
        

    except Exception as e:
        logger.error("Data Pre-Processing failed : %s",e)

if __name__ == "__main__":
    main()