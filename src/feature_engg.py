import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import logging

# Logging Setup and Configuration

logDir = 'logs'
os.makedirs(logDir, exist_ok=True)

logger = logging.getLogger('feature_engg')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

logPath = os.path.join(logDir,'feature_engg.log')
fileHandler = logging.FileHandler(logPath)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

#Making scaler object
scaler = StandardScaler()
scaler.set_output(transform="pandas")

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

def features(data : pd.DataFrame) -> pd.DataFrame:
    """Create engineered features and remove redundant columns.

    This function computes a daily calorie index, drops calorie intake/requirement
    source columns, and removes low-correlation or redundant features such as
    diet type encodings, gender, and water intake.

    Args:
        data: A pandas DataFrame with processed dataset features.

    Returns:
        A DataFrame containing the final selected feature set.
    """
    try:
        # Reducing Daily Calorie Intake and Requirement columns into one Daily Calorie Index
        data['daily_calorie_index'] = data['Daily_Calorie_Consumed'] - data['Daily_Calorie_Requirement']
        data = data.drop(columns=['Daily_Calorie_Consumed', 'Daily_Calorie_Requirement'])

        """     Upon feature analysis of data we found the standard correlation 
                of target from features like diet type is 
                almost zero, so we'll drop them.       
                
                Other features we are dropping are - Gender, Water_Intake_Liters """
        
        data = data.drop(columns=['Diet_Type_Balanced',	'Diet_Type_High Protein', 
                                  'Diet_Type_Keto', 'Diet_Type_Mediterranean', 
                                  'Diet_Type_Vegan', 'Diet_Type_Vegetarian',
                                  'Gender', 'Water_Intake_Liters'])
        return data
    
    except Exception as e:
        logger.error("Unexpected error while creating Features : %s", e)
        raise
    
def scalingandsaving(train_data : pd.DataFrame, test_data : pd.DataFrame, scaler : StandardScaler, datadir : str) -> None:
    """Scale numeric features and save final train/test datasets.

    This function applies a StandardScaler to numeric columns while preserving
    excluded identifier or target columns, then writes the final datasets to disk.

    Args:
        train_data: Final training DataFrame before scaling.
        test_data: Final testing DataFrame before scaling.
        scaler: A fitted or unfitted StandardScaler instance.
        datadir: Output directory for the scaled CSV files.
    """
    try:
        cols_to_exclude = ['target', 'Activity_Level', 'Gender']

        exclude_train = [col for col in cols_to_exclude if col in train_data.columns]
        exclude_test = [col for col in cols_to_exclude if col in test_data.columns]

        train_features_to_scale = train_data.drop(columns=exclude_train)
        test_features_to_scale = test_data.drop(columns=exclude_test)

        train_scaled_feats = scaler.fit_transform(train_features_to_scale)
        test_scaled_feats = scaler.transform(test_features_to_scale)

        train_final = pd.concat([train_scaled_feats, train_data[exclude_train]], axis=1) # type:ignore
        test_final = pd.concat([test_scaled_feats, test_data[exclude_test]], axis=1)    # type:ignore

        os.makedirs(datadir, exist_ok=True)
        traindataPath = os.path.join(datadir, 'train.csv')
        testdataPath = os.path.join(datadir, 'test.csv')

        train_final.to_csv(traindataPath, index=False)
        test_final.to_csv(testdataPath, index=False)

        logger.debug("Training data saved to %s", traindataPath)
        logger.debug("Testing data saved to %s", testdataPath)
        logger.info("Feature selection and selective scaling done! Dataset ready.")

    except Exception as e:
        logger.error("Unexpected error while scaling data : %s", e)
        raise
    
def main():
    """Run the feature engineering pipeline and save final scaled datasets."""

    try:
        processedtrainPath = './data/processed/train.csv'
        processedtestPath = './data/processed/test.csv'

        processedTrain = loadRawData(processedtrainPath)
        processedTest = loadRawData(processedtestPath)

        finalTrain = features(processedTrain)
        finalTest = features(processedTest)

        savedir = 'data/final'

        scalingandsaving(train_data=finalTrain, test_data=finalTest, scaler=scaler, datadir=savedir)

    except Exception as e:
        logger.error("Feature engineering failed : %s",e)

if __name__ == "__main__":
    main()
    
 