import sys
import pandas as pd
from sqlalchemy import create_engine

def convert_categories(categories: pd.DataFrame)->pd.DataFrame:
    '''
    Function: change string category to numerical (0-1)
    Args:DataFrame of the categories to be encoded
    Return:encoded categories
    '''
    for column in categories:

        categories[column] = categories[column].map(
            lambda x: 1 if int(x.split("-")[1]) > 0 else 0 )
    return categories


def split_categories(categories: pd.DataFrame)->pd.DataFrame:
    '''
    Function to transform categories into one-hot encoded 
    Args: categories 
    Return: encoded categories 
    '''
    categories = categories['categories'].str.split(';',expand=True)
    row = categories.iloc[[1]].values[0]
    categories.columns = [ x.split("-")[0] for x in row]
    categories = convert_categories(categories)
    #return the encoded categories
    return categories



def load_data(messages_filepath: str, categories_filepath: str)->pd.DataFrame:
    '''
    Function to load the dataset
    Args:messages_filepath: the file path to the messages csv
         categories_filepath: the file path to the categories csv
    Return: a merged dataset
    '''
    messages = pd.read_csv(messages_filepath)
    categories = split_categories(pd.read_csv(categories_filepath))
    return pd.concat([messages,categories],join="inner", axis=1)


def clean_data(df: pd.DataFrame)->pd.DataFrame:
    '''
    Function to clean the data
    Args: df: Data to be cleaned 
    Return: df: Cleaned data
    '''
    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filename: str)->None:
    '''
    Function to save the database
    Args: df: The DataFrame to save, database_filename: Database path (where it will be saved)
    '''
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()