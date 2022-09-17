import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    args:messages data csv
        catergories csv
    return merged dataframe
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    ##merge dataframes
    df = pd.merge(messages,categories,on='id')
    
    return df
    
def convert_to_binary(x):
    '''
    info:   cleans each cell in category removes string
            returns 1 or 0
    arg:    cell content (string)
    return: 0 or 1 (int)
    '''
    x= int(x[-1])
    if x >=1:
        x=1
    else:
        x=0
    return x



def clean_data(df):
    '''
    args: merged uncleaned dataframe
    return: cleaned dataframe
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat =';',expand=True)
    row = categories.loc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2]) 
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:convert_to_binary(x))

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")
    # drop the original categories column from `df`
    df = df.drop('categories',axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    frames = [df,categories]  # List of your dataframes
    df= pd.concat(frames,axis=1)
    
    # drop duplicates
    print(f'duplicate check == {df.duplicated(keep="first").sum()}')
    df.drop(df[df.duplicated(keep="first")].index, inplace=True)
    print(f'checking dupliates are equal to 0 after removing duplicates{df.duplicated(keep="first").sum()}')

    
    return df
    


def save_data(df, database_filename):
    
    print(df.columns)
    engine = create_engine(f'sqlite:///{database_filename}')
    try:
        df.to_sql('DisasterResponsedb', engine, index=False)
    except:
        engine.execute(f"DROP TABLE DisasterResponsedb")
        df.to_sql('DisasterResponsedb', engine, index=False) 


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