# # # Notebook for data cleaning

#-----------#
# # Index # # 
#-----------#
# 1. Load libraries
# 2. Method / function to load the data
# 3. Method to clean the data
# 4. Method to save the cleaned data 
# 5. Pull All methods in one 

#---------------------------#
# # # 1. Load libraries # # #
#---------------------------#
# Basic libraries
import sys
import pandas as pd
import re

# Save cleaned data into a sqlite data base
from sqlalchemy import create_engine

#-----------------------------------------------#
# # # 2. Method / function to load the data # # #
#-----------------------------------------------#
def load_data(messages_filepath, categories_filepath):
    '''
    This method allow load the data:
    * Messages
    * categories
    Return a description of the loaded data and the two databases 
    '''
    print("Loading data ...")

    messages = pd.read_csv(messages_filepath,
                       encoding='latin-1')
    
    categories = pd.read_csv(categories_filepath,
                       encoding='latin-1')

    print(f"Dataset messages has {messages.shape[0]} rows y {messages.shape[1]} columns")
    print(f"Dataset categories has {categories.shape[0]} rows y {categories.shape[1]} columns")
    
    print("Data loading process finished 7:D")

    return messages, categories

#-------------------------------------#
# # # 3. Method to clean the data # # # 
#-------------------------------------#
def clean_data(df):
    '''
    This method allow clean the loaded messages and categories data:
    * Merge data
    * Split categories
    * Set each value of the columns in boolean values (categories data)
    * Identify potencial empty categories
    * Replace categories column in new df
    * Drop duplicated values
    Return a description with the cleaned dataset and the dataset 
    '''
    # Load the data to clean
    messages, categories = df

    # Merge data
    merged_df = pd.merge(messages,
                         categories,
                         on = "id")
    
    # Split all categories
    categories = merged_df['categories'].str.split(';', expand=True)
    categories.columns = [re.sub(r"(.*)(-.*)", r"\1", x) for x in categories.iloc[0]]

    # Set each value of the columns in boolean values
    for column in categories.columns:
        categories[column] = [re.sub(r"(.*-)(\d)", r"\2", x) for x in categories[column]]
        categories[column] = categories[column].astype(int)
    
    # Identify potencial empty categories
    validate_categories = categories.sum(axis = 0).to_frame().reset_index().rename(columns={'index':'category', 0:'count_results'})
    empty_categories = validate_categories[validate_categories['count_results'] == 0]
    categories_2 = categories.drop(empty_categories['category'], axis=1)

    if len(empty_categories) > 0:
       print(f"Dropped empty {len(empty_categories)} categories. \n Not exist observations for train the model for this categories")
       print("Droped categories: ")
       for cat in empty_categories['category']:
          print("\t *", cat)   
    else:
        print("All categories are included!")

    # Replace categories column in new df
    merged_df.drop('categories', axis=1, inplace=True)
    merged_df = pd.concat([merged_df, categories_2], axis = 1)

    # Drop duplicated values
    merged_df.drop_duplicates(inplace=True)

    # Export Cleaned dataset
    print(f"Cleaned dataset has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
    # print(merged_df.head())
    
    return merged_df

#--------------------------------------------#
# # # 4. Method to save the cleaned data # # #
#--------------------------------------------#
def save_data(df, database_filename):
    '''
    This method allow export the base in a sqlite format.
    Please, in database_filename argument write the choosen path that want to save the database.
    '''
    path_future_file = "sqlite:///" + database_filename
    print("File: ", path_future_file)
    engine = create_engine(path_future_file)
    df.to_sql(database_filename, engine, index=False)
    print("Database was successfully exported")

#------------------------------------#
# # # 5. Pull All methods in one # # #
#------------------------------------#
def main(messages_filepath, categories_filepath, database_filename):
    '''
    This method allows you to run the entire data cleansing ETL process using methods 
    that were created earlier.

    * Load data
    * Tranform / Clean Data
    * Export Data

    Args:
    * messages_filepath: Path of the messages data base
    * categories_filepath: Path of the categories data base
    * database_filename: Path of the cleaned database for export whit db extension file. Example: data/wished_table_name.db
    '''

    # Loading Data
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    # Cleaning Data
    cleaned_data = clean_data(df)
    
    print('Saving data...\n    DATABASE: {}'.format(database_filename))
    save_data(cleaned_data, database_filename)
    print('Cleaned data saved to database!')

    pass


if __name__ == '__main__':
    main(messages_filepath = "data/disaster_messages.csv", 
         categories_filepath = "data/disaster_categories.csv", 
         database_filename = "data/cleaned_fullbase_messages.db")

    
    