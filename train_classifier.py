# # # Notebook for model training

#-----------#
# # Index # # 
#-----------#
# 1. Load libraries
# 2. Method for Load data
# 3. Method for tokenize messages
# 4. Method for build model
# 5. Method for evaluate model
# 6. Method for save model
# 7. Pull All methods in one 

#---------------------------#
# # # 1. Load libraries # # #
#---------------------------#
# Basic libraries
import sys
import pandas as pd
import numpy as np
import pickle
import joblib
import re
from sqlalchemy import create_engine
import time

# Necesary NLTK statments
import nltk
nltk.download(['punkt', 'wordnet'])
# from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# libraries for models with sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score

#----------------------#
# # # 2. Load data # # #
#----------------------#

def load_data(database_filepath):
    '''
    This method allow load the cleaned data with process_data.py code
    Return a description of the loaded data and the two databases
    
    Args:
        * database_filepath: Path of the cleaned database (.db) file.
        Example: data/cleaned_fullbase_messages.db

    '''
    # Extrat filename for load sqllite table
    filename = re.sub(r'(.*/)(.*.db$)', r'\2', database_filepath)
    filename = re.sub(r'(.*)(\.db)', r'\1', filename)
    print(f"Loading {filename} data base ...")

    # Creating engine and loading data sqlite base
    database_engine = "sqlite:///" + database_filepath   # database_filepath = "data/cleaned_fullbase_messages.db"
    engine = create_engine(database_engine)
    df = pd.read_sql_table(database_filepath, engine)
    
    print(f"Dataset loaded has {df.shape[0]} rows y {df.shape[1]} columns")
    
    # Segmenting bases. Predictor variable and dependent variables 
    X = df[["message"]]
    Y = df.drop(columns=['id', 'message', 'original', 'genre']) 
    category_names = Y.columns

    print("Data loading process finished 7:D")

    print("X - NAMES")
    print(X.columns)
    print("\n")
    print("Y - NAMES")
    print(Y.columns)
    print("\n")
    print("Categories")
    print(category_names)

    return X, Y, category_names

# load_data(database_filepath = "data/cleaned_fullbase_messages.db")

#-----------------------------------------#
# # # 3. Method for tokenize messages # # #
#-----------------------------------------#
def tokenize(text):
    '''
    Method that allow tokenize text that are be used by train the model
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#-----------------------------------#
# # # 4. Method for build model # # #
#-----------------------------------#
def build_model():
    '''
    Method for build the model according to the specifications
        * Building a Pipeline
        * Creating a Grid parameters
        * Create grid search object
    Return a model that be fitted in a next step
    '''

    print("Building a Pipeline")
    pipeline = Pipeline([("vect", CountVectorizer(tokenizer=tokenize)),  # tokenizer=tokenize
                         ("tfidf", TfidfTransformer()),
                         ("clf", MultiOutputClassifier(LogisticRegression()))
                        ])

    print("Creating a Grid parameters")
    parameters = {
        'clf__estimator__C': [0.01, 0.1, 1],# 10, 50, 100],
        'clf__estimator__penalty': ['l2'], # 'l1', 'elasticnet', 'None'],
        'clf__estimator__solver': ['lbfgs','liblinear'],# 'sag', 'newton-cg', 'saga'],
        'clf__estimator__max_iter': [100,300]#,1000]
    }
    
    print("create grid search object")
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


#--------------------------------------#
# # # 5. Method for evaluate model # # #
#--------------------------------------#
def evaluate_model(model, X_test, y_test, category_names):
    '''
    Method that create a evaluation model by two levels: 
    a general level and by category level
    
        * Using the model for estimate in the test base
        * Creating a sub-method to evaluate the model in general
        * Calculate Results by category
    
        Return_ Two databases with results in general and by category
    '''

    # Using the model for estimate in the test base
    y_pred = model.predict(X_test)
    y_pred_2 = pd.DataFrame(y_pred) # Convert to pandas df
    y_pred_2.columns = y_test.columns

    #  Method for general results of the model
    def display_results(y_test, y_pred):
        labels = np.unique(y_pred)
        f1_value = f1_score(y_test, y_pred, average='weighted')
        precision_value = precision_score(y_test, y_pred, average='weighted')
        recall_value = recall_score(y_test, y_pred, average='weighted')
        accuracy_value = (y_pred == y_test).mean()
        
        metrics_df = pd.DataFrame(columns=['Accuracy','f1_score', 'precision', 'recall'],
                                data=[[accuracy_value, f1_value, precision_value, recall_value]]).T.reset_index()
        metrics_df.columns = ['METRIC','SCORE']
        return metrics_df
    
    mlb = MultiLabelBinarizer()
    y_test_binarized = mlb.fit_transform(y_test.values)
    y_pred_binarized = mlb.transform(y_pred)
    general_results = display_results(y_test_binarized, y_pred_binarized)

    
    # # Results by each category
    category_results = pd.DataFrame(columns=['Category', 'f1_score', 'precision', 'recall'])

    y_test_aa = y_test.reset_index(drop=True)
    y_pred_aa = y_pred_2.reset_index(drop=True)

    for cat in category_names:
        f1_value = f1_score(y_test_aa[cat], y_pred_aa[cat], average='weighted')
        precision_value = precision_score(y_test_aa[cat], y_pred_aa[cat], average='weighted')
        recall_value = recall_score(y_test_aa[cat], y_pred_aa[cat], average='weighted')
        accuracy_value = (y_test_aa[cat] == y_pred_aa[cat]).mean()

        row_cat = pd.DataFrame({
            'Category': [cat], 
            'f1_score': [f1_value], 
            'precision': [precision_value], 
            'recall': [recall_value]
            })
        
        # Concat the results of each category
        category_results = pd.concat([category_results, row_cat], ignore_index=True)

    return general_results, category_results

#----------------------------------#
# # # 6. Method for save model # # #
#----------------------------------#
def save_model(model, model_filepath):
    '''
    Method that save fitted model in a pikkle file
    Args:
        * model: fitted model that will be save
        * model_filepath : path of the pikkle file. This str has to end with -.pkl 

    '''
    print(f"Saving {model} in a pikkle file ...")
    joblib.dump(model, model_filepath)

    print(f"Fitted model are be saved in {model_filepath}")
    

#------------------------------------#
# # # 7. Pull All methods in one # # #
#------------------------------------#
def main(database_filepath, model_filepath):
    '''
    This method allow to execute all process of the train the multi clasification model following next steps:
        * Load data
        * Build and train model
        * Evaluate model
        * Save the model
    
    Args of the method:
        * database_filepath : Path of the cleaned base for train the clasification model  
        * model_filepath : Path where you want to save the model in .pkl format. Example: models/name_of_model.pkl
    '''

    start_time = time.time()
    # Load data
    print('Loading data ... DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("\n")
    
    # Build and train model
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train['message'], Y_train)
    print("\n")
    print("\n")
    print("Better combination of hiper parameters: ", model.best_params_)

    print("\n")
    # Evaluate model
    print('Evaluating model...')
    rend_gnrl, rend_cat = evaluate_model(model, X_test['message'], Y_test, category_names)
    print("\n")
    print("---rendimiento general---")
    print(rend_gnrl)
    print("\n")
    print("---rendimiento por categoria---")
    print(rend_cat)
    print("\n")
    
    # Save the model
    print("Saving model in a pikkle format ...")
    save_model(model, model_filepath)
    print('Trained model saved!')

    end_time = time.time()  
    execution_time = end_time - start_time
    print(f"Execute time: {execution_time} sec.")

#-----------------#
# # # Pruebas # # #
#-----------------#
if __name__ == '__main__':
    main(database_filepath = "data/cleaned_fullbase_messages.db", 
         model_filepath = 'models/best_trained_model.pkl')
