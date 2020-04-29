import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from typing import Tuple, List
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath: str)->Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Function Load data from database into dataframe
    Args:database_filepath: Database path.
    Return: X: Disaster messages, y: categories, categories: Disaster category names.
    """
    #Load Database
    
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response",engine) 
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)
    categories = y.columns.values
     
    return X, y, categories

def tokenize(text: str)->List[str] :
    '''
    Function for tokenizing string
    Args: string text (Disaster message)
    Returns: tokens list
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
   
    return [lemmatizer.lemmatize(x).lower().strip() for x in tokens]
def build_model()->GridSearchCV:
    '''
    Function to build pipeline and GridSearchCV
    Return: Model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())])
    parameters = {
        'clf__min_samples_split': [5,10, 15],
        'clf__n_estimators': [50, 100, 150]}
    cv = GridSearchCV(pipeline, param_grid=parameters,
                      scoring='accuracy',verbose= 1,n_jobs =-1)

    return cv

def evaluate_modell(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: List)->None:
    ''' 
    Function evaluate model by printing classification report
    Input: model, X_test, Y_test, category_names
    Output: classification report
    '''
    Y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        print(category_names[i])
        print("\tAccuracy: {:.4f}\t\t% Precision: {:.4f}\t\t% Recall: {:.4f}\t\t% F1_score: {:.4f}".format(
            accuracy_score(Y_test[:, i], Y_pred[:, i]),
            precision_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            recall_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')
        ))

def save_model(model: GridSearchCV, model_filepath: str)-> None:
    '''
    Function  Saves the model into a pickle file 
    Args: Model, filepath
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()