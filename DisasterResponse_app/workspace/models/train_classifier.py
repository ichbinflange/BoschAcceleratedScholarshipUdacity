import sys
# import libraries
import pandas as pd
import numpy as np
#database
from sqlalchemy import create_engine
#nltk
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
nltk.download('punkt')
##sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
##pickle
import pickle

def load_data(database_filepath):
    '''
    args: path to database
    return X(messages),Y(categories),Y.column (category column names)
    '''
    cnx = create_engine(f'sqlite:///{database_filepath}').connect()
    df = pd.read_sql_table(
        'DisasterResponsedb', cnx)
    df = df.reset_index()
    
    print('df loaded !')
    print(df.columns)
    #get X,Y data
    X =  df.message.values
    Y=  df.iloc[:,4:]
    Y=(Y.astype('str'))
    #category_names = 
       
    return X,Y,Y.columns


def tokenize(text):
    '''
    args: texts
    return tokenized and normalized texts
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    review = re.sub('[^a-zA-Z0-9]', ' ', text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    review = word_tokenize(review)
    
    return text


def build_model():
    '''
    info: converts texts to digits creates tfidf and builds model pipeline 
    args: None
    return pipeline model classifier
    '''
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(RandomForestClassifier())
                    )])
    #parameters = {
     #   'vect__max_df':[0.5,0.9],
      #  'clf__estimator__n_estimators': [20, 40]
    #}


    #cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=4)
   
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for index, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], y_pred[:,index]))

    avg = (y_pred == Y_test).mean().mean()
    print("Accuracy Overall:\n", avg)


def save_model(model, model_filepath):
    try:
        pickle.dump(model, open(model_filepath, 'wb'))
        print('model saved')
    except:
       print('error !')
       print('model not saved')


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