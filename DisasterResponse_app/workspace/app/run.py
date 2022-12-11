import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


    

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponsedb', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
   
    #genre and search and rescue
    ss1 = df[df['search_and_rescue']==1].groupby('genre').count()['message']
    ss0 = df[df['search_and_rescue']==0].groupby('genre').count()['message']
    genre_names = list(ss1.index)

    
    
   #genre and search and rescue
    aid_rel1 = df[df['aid_related']==1].groupby('genre').count()['message']
    aid_rel0 = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_names = list(aid_rel1.index)
    
    
    # word cloud data
    message_uniquecounts = df['message'].unique().tolist()
    message_list = [len(tokenize(message)) for message in message_uniquecounts]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #Example graphs
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        #search and rescue
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=ss1,
                    name = 'search and rescue'

                ),
                Bar(
                    x=genre_names,
                    y= ss0,
                    name = 'Not search and rescue'
                )
            ],

            'layout': {
                'title': 'Distribution of message by genre and search and rescue class ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
       ##aid related
       {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_rel1,
                    name = 'aid related'

                ),
                Bar(
                    x=genre_names,
                    y= aid_rel0,
                    name = 'Not aid related'
                )
            ],

            'layout': {
                'title': 'Distribution of message by genre and aid related messages ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
       ##messages
       {
            'data': [
                Histogram(
                    x=message_list,
                )
            ],

            'layout': {
                'title': 'Messages Histogram',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "word count in message",
                    'range': [0,100]
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    
    print(query)
    
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))


    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()