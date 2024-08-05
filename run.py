import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Heatmap, Layout, Figure
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

# Please, to write the path of the files write the complete paths for more security
# In the table name write the table name clearly. 
# If isn't run, add parts of the path and try again... 

app = Flask(__name__)

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

# load data
# print("----- Loading Data -----")
table_name = "data/cleaned_fullbase_messages.db" 
database_filepath = "/workspace/home/data/cleaned_fullbase_messages.db"

database_engine = "sqlite:///" + database_filepath
engine = create_engine(database_engine)
df = pd.read_sql_table(table_name, engine)

# load model
print("----- Loading Model -----")
model = joblib.load("/workspace/home/models/best_trained_model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # # # Barplot
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # # # Heatmap
    pivot_table_1 = df.pivot_table(index = 'genre', aggfunc = 'sum')
    numeric_columns = pivot_table_1.select_dtypes(include='number').columns
    normalized_pivot_table_1 = pivot_table_1[numeric_columns].apply(lambda x: x / x.sum())

    # Convertir el DataFrame a una lista de listas para el heatmap
    z_data = normalized_pivot_table_1.values.tolist()
    x_data = normalized_pivot_table_1.columns.tolist()
    y_data = normalized_pivot_table_1.index.tolist()

    text_data = [[f"{value:.2%}" for value in row] for row in normalized_pivot_table_1.values]

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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

        {
        'data': [
            Heatmap(
                z=z_data,
                x=x_data,
                y=y_data,
                text=text_data,
                zmin=0,
                zmax=1,
                colorscale=[[0, 'white'], [1, 'red']],
                colorbar={'title': 'Proporción'}
            )
        ],
        'layout': {
            'title': 'Message Genres Distribution by category',
            'xaxis': {
                'title': 'Category',
                'tickangle': 270
            },
            'yaxis': {
                'title': 'Gender'
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
