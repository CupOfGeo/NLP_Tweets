import dash
import dash_core_components as dcc
import dash_html_components as html
from tensorflow import keras
import random
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = keras.models.load_model('DONTPUSH/GRU512trump_model.h5')
model.load_weights('DONTPUSH/GRU512model_weights_saved_long.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 60

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',
             children='Enter a value and press submit')
])


@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    out = ""
    value = str(value)
    seed_text = value
    next_words = random.randint(20,max_sequence_len)
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #print(token_list)
        predicted = model.predict(token_list, verbose=0)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted.argmax():
                output_word = word
                break
        seed_text += " " + output_word
        out += " " + output_word

    #print(seed_text)
    return str(out)



if __name__ == '__main__':
    app.run_server(debug=True)
