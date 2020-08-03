from flask import Flask,request,render_template
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
MAX_SEQUENCE_LENGTH = 300
from keras.preprocessing.sequence import pad_sequences
global model, graph
import tensorflow as tf
graph = tf.get_default_graph()

model=load_model('Sentiment_model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/login',methods =['POST'])
def login():
     text= request.form["a"]
     with graph.as_default():
         x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)
         score = model.predict([x_test])[0]
         if(score <0.4):
             label = "0"
         if(score >=0.4):
             label = "1"
         return {"label" : label,"score": float(score)}
     return render_template('index1.html')
   
    
   

if __name__ =='__main__':
    app.run(debug =True)
