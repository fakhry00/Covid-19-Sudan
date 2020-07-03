import numpy as np
from flask import Flask, request, jsonify, render_template
from kmodes.kmodes import KModes
#from sklearn.svm import SVC
#from sklearn import svm
import pickle

app = Flask(__name__)
kmodemodel = pickle.load(open('kmodemodel.pkl', 'rb'))
#svcmodel = pickle.load(open('svcmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = kmodemodel.predict(final_features)
    #predictionsvc = svcmodel.predict(final_features)

    #output = prediction[0]
    #outputsvc = predictionsvc[0]
    

    if prediction[0] == 0:
        output = 'Negative' 
    else: 
        output = 'Positive' 

    return render_template('index.html', prediction_text='Your Covid-19 Test is {} '.format(output))
    

if __name__ == "__main__":
    app.run(debug=True)