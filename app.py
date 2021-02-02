from flask import Flask,render_template,request,url_for
from scipy import stats
#EDA Packages
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)
cols = ['gender', 'caste', 'mathematics_marks','english_marks', 'science_marks', 'science_teacher','languages_teacher','guardian', 'internet']
with open('model1.pkl', 'rb') as file:  
    model1 = pickle.load(file)
with open('model3.pkl', 'rb') as file:  
    model3 = pickle.load(file)
with open('model4.pkl', 'rb') as file:  
    model4 = pickle.load(file)
with open('model5.pkl', 'rb') as file:  
    model5 = pickle.load(file)

@app.route("/")
def index():
	return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():

    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
	int_features[2] = int_features[2]/100
	int_features[3] = int_features[3]/100
	int_features[4] = int_features[4]/100
        final = np.array(int_features)
        data_unseen = pd.DataFrame([final], columns = cols)
        prediction = model1.predict(data_unseen)
        prediction1 = int(prediction[0])
        prediction = model3.predict(data_unseen)
        prediction3 = int(prediction[0])
        prediction = model4.predict(data_unseen)
        prediction4 = int(prediction[0])
        prediction = model5.predict(data_unseen)
        prediction5 = int(prediction[0])
        pred = stats.mode([prediction1,prediction3,prediction4,prediction5])[0][0]
        if pred==0:
            prediction = "NOT DROP"
        elif pred==1:
            prediction = "DROP"
    return render_template('index.html',pred='STUDENT WILL  {}'.format(prediction))
if __name__ == '__main__':
	app.run(debug=True)
