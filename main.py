from flask import Flask,render_template,request,url_for
from scipy import stats
#EDA Packages
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
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
def transf(dat):
    atr=[]
    for x in dat['continue_drop']:
        if x=="continue":
            atr.append(0)
        else:
            atr.append(1)
    dat['continue_drop']=atr
    dat['guardian']= label_encoder.fit_transform(dat['guardian'])
    dat['gender']= label_encoder.fit_transform(dat['gender'])
    dat['internet']= label_encoder.fit_transform(dat['internet'])
    dat['caste']= label_encoder.fit_transform(dat['caste'])
    return dat
def mode (A,B,C,D):
    tzero = 0
    tone=0
    n= len(A)
    results=[]
    for i in range(0,n):
        one = 0
        zero = 0
        if(A[i]==0):
            zero = zero+1
        if(A[i]==1):
            one = one +1
        if(B[i]==0):
            zero = zero+1
        if(C[i]==0):
            zero = zero+1
        if(B[i]==1):
            one = one +1
        if(C[i]==1):
            one = one +1
        if(D[i]==0):
            zero = zero+1
        if(D[i]==1):
            one = one +1
        if(zero>one):
            results.append(0)
            tzero=tzero+1
        if(one>zero):
            results.append(1)
            tone=tone+1
        if(one == zero):
            if(tzero>tone):
                results.append(0)
                tzero=tzero+1
            else:
                results.append(1)
                tone=tone+1
    return results
@app.route("/")
def index():
	return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():

    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final = np.array(int_features)
        data_unseen = pd.DataFrame([final], columns = cols)
        data_unseen['mathematics_marks'] = data_unseen['mathematics_marks']/100
        data_unseen['english_marks'] = data_unseen['english_marks']/100
        data_unseen['science_marks'] = data_unseen['science_marks']/100
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
        else:
            prediction = "DROP"
    return render_template('index.html',pred='STUDENT WILL  {}'.format(prediction))
def guess (dataset):
    ids=[]
    A = model1.predict(dataset)
    B = model5.predict(dataset)
    C =model3.predict(dataset)
    D = model4.predict(dataset)
    results = mode(A,B,C,D)
    index = find(results)
    return index
def find(A):
    resul = []
    n = len(A)
    for i in range (0,n):
        if A[i]==1:
            resul.append(i)
    return resul
@app.route("/data",methods=['POST'])
def data():
	if request.method == 'POST':
		f = request.form['csvfile']
		with open(f) as file:
			csvfile = pd.read_csv(file)
		df = transf(csvfile)
		X = df[['gender', 'caste', 'mathematics_marks','english_marks', 'science_marks', 'science_teacher','languages_teacher', 'guardian','internet']]
		adc=guess(X)
		ids =[]
		for i in adc:
			ids.append(df['student_id'][i])
		ids=pd.DataFrame(ids)
		ids=ids.rename(columns={0: "Student_ID"})
#		out = model1.predict(csvfile)
		return render_template('data.html',data =ids.to_html())
if __name__ == '__main__':
	app.run(debug=True)