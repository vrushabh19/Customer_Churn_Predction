from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

df = pd.read_csv('clean_data.csv')

model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = StandardScaler()

@app.route('/', methods = ['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['Gender']
        seniorcitizen = int(request.form['SeniorCitizen'])
        partner = request.form['Partner']
        dependents = request.form['Dependents']
        tenure = request.form['Tenure']
        phoneservice = request.form['PhoneService']
        multiplelines = request.form['MultipleLines']
        internetservice = request.form['InternetService']
        onlinesecurity = request.form['OnlineSecurity']
        onlinebackup = request.form['OnlineBackup']
        deviceprotection = request.form['DeviceProtection']
        techsupport = request.form['TechSupport']
        streamingtv = request.form['StreamingTV']
        streamingmovies = request.form['StreamingMovies']
        contract = request.form['Contract']
        paperlessbilling = request.form['PaperlessBilling']
        paymentmethod = request.form['PaymentMethod']
        monthlycharges = float(request.form['MonthlyCharges'])
        totalcharges = float(request.form['TotalCharges'])
        
        record = [[gender,seniorcitizen,partner,dependents,tenure,phoneservice,multiplelines,internetservice,
        onlinesecurity,onlinebackup,deviceprotection,techsupport,streamingtv,streamingmovies,
        contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges]]   

        new_record = pd.DataFrame(record, columns = df.columns)

        df1 = pd.concat([df, new_record], ignore_index=True)

        col = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
               'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
               'Contract','PaperlessBilling','PaymentMethod']

        df1[['MonthlyCharges','TotalCharges']] = scaler.fit_transform(df1[['MonthlyCharges', 'TotalCharges']])

        df2 = pd.get_dummies(df1[col], drop_first=True)

        data = pd.concat([df1[['SeniorCitizen',	'tenure' ,'MonthlyCharges','TotalCharges']], df2], axis=1)

        output = model.predict(data.tail(1))
        probablity = model.predict_proba(data.tail(1))[:,1][0]

        if output[0] == 0:
            text = 'The customer is loyal to company & confidence is {}%'.format(probablity.round(2)*100)
        else:
            text = 'The customer is likely to churn & confidence is {}%'.format(probablity.round(2)*100)

        # return render_template('index.html', prediction_text = 'The output is {}'.format(output))
        return render_template('index.html', prediction_text = text)
    
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)