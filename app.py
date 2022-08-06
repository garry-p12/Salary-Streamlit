import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression 

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title('Employee Salary prediction')
    st.header('Will you get that raise? Where do you stand?')

with dataset:
    data = pd.read_csv('employees (1) (1) (1) (2).csv')
    data['last_evaluation'].fillna('0',inplace=True)
    data['department'].fillna('support',inplace=True)
    data['satisfaction'].fillna(0,inplace=True)
    data['tenure'].fillna(0,inplace=True)
    le = LabelEncoder()
    label = le.fit_transform(data['EmployeeName'])
    label = le.fit_transform(data['Agency'])
    label = le.fit_transform(data['fname'])
    label = le.fit_transform(data['lname'])
    data.drop("EmployeeName", axis=1, inplace=True)
    data.drop("Agency", axis=1, inplace=True)
    data.drop("fname", axis=1, inplace=True)
    data.drop("lname", axis=1, inplace=True)
    data["EmployeeName"] = label
    data["Agency"] = label
    data["fname"] = label
    data["lname"] = label
    data['status']=data['status'].map({'Employed':1,'Left':0})
    data['salary']=data['salary'].map({'low':0,'medium':1,'high':2})
    data['department']=data['department'].map({'product':0,'sales':1,'support':2,'temp':3,'IT':4,'admin':5,'engineering':6,'finance':7,'information_technology':8,'management':9,'marketing':10,'procurement':11})
    #tenure = tenure.to_frame(name='tenure')  
    data = data.drop(['Agency'],axis=1)
    data = data.drop(['EmployeeName'],axis=1)
    data = data.drop(['fname'],axis=1)
    data = data.drop(['lname'],axis=1)
    data = data.drop(['last_evaluation'],axis=1)
    data = data.drop(['avg_monthly_hrs'],axis=1)
    ten = pd.DataFrame(data['tenure'].value_counts().head(50))
    n_projects = pd.DataFrame(data['n_projects'].value_counts().head(50))
    department = pd.DataFrame(data['department'].value_counts().head(50))
    status = pd.DataFrame(data['status'].value_counts().head(50))
    salary = pd.DataFrame(data['salary'].value_counts().head(50))
    st.metric(label='Tenure',value='1',delta='1')
    st.metric(label='Number of Projects',value='0',delta='1')
    st.area_chart(n_projects)
    st.area_chart(department)
    st.area_chart(status)    
    st.area_chart(salary)


target = np.array(data.drop(['salary1'],1))
features = np.array(data['salary1'])


with model_training:
    sel_col,disp_col = st.columns(2)
    x_train , x_test , y_train , y_test = train_test_split(target,features,test_size=0.25,random_state=42)
    gb = GradientBoostingRegressor
    regr = gb()
    regr.fit(x_train,y_train)
    y_pred = regr.predict(x_test) 	
    def lr_param_selector(tenure,rating,projects,salary,status,satisfaction,age,department):
        prediction = regr.predict([[tenure, rating, n_projects, salary,status,satisfaction,age,department]])
        return prediction
         


    tenure = st.number_input('Tenure')
    rating = st.number_input('Rating') 
    n_projects = st.number_input("Number of projects completed") 
    salary = st.number_input("Salary level(0 for low,1 for medium,2 for high)")
    status = st.number_input('(0 for left, 1 for employed)')
    satisfaction = st.number_input('Satisfaction(between 0 and 1)')
    age = st.number_input('Age')
    department = st.number_input('Department(0-6)')
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"):
        result = lr_param_selector(tenure, rating, n_projects, salary, status,satisfaction,age,department) 
    st.success('The predicted salary is{}'.format(result))
    
