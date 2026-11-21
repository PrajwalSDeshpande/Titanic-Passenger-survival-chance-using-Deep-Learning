import streamlit  as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
st.title('Passenger survival chance in Titanic journey')
pclass=st.slider('Enter the Passenger Class for user',1,3)
sex=st.selectbox('Enter the Passenger Gender',['male','female'])
sibsps=st.slider('Enter the total number of Sibling and Spouse',1,8)
parch=st.slider('Enter the total number of Parent and Child',0,6)
fare=st.number_input("Enter the fare of Passenger")
embarked=st.selectbox('Enter Passenger station from where they started their journey',['Southampton','Chebourg','Queenstown'])
data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsps,'Parch':parch,'Fare':fare,'Embarked':embarked}])
model=load_model('model.h5')
with open('Label_Encoder.pkl','rb') as file:
    label=pickle.load(file)
with open('OneHot_Encoder.pkl','rb') as file:
    onehot=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)
data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])
y=model.predict(data)
y=y[0][0]
def chance(y):
    if y>0.5:
        return "The passenger will survive journey"
    else:
        return "The passenger will not survive journey"
if st.button('Predict survival chance'):
    st.write('probability of Survival of the Passenger is:',y)
    st.write(chance(y))