import streamlit as st
import pickle
import numpy as np

st.header("Predicition to Purchase")

st.write("Demo of kNN Method")

gender = st.text_input(label="Gender")
age = st.number_input(label="Age")
est_sal = st.number_input(label="Estimated Salary")

if gender == 'Male':
    gen = 1
else:
    gen = 0    

feature=np.array([gen,age,est_sal])
ans = np.array(['No, the given customer will not a make a purchase','Yes, the given customer will make a purchase'])

submit = st.button('Submit')

if submit:
    model=pickle.load(open('model3.pkl','rb'))
    final=model.predict([feature])
    final_ans=ans[final[0]]
    print(final_ans)
    st.write(final_ans)