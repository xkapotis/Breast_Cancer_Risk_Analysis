import streamlit as st
import pandas as pd
import numpy as np

st.title("Risk Analysis in Cancer !")
st.markdown("## Please fill the infos and then press submit !")


age = st.selectbox(
    'Age',
    ('0', '1', '2',"3","4","5")
)
# st.text(age)


menopause = st.selectbox(
    'menopause',
    ('0', '1', '2')
)
tumor_size = st.selectbox(
    'tumor-size',
    ('0', '1', '2',"3","4","5","6","7","8","9","10")
)

inv_nodes = st.selectbox(
    'inv-nodes',
    ('0', '1', '2',"3","4","5","6")
)
node_caps = st.selectbox(
    'node-caps',
    ("0","1","200")
)
deg_malig = st.selectbox(
    'deg-malig',
    ("1","2","3")
)
breast = st.selectbox(
    'breast',
    ("0","1")
)
breast_quad = st.selectbox(
    'breast-quad',
    ('0 "dsdsd"', '1', '2',"3","4","200")
)



def svm(age, menopause, tumor_size , inv_nodes ,node_caps, deg_malig , breast,breast_quad):
    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('./data/final.csv')
    # st.text(df)
    data = df[["age", "menopause", "tumor-size" , "inv-nodes" ,"node-caps", "deg-malig" , "breast", "breast-quad"]]
    # st.text(data)
    res = df["Class"]
    # st.text(res)

    data_for_prediction = [[age, menopause, tumor_size , inv_nodes ,node_caps, deg_malig , breast,breast_quad]]
    # st.text(data_for_prediction)
    df_for_prediction = pd.DataFrame(data_for_prediction, columns = ["age", "menopause", "tumor-size" , "inv-nodes" ,"node-caps", "deg-malig" , "breast", "breast-quad"])

    X_train =  data
    X_test = df_for_prediction
    y_train = res


    #Import svm model
    from sklearn import svm

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    st.text(y_pred)


if st.button("Submit"):
    svm(age, menopause, tumor_size , inv_nodes ,node_caps, deg_malig , breast,breast_quad[0])


    
