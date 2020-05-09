import streamlit as st

import pandas as pd
import numpy as np
    # Import train_test_split function



st.title("Risk Analysis in Cancer !")




###
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
    ('0', '1', '1',"3","4","5","6")
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

if st.button("Submit"):
    st.text(breast + " " + breast_quad[0])


# st.button("Button")
def run():
    x = int(breast_quad[0])

    y = x + 5

    st.text(y)


def svm():
    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    # atributes = [ 'Class', 'age', 'menopause', 'tumor-size' ,'inv-nodes', 'node-caps', 'deg-malig' ,'breast', 'breast-quad', 'irradiat']
    # missing = [np.nan , None , "" , "?"]
    df = pd.read_csv('./final.csv')
    # st.text(df)
    data = df[["age", "menopause", "tumor-size" , "inv-nodes" ,"node-caps", "deg-malig" , "breast", "breast-quad"]]

    res = df["Class"]
    st.text(res)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, res, test_size=0.5,random_state=30) # 80% training and 20% test

    #Import svm model
    from sklearn import svm

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    result = metrics.accuracy_score(y_test, y_pred)
    # Model Accuracy: how often is the classifier correct?
    st.text(result)
    
# data = svm()


if st.button("About"):
    run()
    svm()

# ## input field
# firstname = st.text_input("Enter your firstname")
# if st.button("Submit"):
#     result = firstname.title()
#     st.success(result)

    
