import streamlit as st
import pandas as pd
import numpy as np

st.title("Application for Risk Analysis")
st.markdown("## Please fill the following fields and press submit to calculate the possibility of breast cancer reappear")


age = st.selectbox(
    'age',
    ('0 "20-29" ', '1 "30-39" ', '2 "40-49" ','3 "50-59" ','4 "60-69" ','5 "70-79" ')
)
# st.text(age)


menopause = st.selectbox(
    'menopause',
    ('0 "premeno" ', '1 "ge40" ', '2 "lt40" ')
)
tumor_size = st.selectbox(
    'tumor-size',
    ('0 "0-4" ', '1 "5-9" ', '2 "10-14" ','3 "15-19" ','4 "20-24" ','5 "25-29" ','6 "30-34" ','7 "35-39" ','8 "40-44" ','9 "45-49" ','10 "50-54" ')
)

inv_nodes = st.selectbox(
    'inv-nodes',
    ('0 "0-2" ', '1 "3-5" ', '2 "6-8" ','3 "9-11" ','4 "12-14" ','5 "15-17" ','6 "24-26" ')
)
node_caps = st.selectbox(
    'node-caps',
    ('0 "no"','1 "yes"')
)
deg_malig = st.selectbox(
    'deg-malig',
    ('1','2','3')
)
breast = st.selectbox(
    'breast',
    ('0 "left" ','1 "right"')
)
breast_quad = st.selectbox(
    'breast-quad',
    ('0 "left_low"', '1 "right_up"', '2 "left-up"','3 "right-low"','4 "central"')
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

    if y_pred == 0:
        result = "Most possible is not to reappear with the possibility of error 25%"
    elif y_pred == 1:
        result = "Most possible is reappear with the possibility of error 25%"

    st.text(result)


if st.button("Submit"):
    svm(age[0], menopause[0], tumor_size[0], inv_nodes[0], node_caps[0], deg_malig[0], breast[0], breast_quad[0])