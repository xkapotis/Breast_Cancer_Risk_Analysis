import streamlit as st

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
    ('0', '1', '2',"3","4","200")
)

if st.button("Submit"):
    st.text(breast + " " + breast_quad)


# st.button("Button")

# if st.button("About"):
#     st.text("clicked!")

# ## input field
# firstname = st.text_input("Enter your firstname")
# if st.button("Submit"):
#     result = firstname.title()
#     st.success(result)

    
