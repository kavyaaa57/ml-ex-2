import streamlit as st
import pandas as pd
from id3 import DecisionTreeID3
st.markdown("<h1 style='text-align: center;'>Code X</h1>", unsafe_allow_html=True)
st.title("ID3 Decision Tree Classifier")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    target = st.selectbox("Select the target column", data.columns)
    features = st.multiselect("Select feature columns", data.columns)

    if target and features:
        X = data[features].values
        y = data[target].values

        tree = DecisionTreeID3()
        tree.fit(X, y)

        st.write("Decision Tree:", tree.tree)

        if st.checkbox("Make predictions"):
            test_data = st.text_area("Enter test data (comma-separated)", "0,1,0")
            if test_data:
                test_sample = np.array([list(map(int, test_data.split(',')))])
                prediction = tree.predict(test_sample)
                st.write("Prediction:", prediction[0])
