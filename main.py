import streamlit as st 
import pandas as pd 
import csv 
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import apriori, association_rules

header = st.container()
dataset= st.container()
feature = st.container()
model_training = st.container()
customer_input = st.container()

with header: 
    st.title('Welcome to the Customer Choice Prediction Model')


with dataset: 
    st.header('Dataset')


    dataset = []
    with open('data/DataSetA.csv') as file: 
        reader = csv.reader(file, delimiter=',')
        for row in reader: 
            dataset += [row]
    
    te = TransactionEncoder()
    x = te.fit_transform(dataset)

    df = pd.DataFrame(x, columns = te.columns_)
    df.columns = df.columns.str.lower()
    df = df.drop([''],axis = 1)

    st.write(df.head())


with model_training: 
    
    freq_itemset = apriori(df, min_support = 0.042, use_colnames=True)
    rules = association_rules(freq_itemset, metric = 'confidence', min_threshold = 0.46)


with customer_input: 
    st.header("Customers Choice: ")

    user_input= st.selectbox(
            'Choose Product',
            ['','bread', 'butter', 'cheese', 'coffee powder', 'ghee', 'lassi','milk','panner','sugar','sweet','tea powder','yougurt']
        )
    user_input = user_input.lower()
    st.write(rules[rules['antecedents'] == {user_input}].iloc[:,[0,1,5]])
