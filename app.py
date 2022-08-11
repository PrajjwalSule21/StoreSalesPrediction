import numpy as np
import pickle
import pandas as pd
import streamlit as st


Modelpath = 'f:/Ineuron_Internship/StoreSalesPrediction/BestModel/'
pickle_in = open(Modelpath + "Random_forest_regressor.pkl","rb")
regressor = pickle.load(pickle_in)



def predict_sales(Item_Weight, Outlet_Size, Item_Fat_Content, Outlet_Type, Item_Type, Item_MRP):

    data = [Item_Weight, Outlet_Size, Item_Fat_Content, Outlet_Type, Item_Type, Item_MRP]
    feature_value = [np.array(data)]
    features_names = ['Item_Weight', 'Outlet_Size', 'Item_Fat_Content', 'Outlet_Type', 'Item_Type', 'Item_MRP']

    df = pd.DataFrame(feature_value, columns=features_names)

    my_predict = regressor.predict(df)


    return f"Predicted sale of Item from the particluar Outlet is : {np.round(my_predict,2)[0]} Rupees"





def main():
    st.title("Store Sales Prediction")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Sales Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Item_Weight = st.number_input("Enter the Weight of Item in KG", min_value=1)
    if Item_Weight <= 0 or Item_Weight >= 1000:
        return st.error('Provide valid Weight of Item')

    Outlet_Size = st.selectbox("Select the Size of Outlet", ['Small', 'Medium', 'Large'])
    if Outlet_Size == 'Small':
        Outlet_Size = 0.0
    elif Outlet_Size == 'Medium':
        Outlet_Size = 1.0
    else:
        Outlet_Size = 2.0

    Item_Fat_Content = st.selectbox("Select the Fat Content of Item", ['Low Fat', 'Regular'])
    if Item_Fat_Content == 'Low Fat':
        Item_Fat_Content = 0.0
    else:
        Item_Fat_Content = 1.0



    Outlet_Type = st.selectbox("Select the type of Outlet", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
    if Outlet_Type == 'Grocery Store':
        Outlet_Type = 0.0
    elif Outlet_Type == 'Supermarket Type1':
        Outlet_Type = 1.0
    elif Outlet_Type == 'Supermarket Type2':
        Outlet_Type = 2.0
    else:
        Outlet_Type = 3.0


    Item_Type = st.selectbox('Select the type of Item ',['Others', 'Seafood', 'Breakfast', 'Starchy Foods', 'Hard Drinks', 'Breads', 'Meat', 'Soft Drinks', 'Health and Hygiene', 'Baking Goods', 'Canned', 'Dairy', 'Frozen Foods', 'Household', 'Snack Foods', 'Fruits and Vegetables'])
    if Item_Type == 'Others':
        Item_Type = 0.0
    elif Item_Type == 'Seafood':
        Item_Type = 1.0
    elif Item_Type == 'Breakfast':
        Item_Type = 2.0
    elif Item_Type == 'Starchy Foods':
        Item_Type = 3.0
    elif Item_Type == 'Hard Drinks':
        Item_Type = 4.0
    elif Item_Type == 'Breads':
        Item_Type = 5.0
    elif Item_Type == 'Meat':
        Item_Type = 6.0
    elif Item_Type == 'Soft Drinks':
        Item_Type = 7.0
    elif Item_Type == 'Health and Hygiene':
        Item_Type = 8.0
    elif Item_Type == 'Baking Goods':
        Item_Type = 9.0
    elif Item_Type == 'Canned':
        Item_Type = 10.0
    elif Item_Type == 'Dairy':
        Item_Type = 11.0
    elif Item_Type == 'Frozen Foods':
        Item_Type = 12.0
    elif Item_Type == 'Household':
        Item_Type = 13.0
    elif Item_Type == 'Snack Foods':
        Item_Type = 14.0
    else:
        Item_Type = 15.0



    Item_MRP = st.number_input('Give the MRP of Item', min_value=1)
    if Item_MRP <= 0:
        return st.error('Provide valid MRP of Item')


    result = ""
    if st.button("Predict"):
        result = predict_sales(Item_Weight, Outlet_Size, Item_Fat_Content, Outlet_Type, Item_Type,  Item_MRP)

    st.success(result)


if __name__ == '__main__':
    main()



