import streamlit as st
import numpy as np
import pandas as pd
import joblib

## load all model saved 
model=joblib.load('best_model_xg.pkl')
sc=joblib.load('scaler.pkl')
pt_engine=joblib.load('pt_engine.pkl')
pt_max_power=joblib.load('pt_max_power.pkl')
training_column=joblib.load('model_columns.pkl')

st.set_page_config(page_title= 'Welcome to Car selling center',layout='centered')
st.title('car price prediction')
st.write('Enter car details below')
car_name=st.text_input('Car Name','Hyundai i20')
age=st.number_input('Car Age',min_value=0,value=5,max_value=20)
fuel=st.selectbox('Fuel Type',['Petrol','Diesel','CNG','LPG'])
seller_type=st.selectbox('Seller Type',['Individual','Dealer','Trustmark Dealer'])
transmission=st.selectbox('Transmission',['Manual','Automatic'])
owner=st.selectbox('Owner_type',['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'])
mileage=st.number_input('Mileage',min_value=5,value=12)
engine=st.number_input('Engine',min_value=50,value=120)
max_power=st.number_input('Max Power',min_value=50,value=120)
seats=st.number_input('seats',min_value=2,max_value=14,value=5)
km_driven=st.number_input('KM Driven',min_value=100,value=100000)

if st.button('Predict Selling Price'):
    brand=car_name.split()[0]
    def brand_segment(brand):

        luxury = ['Lexus','Volvo','Jaguar','BMW','Land','Audi','Mercedes-Benz']
        mid = ['MG','Jeep','Kia','Isuzu','Force','Toyota','Mitsubishi','Mahindra','Honda','Ford','Volkswagen','Skoda']

        if brand in luxury:
            return 2

        elif brand in mid:
            return 1

        else:
            return 0

    brand_segment_value = brand_segment(brand)
    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }
    owner=owner_map[owner]
    transmission=1 if transmission=='Automatic' else 0
    km_driven=np.log1p(km_driven)
    engine=pt_engine.transform([[engine]])[0][0]
    max_power=pt_max_power.transform([[max_power]])[0][0]
    input_df = pd.DataFrame({

        'brand_segment': [brand_segment_value],

        'age': [age],

        'owner': [owner],

        'transmission': [transmission],

        'mileage': [mileage],

        'engine': [engine],

        'max_power': [max_power],

        'seats': [seats],

        'log_km_Driven': [km_driven],

        'fuel': [fuel],

        'seller_type': [seller_type]

    })

    # =====================
    # ONE HOT ENCODING
    # =====================

    fuel_dummies = pd.get_dummies(
        input_df['fuel'],
        prefix='fuel',
        dtype=int,
        drop_first=False
    )

    seller_dummies = pd.get_dummies(
        input_df['seller_type'],
        prefix='seller',
        dtype=int,
        drop_first=False
    )

    input_df = pd.concat(
        [
            input_df,
            fuel_dummies,
            seller_dummies
        ],
        axis=1
    )

    input_df.drop(
        ['fuel', 'seller_type'],
        axis=1,
        inplace=True
    )

    # =====================
    # ADD MISSING COLUMNS
    # =====================

    for col in training_column:

        if col not in input_df.columns:

            input_df[col] = 0

    # =====================
    # COLUMN ORDER
    # =====================

    input_df = input_df[
        training_column
    ]

    # =====================
    # SCALING
    # =====================

    num_cols = [

        'mileage',

        'engine',

        'max_power',

        'age',

        'log_km_Driven'
    ]
    input_df[num_cols] = pd.DataFrame(
    sc.transform(input_df[num_cols]),
    columns=num_cols,
    index=input_df.index)

    input_df = input_df.astype(float)
    # st.write(input_df)

    # input_df[num_cols] = sc.transform(
    #     input_df[num_cols])
    # st.write(input_df)
    # st.write(training_column)

    # st.write(input_df.columns.tolist())

    # =====================
    # PREDICTION
    # =====================

    prediction = model.predict(input_df)
    # st.write("Raw Prediction:", prediction[0])
    # =====================
    # REVERSE LOG
    # =====================

    # final_price = np.expm1(
    #     prediction[0]
    # )
    st.write("Final Price:", prediction[0])
    # =====================
    # OUTPUT
    # =====================

    st.success(
        f'Predicted Selling Price: ₹ {prediction[0]}'
    )
