import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import torch
import requests
import torch.nn as nn
import torchmetrics
import torch.optim as optim
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights
import pytorch_lightning as pl
import pickle
import io
from PIL import Image
from torchvision import transforms

def get_lat_long(city_name):
    url = f"https://nominatim.openstreetmap.org/search"
    params = {
        'q': city_name,
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = data[0]['lat']
            lon = data[0]['lon']
            return lat, lon
        else:
            return "City not found."
    else:
        return "Error in fetching data."


def get_temperature(lat, lon, api_key):
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }

    response = requests.get(base_url, params=params)
    
    
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        return temperature, humidity
    else:
        return st.write(f"Could not fetch temperature for this city. Please check the city name.")


#Streamlit APP

# st.title("AgroAI")
st.sidebar.image("agro.png")
prblm = st.sidebar.radio(
    "Select a Feature",
    ["Crop Recommendation", "Crop Health Monitoring", "Smart Irrigation"],
    captions=[
        "Get data-driven recommendations for the most suitable crop to cultivate.",
        "Analyze and assess the current health status of your crops.",
        "Optimize water usage with intelligent irrigation insights.",
    ],
)
# tab1, tab2, tab3 = st.tabs(["___Crop Prediction___", "___Crop Health___", "___Smart Irrigation___"])


## 1. Crop Prediction

if prblm == "Crop Recommendation":

    # Load nested data from file
    @st.cache_data
    def load_data():
        with open("countries+states+cities.json", "r", encoding="utf-8") as f:
            return json.load(f)

    data = load_data()

    # Get list of country names
    country_names = [country["name"] for country in data]
    
# with tab1:
    cropLogistic = joblib.load("LogisticRegression.pkl")
    cropDecisionTree = joblib.load("DecisionTree.pkl")
    cropRandomForest = joblib.load("RandomForest.pkl")

    st.subheader("Crop Recommendation")
    cropModel = st.sidebar.selectbox("Select The Model", ("Decision Tree", "Logistic Regression", "Random Forest"))
    
    # First row: N, P, K horizontally
    col1, col2, col3 = st.columns(3)
    with col1:
        n_quantity = st.text_input("N quantity", "90")
    with col2:
        p_quantity = st.text_input("P quantity", "42")
    with col3:
        k_quantity = st.text_input("K quantity", "43")

    #Country, State and Cities
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_country_name = st.selectbox("Select Country", country_names)
    # Find selected country object
    selected_country = next((c for c in data if c["name"] == selected_country_name), None)
    if selected_country and "states" in selected_country:
        state_names = [state["name"] for state in selected_country["states"]]
        with c2:
            selected_state_name = st.selectbox("Select State", state_names)
        # Find selected state object
        selected_state = next((s for s in selected_country["states"] if s["name"] == selected_state_name), None)
        if selected_state and "cities" in selected_state:
            city_names = [city["name"] for city in selected_state["cities"]]
            with c3:
                selected_city_name = st.selectbox("Select City", city_names)
        else:
            st.warning("No cities found for this state.")
    else:
        st.warning("No states found for this country.")


    # Second row: Temp & Humidity horizontally
    lat, lon = get_lat_long(selected_city_name)
    apikey="871d1f48565319dfaa86fdb716afa123"
    tempr, humid = get_temperature(lat, lon, apikey)

    col4, col5 = st.columns(2)
    with col4:
        temp = st.text_input("Temperature(Auto Fill)", tempr)
    with col5:
        humidity = st.text_input("Humidity(Auto Fill)", humid)

    # Third row: pH and Rainfall horizontally
    col6, col7 = st.columns(2)
    with col6:
        ph = st.text_input("pH", "6.50")
    with col7:
        rainfall = st.text_input("Rainfall", "202.9")


    data1 = [[n_quantity, p_quantity, k_quantity, temp, humidity, ph, rainfall]]
    X1 = pd.DataFrame(data1, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    
    if cropModel == "Logistic Regression":
        model1 = cropLogistic
        y = model1.predict(X1)
    elif cropModel == "Decision Tree":
        model1 = cropDecisionTree
        y = model1.predict(X1)
    elif cropModel == "Random Forest":
        model1 = cropRandomForest
        y = model1.predict(X1)
        
    if st.button("Predict", type="primary"):
        st.success(f"Based on the provided soil and environmental conditions, the most suitable crop to grow is: **{y}**.")
        st.info("This recommendation is based on parameters such as nitrogen, phosphorous, potassium levels, temperature, humidity, pH, and rainfall.")


## 2. Crop Health
elif prblm == "Crop Health Monitoring":

# with tab2:
    st.subheader("Crop Health Monitoring")
    

    class FineTuneModel(pl.LightningModule):
        def __init__(self, num_classes=18, model_version="googlenet", denselayer_size=128, dropout=0.5, l_rate=0.0005):
            super(FineTuneModel, self).__init__()
            self.save_hyperparameters()
            self.learning_rate = l_rate

            self.train_acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
            
            # Load GoogLeNet using weights API
            if model_version == "googlenet":
                self.model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
            else:
                self.model = models.__dict__[model_version](pretrained=True)

            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # Replace final classifier
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, denselayer_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(denselayer_size, num_classes)
            )

            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            output = self.model(x)
            if isinstance(output, tuple) or hasattr(output, "_asdict"):  # handle namedtuple
                return output.logits  # or output[0] also works
            return output



        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", acc, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    model = FineTuneModel(
    num_classes=18,
    model_version="googlenet",
    denselayer_size=128,
    dropout=0.5,
    l_rate=0.0005)
    

    # Define the custom unpickler to load the model on CPU
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
            return super().find_class(module, name)

    def cpu_joblib_load(path):
        with open(path, 'rb') as f:
            return CPUUnpickler(f).load()

    # Load your saved model
    state_dict = torch.load("model_weights.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    uploaded_file = st.file_uploader("Choose a image")

    if uploaded_file:
        # Load the image you want to predict
        img_path = uploaded_file
        img = Image.open(img_path).convert("RGB")
        st.image(img_path)

        # Apply the same transformation used in your DataModule
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)  # Make prediction
            predicted_class_idx = output.argmax(dim=1).item()  # Get the predicted class index

        # Get class names from the DataModule
        class_names = ['Bacterialblight rice',
        'BlackPoint wheat',
        'Brownspot rice',
        'FusariumFootRot wheat',
        'HealthyLeaf wheat',
        'Iris yellow virus_onion',
        'LeafBlight wheat',
        'Leafsmut rice',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Stemphylium leaf blight and collectrichum leaf blight onion',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___healthy',
        'WheatBlast',
        'healthy_onion',
        'purple blotch onion']  # Replace with actual class names

        # Map to class label
        predicted_class_name = class_names[predicted_class_idx]
        if "healthy" in predicted_class_name.lower():
            st.success(f"✅ The crop appears to be healthy: **{predicted_class_name.replace('_', ' ').title()}**.")
            # st.balloons()
            st.info("No disease symptoms detected. Continue regular monitoring and care.")
        else:
            st.error(f"⚠️ Disease Detected: **{predicted_class_name.replace('_', ' ').title()}**")
            st.warning("It's recommended to take appropriate preventive or remedial actions. Consider consulting an agricultural expert.")
    else:
        st.write("Please upload a picture of crop")
## 3. Smart Irrigation
elif prblm == "Smart Irrigation":
# with tab3:
    le_crop = joblib.load("le_crop.pkl")
    le_state = joblib.load("le_state.pkl")
    xgb_model = joblib.load("xgboost_irrigation_model.pkl")
    rf_model = joblib.load("random_forest_irrigation_model.pkl")

    st.subheader("Smart Irrigation")
    irrModel = st.sidebar.selectbox("Select The Model", ("Random Forest", "XGBoost"))

    
    # First row: N, P, K horizontally
    cl1, cl2, cl3 = st.columns(3)
    with cl1:
        Soil_Moisture = st.text_input("Soil Moisture", "0.0629")
    with cl2:
        Tmax_value = st.text_input("Tmax_value", "39.29")
    with cl3:
        Tmin_value = st.text_input("Tmin_value", "23.033")

    # Second row: Temp & Humidity horizontally
    cl4, cl5, cl6 = st.columns(3)
    with cl4:
        Rainfall_value = st.text_input("Rainfall_value", "6.89")
    with cl5:
        Rainfall_in_last_4_months = st.text_input("Rainfall_in_last_4_months", "10.38")
    with cl6:
        Month = st.text_input("Month", "04")

    # Third row: pH and Rainfall horizontally
    cl7, cl8 = st.columns(2)
    with cl7:
        State = st.selectbox("State",
                             ('Rajasthan','Andaman & Nicobar', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
                                'Chandigarh', 'Chhattishgarh', 'Daman and Diu and Dadra and Nagar Haveli',
                                'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir',
                                'Jharkhand', 'Karnataka', 'Kerala', 'Ladakh', 'Lakshadweep', 'Madhya Pradesh',
                                'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
                                'Puducherry', 'Punjab', 'Sikkim', 'Tamilnadu', 'Telengana',
                                'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
                                ))

    with cl8:
        Crop = st.selectbox("Crop",
                             ( 'Tomato','Wheat', 'Rice', 'Potato'))

    
    State_Code = le_state.transform([State])
    Crop_Code = le_crop.transform([Crop])



    data3 = [[Soil_Moisture, Tmax_value, Tmin_value, Rainfall_value, Rainfall_in_last_4_months, Month, State_Code, Crop_Code]]
    X3 = pd.DataFrame(data3, columns=['Rainfall_value', 'Tmax_value', 'Tmin_value', 'Soil Moisture',
            'rainfall_in_last_4_months', 'Month', 'State_Code', 'Crop_Code'])
    
    if irrModel == "Random Forest":
        output = rf_model.predict(X3)
    elif irrModel == "XGBoost":
        X3["Rainfall_value"] = pd.to_numeric(X3["Rainfall_value"], errors="coerce")
        X3["Tmax_value"] = pd.to_numeric(X3["Tmax_value"], errors="coerce")
        X3["Tmin_value"] = pd.to_numeric(X3["Tmin_value"], errors="coerce")
        X3["Soil Moisture"] = pd.to_numeric(X3["Soil Moisture"], errors="coerce")
        X3["rainfall_in_last_4_months"] = pd.to_numeric(X3["rainfall_in_last_4_months"], errors="coerce")
        X3["Month"] = pd.to_numeric(X3["Month"], errors="coerce")
        X3["State_Code"] = pd.to_numeric(X3["State_Code"], errors="coerce")
        X3["Crop_Code"] = pd.to_numeric(X3["Crop_Code"], errors="coerce")
        output = xgb_model.predict(X3)
    # output = rf_model.predict(X3)

    
    if st.button("Submit", type="primary"):
        if output == [1]:
            st.success("✅ Conditions are suitable for irrigation.")
            st.info("You can proceed with watering. Ensure proper timing and method to maximize efficiency.")
        else:
            st.error("⚠️ Conditions are not suitable for irrigation.")
            st.warning("Avoid irrigation at this time to prevent water stress or wastage. Monitor conditions regularly.")
