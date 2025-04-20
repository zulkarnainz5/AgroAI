import streamlit as st
import json
import requests

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





# Load nested data from file
@st.cache_data
def load_data():
    with open("countries+states+cities.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

# Get list of country names
country_names = [country["name"] for country in data]
selected_country_name = st.selectbox("Select Country", country_names)

# Find selected country object
selected_country = next((c for c in data if c["name"] == selected_country_name), None)

if selected_country and "states" in selected_country:
    state_names = [state["name"] for state in selected_country["states"]]
    selected_state_name = st.selectbox("Select State", state_names)

    # Find selected state object
    selected_state = next((s for s in selected_country["states"] if s["name"] == selected_state_name), None)

    if selected_state and "cities" in selected_state:
        city_names = [city["name"] for city in selected_state["cities"]]
        selected_city_name = st.selectbox("Select City", city_names)

        st.success(f"You selected: {selected_country_name} > {selected_state_name} > {selected_city_name}")
    else:
        st.warning("No cities found for this state.")
else:
    st.warning("No states found for this country.")

lat, lon = get_lat_long(selected_city_name)
st.write(lat, lon)



