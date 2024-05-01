import streamlit as st
import openpyxl
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import pickle_mixin
import geopy
st.set_page_config(layout="wide")

# Load the trained model from a pickle file
@st.cache_resource()
def load_model():
    with open('xgboost_model_occupancy_rate.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_model2():
    with open('xgboost_model_adr.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model_occupancy = load_model()
model_adr = load_model2()

from PIL import Image

def display_title_with_local_logo():
    # Load your logo image
    logo = Image.open("airnbnb_logo.png")  # Update the path if the logo is in a specific folder
    
    # Create columns for logo and title
    col1, col2 = st.columns([1, 8])
    
    # Display the logo in the first column
    with col1:
        st.image(logo, width=100)  # Adjust width to suit your design needs
    
    # Display the title in the second column
    with col2:
        st.markdown("""
            <h1 style="color: #FF5A5F; line-height: 60px;">Property Management Platform</h1>
            """,
            unsafe_allow_html=True
        )

# Call this function to display the styled title with logo
display_title_with_local_logo()
municipality_data = pd.read_excel("municipality_mapping.xlsx")
municipality_data.set_index('Municipality_y', inplace=True)


# Define your province and municipalities mapping
provinces_municipalities = {
    'Castellón': ['Castelló de la Plana', 'Vila-real', 'Morella', 'Traiguera', 'Càlig',
                  'El Toro', 'Cabanes', "Vall d'Alba", 'Villafranca del Cid', 'Lucena del Cid',
                  'Artana', 'Altura', 'Viver', 'Borriol', 'Moncofa', 'Almenara', 'Nules',
                  'Alcalà de Xivert', "L' Alcora", 'Almassora', 'Benicarló', 'Benicasim',
                  'Burriana', 'Onda', 'Oropesa del Mar', 'Peñíscola', 'Segorbe', 'Torreblanca',
                  "La Vall d'Uixó", 'Vinaròs', 'Alquerías del Niño Perdido', 'Betxí'],
    'Valencia': ['Gandia', 'Sagunto', 'València', 'Torrent', 'Daimús', 'Bellreguard', 'Ador',
                 'Real', 'Benifaió', 'Manuel', 'Llutxent', 'Xeraco', 'Corbera',
                 'Albalat de la Ribera', 'Albal', 'Riba-roja de Túria',
                 'Bonrepòs i Mirambell', 'Albuixech', 'Serra', 'Estivella', 'Pedralba',
                 'Villar del Arzobispo', 'Ademuz', 'Chelva', 'Venta del Moro', 'Chiva',
                 'Catadau', 'Yátova', 'Cofrentes', 'Navarrés', 'Quart de Poblet', 'Anna',
                 'Enguera', 'Mogente', 'Aielo de Malferit', 'Albaida', 'Rótova', 'Montaverner',
                 'Montroy', 'Alfafar', 'Llíria', 'Museros', 'Oliva', 'Alberic', 'Bocairent',
                 'Sollana', "L' Alcúdia de Crespins", 'Ayora', 'Benigànim', 'Guadassuar',
                 'Náquera', "Canet d'En Berenguer", 'Turís', 'Villanueva de Castellón',
                 'Rocafort', 'Almàssera', 'Foios', 'La Pobla de Farnals', "L' Olleria",
                 'Cheste', 'El Puig de Santa Maria', 'San Antonio de Benagéber',
                 'Rafelbunyol', 'Almussafes', 'Tavernes Blanques', 'Buñol', 'Massanassa',
                 'Vilamarxant', 'Alcàsser', 'Sedaví', 'Meliana', 'Benaguasil', 'Picanya',
                 'Utiel', "L' Alcúdia", 'Godella', 'Alginet', 'Canals', 'Benetússer', 'Carlet',
                 'Massamagrell', 'Tavernes de la Valldigna', "L' Eliana", 'Silla', 'Puçol',
                 'Requena', 'Carcaixent', 'Picassent', 'Moncada', 'Cullera', 'Bétera',
                 'La Pobla de Vallbona', 'Alboraya', 'Paiporta', 'Algemesí', 'Sueca',
                 'Catarroja', 'Xàtiva', 'Xirivella', 'Alaquàs', 'Manises', 'Aldaia', 'Ontinyent',
                 'Burjassot', 'Mislata', 'Alzira', 'Paterna', 'Llombai'],
    'Alicante': ['Benidorm', 'Elda', 'Alcoy', 'Orihuela', 'Elche', 'San Vicente del Raspeig',
                 'Alicante', 'Torrevieja', 'Algorfa', 'Formentera del Segura', 'Jacarilla',
                 'Cox', 'Albatera', 'Hondón de las Nieves', 'Sax', 'Tibi', 'Biar',
                 'Sanet y Negrals', 'Busot', 'Benilloba', 'Beniarrés', 'Els Poblets',
                 'Beniarbeig', 'Xaló', 'Orba', 'Benitachell', "L' Alfàs del Pi", 'Almoradí',
                 'Altea', 'Aspe', 'Banyeres de Mariola', 'Benejúzar', 'Benissa', 'Calp',
                 "Callosa d'en Sarrià", 'Callosa de Segura', 'El Campello', 'Castalla',
                 'Catral', 'Cocentaina', 'Crevillent', 'Dénia', 'Dolores', 'Finestrat',
                 'Gata de Gorgos', 'Guardamar del Segura', 'Ibi', 'Jávea', 'Jijona',
                 'Monforte del Cid', 'Monóvar', 'Mutxamel', 'Novelda', 'La Nucia', 'Ondara',
                 'Onil', 'Pedreguer', 'Pego', 'Petrer', 'Polop', 'Redován', 'Rojales',
                 'San Fulgencio', "Sant Joan d'Alacant", 'San Miguel de Salinas',
                 'Santa Pola', 'Villajoyosa', 'Villena', 'Pilar de la Horadada']
}



# User selects a province and a municipality
province = st.selectbox("Select Province", list(provinces_municipalities.keys()))
municipalities = provinces_municipalities[province]
selected_municipality = st.selectbox("Select Municipality", municipalities)

# Retrieve and store municipality-specific data
if selected_municipality in municipality_data.index:
    municipality_info = municipality_data.loc[selected_municipality]
    # The data is fetched and stored but not displayed

# Model selection
model_choice = st.radio("Select model to fit:", ('Occupancy Rate', 'ADR'))

if model_choice == 'Occupancy Rate':
    # Define a function to format titles for consistency
    # Define a function to format titles with more style
    def format_title(text, color='#0c6596'):
        st.markdown(f"<h2 style='text-align: left; color: {color};'>{text}</h2>", unsafe_allow_html=True)

    # Custom styles for the submit button
    def style_submit_button():
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    font-size: 20px;
                    width: 100%;
                    height: 3em;
                    border-radius: 5px;
                    border: none;
                    background-color: #1976D2;
                    color: white;
                }
            </style>""", unsafe_allow_html=True)
    months = {
    'January': 1, 'February': 2, 'March': 3,
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12
    }


    # Main form
    with st.form("property_details"):

        # Initialize a dictionary to hold inputs
        property_inputs = {}

        # Property Specifications Section
        format_title("Property Specifications", color='#0c6596')
        col1, col2 = st.columns(2)
        
        with col1:
            property_inputs['n_rooms'] = st.number_input('Number of Rooms', min_value=1)
            property_inputs['n_baths'] = st.number_input('Number of Bathrooms', min_value=1)
            property_inputs['max_guests'] = st.number_input('Maximum Guests', min_value=1)
            property_inputs['min_stay'] = st.number_input('Minimum Stay (days)', min_value=1)
            property_inputs['n_photos'] = st.number_input('Number of Photos', min_value=0)
        
        with col2:
            property_inputs['property_type'] = st.selectbox("Property Type", ('multi_family', 'room', 'hotel', 'single_family', 'unknown'))
            property_inputs['property_subtype'] = st.selectbox("Property Subtype", ('flat', 'loft', 'unknown', 'special_house', 'detached_house','flat_house_room', 'hospitality_room', 'studio'))
            property_inputs['property_use'] = st.selectbox("Property Use", ('residential', 'unknown', 'hotel_hospitality'))
            property_inputs['amenities'] = st.multiselect("Select Amenities", ['Kitchen', 'Washer', 'TV', 'Essentials', 'Wireless Internet', 'Heating', 'AC', 'Pool', 'Hair-Dryer', 'Free Parking', 'Hot Water', 'Elevator', 'Laptop-Friendly'])

        # Distinct Section for Month of Rental
        st.markdown("---")
        format_title("Rental Period", color='#d89614')
        month_name = st.selectbox("Select the rental month", list(months.keys()))
        property_inputs['month'] = months[month_name]  # Translate month name to number


        # Pricing and Availability Section
        format_title("Pricing and Availability", color='#0c6596')
        col3, col4 = st.columns([3, 2])
        
        with col3:
            if model_choice == 'Occupancy Rate':
                property_inputs['adr_usd'] = st.number_input('Desired Price (USD)', min_value=0.0, format="%.2f")
            else:
                property_inputs['occupancy_rate'] = st.slider('Desired Occupancy Rate', 0.0, 1.0, 0.5)

        with col4:
            property_inputs['deposit_usd'] = st.number_input('Deposit (USD)', min_value=0.0, format="%.2f")
            property_inputs['cleaning_fee_usd'] = st.number_input('Cleaning Fee (USD)', min_value=0.0, format="%.2f")
            property_inputs['extra_people_fee_usd'] = st.number_input('Extra People Fee (USD)', min_value=0.0, format="%.2f")

        # Rental History Section
        format_title("Rental History - Last 12 Months", color='#0c6596')
        col5, col6 = st.columns(2)
        
        with col5:
            property_inputs['n_reviews_ltm'] = st.number_input('Number of Reviews', min_value=0)
            property_inputs['available_days_ltm'] = st.number_input('Available Days', min_value=0, max_value=365)
            property_inputs['reservation_days_ltm'] = st.number_input('Reservation Days', min_value=0, max_value=365)
        
        with col6:
            property_inputs['blocked_days_ltm'] = st.number_input('Blocked Days', min_value=0, max_value=365)
            property_inputs['n_bookings_ltm'] = st.number_input('Number of Bookings', min_value=0)
            property_inputs['annual_revenue_usd'] = st.number_input('Annual Revenue (USD)', min_value=0.0, format="%.2f")
            property_inputs['occupancy_rate_ltm'] = st.slider('Occupancy Rate (%)', 0.0, 1.0, 0.05)

        # Rating and Policies Section
        format_title("Rating and Policies", color='#0c6596')
        col7, col8 = st.columns(2)
        
        with col7:
            property_inputs['rating'] = st.slider('Rating (Last 12 Months)', 0, 100, 5)
        
        with col8:
            property_inputs['policy_category'] = st.selectbox("Policy Category", ['Flexible', 'Moderate', 'Strict'])
            property_inputs['is_instantbookable'] = st.checkbox('Is Instantly Bookable')
            property_inputs['is_superhost'] = st.checkbox('Is Superhost')

        # Fancy Submit Button
        st.markdown("---")
        # Submit Button at the end of the form
        style_submit_button()  # Call to style the submit button
        submit_details = st.form_submit_button("Submit Property Details")

    if submit_details:
        # Process the submitted property details
        print(property_inputs)
    # Retrieve and store municipality-specific data
    if 'province' and 'municipality' in st.session_state:
        if st.session_state.municipality in municipality_data.index:
            municipality_info = municipality_data.loc[st.session_state.municipality]
            property_inputs.update(municipality_info.to_dict())

    # Prepare the DataFrame
    if submit_details:
        property_df = pd.DataFrame([property_inputs])
        st.write("Data ready for model input. Model fitting would be implemented here.")
        # Here you would pr


    # Assuming the Airbnb dataset is stored as "airbnb_data.csv"
    import pandas as pd

    import pandas as pd


    if submit_details:
        # Construct the data dictionary from user inputs and fetched municipality info
        new_property_data = {
            'n_rooms': property_inputs['n_rooms'],
            'n_baths': property_inputs['n_baths'],
            'max_guests': property_inputs['max_guests'],
            'min_stay': property_inputs['min_stay'],
            'n_photos': property_inputs['n_photos'],
            'deposit_usd': property_inputs['deposit_usd'],
            'cleaning_fee_usd': property_inputs['cleaning_fee_usd'],
            'extra_people_fee_usd': property_inputs['extra_people_fee_usd'],
            'is_instantbookable': 1 if property_inputs['is_instantbookable'] else 0,
            'is_superhost': 1 if property_inputs['is_superhost'] else 0,
            'property_type': property_inputs['property_type'],
            'property_subtype': property_inputs['property_subtype'],
            'property_use': property_inputs['property_use'],
            'lat': municipality_info['lat'],
            'lon': municipality_info['lon'],
            # Demographic and geographic data from municipality
            'Población residente (A)': municipality_info['Población residente (A)'],
            'Población residente que se localiza durante el día en su área de residencia (B)': municipality_info['Población residente que se localiza durante el día en su área de residencia (B)'],
            'Población residente encontrada durante el día en otra área (C)': municipality_info['Población residente encontrada durante el día en otra área (C)'],
            'Población no residente que se localiza durante el día en esta área (D)': municipality_info['Población no residente que se localiza durante el día en esta área (D)'],
            'Población total que se localiza durante el día en el área (E=B+D)': municipality_info['Población total que se localiza durante el día en el área (E=B+D)'],
            'Saldo población entra y sale de esta área (F=D-C)': municipality_info['Saldo población entra y sale de esta área (F=D-C)'],
            'Porcentaje de población residente que se localiza durante el día en su área de residencia (B*100/A)': municipality_info['Porcentaje de población residente que se localiza durante el día en su área de residencia (B*100/A)'],
            'Porcentaje de población residente que sale de su área (C*100/A)': municipality_info['Porcentaje de población residente que sale de su área (C*100/A)'],
            'Porcentaje de población no residente que se localiza durante el día en esta área (D*100/A)': municipality_info['Porcentaje de población no residente que se localiza durante el día en esta área (D*100/A)'],
            'Cociente entre población total que se localiza durante el día y población residente (E*100/A)': municipality_info['Cociente entre población total que se localiza durante el día y población residente (E*100/A)'],
            'Porcentaje de población que gana o pierde durante el día (F*100/A)': municipality_info['Porcentaje de población que gana o pierde durante el día (F*100/A)'],
            'Month': property_inputs['month'],
            # Map amenities to binary columns
            'kitchen': 1 if 'Kitchen' in property_inputs['amenities'] else 0,
            'washer': 1 if 'Washer' in property_inputs['amenities'] else 0,
            'tv': 1 if 'TV' in property_inputs['amenities'] else 0,
            'heating': 1 if 'Heating' in property_inputs['amenities'] else 0,
            'ac': 1 if 'AC' in property_inputs['amenities'] else 0,
            'essentials': 1 if 'Essentials' in property_inputs['amenities'] else 0,
            'wireless_internet': 1 if 'Wireless Internet' in property_inputs['amenities'] else 0,
            'hangers': 1 if 'Hangers' in property_inputs['amenities'] else 0,
            'iron': 1 if 'Iron' in property_inputs['amenities'] else 0,
            'pool': 1 if 'Pool' in property_inputs['amenities'] else 0,
            'hair-dryer': 1 if 'Hair Dryer' in property_inputs['amenities'] else 0,
            'free_parking': 1 if 'Free Parking' in property_inputs['amenities'] else 0,
            'hot_water': 1 if 'Hot Water' in property_inputs['amenities'] else 0,
            'elevator': 1 if 'Elevator' in property_inputs['amenities'] else 0,
            'laptop-friendly': 1 if 'Laptop Friendly' in property_inputs['amenities'] else 0,
            'n_reviews_ltm': property_inputs['n_reviews_ltm'],
            'n_bookings_ltm': property_inputs['n_bookings_ltm'],
            'available_days_ltm': property_inputs['available_days_ltm'],
            'reservation_days_ltm': property_inputs['reservation_days_ltm'],
            'blocked_days_ltm': property_inputs['blocked_days_ltm'],
            'annual_revenue_usd': property_inputs['annual_revenue_usd'],
            'occupancy_rate_ltm': property_inputs['occupancy_rate_ltm'],
            'rating': property_inputs['rating'],
            'policy_category': property_inputs['policy_category'],
            'adr_usd': property_inputs['adr_usd']
            # 'occupancy_rate': property_inputs['occupancy_rate']
        }
        # Convert the dictionary into a DataFrame
        new_property_df = pd.DataFrame([new_property_data])


        import pandas as pd
        from geopy.distance import great_circle

        # Load beaches data; assuming 'Latitude' and 'Longitude' columns exist in this data
        df_beaches = pd.read_excel('df_beaches.xlsx')
        # Function to calculate distance to the closest beach
        def get_distance_to_closest_beach(lat, lon):
            airbnb_location = (lat, lon)
            closest_beach = min(
                df_beaches.itertuples(), 
                key=lambda beach: great_circle(airbnb_location, (beach.Latitude, beach.Longitude)).meters
            )
            distance_to_closest_beach = great_circle(airbnb_location, (closest_beach.Latitude, closest_beach.Longitude)).meters
            return distance_to_closest_beach
        # Assuming 'new_property_df' is your DataFrame with the new property details
        # And it already includes 'lat' and 'lon' columns

        # Calculate distance to the closest beach
        new_property_df['distance_to_closest_beach'] = new_property_df.apply(
            lambda row: get_distance_to_closest_beach(row['lat'], row['lon']), axis=1
        )

        # Determine if it is a coastal municipality (within 2000 meters of the nearest beach)
        new_property_df['coastal_municipality'] = (new_property_df['distance_to_closest_beach'] < 2000).astype(int)
        # Print the DataFrame to see all features including the newly added ones
        new_property_df['Province'] = province
        new_property_df['Municipality'] = selected_municipality

        # Add the code for transforming the DataFrame with dummy variables here
        # List of categorical columns
        categorical_cols = ['is_instantbookable', 'is_superhost', 'property_type', 
                            'property_subtype', 'property_use', 'kitchen', 'washer', 
                            'tv', 'essentials', 'wireless_internet', 'hangers', 
                            'iron', 'heating', 'ac', 'pool', 'hair-dryer', 
                            'free_parking', 'hot_water', 'elevator', 'laptop-friendly', 
                            'policy_category', 'Month', 
                            'Municipality', 'Province', 'coastal_municipality']

        # Create dummy variables for categorical columns
        dummy_df = pd.get_dummies(new_property_df, columns=categorical_cols)

        # Display the DataFrame with dummy variables
        def prepare_input(dummy_df, model):
            # Get feature names from the model
            model_features = model.get_booster().feature_names
            
            # Ensure all model features are in the DataFrame, add them if not with a default value
            for feature in model_features:
                if feature not in dummy_df.columns:
                    dummy_df[feature] = 0  # Assigning default value 0 for missing features
            
            # Reorder the DataFrame columns according to the model's feature order
            dummy_df = dummy_df[model_features]
            return dummy_df
        
        prepared_df = prepare_input(dummy_df, model_occupancy)
        prediction = model_occupancy.predict_proba(prepared_df)

        # Convert the probability to percentage and round to two decimal places
        probability_percentage = round(prediction[0][1] * 100, 2)

        # Determine the color and icon based on the occupancy rate probability
        if probability_percentage >= 50:
            color = "#28a745"  # Green color for success
            icon = "✅"  # Check mark icon for success
        else:
            color = "#dc3545"  # Red color for error
            icon = "❌"  # Cross mark icon for error

        # Create a styled prompt for the occupancy rate prediction output
        st.write(f"""
            <div style="background-color:#f8f9fa;padding:10px;border-radius:10px;">
                <h2 style="color:#007bff;">🏠 Occupancy Rate Prediction</h2>
                <p style="font-size:20px;">The probability of the property being rented under those conditions is: <span style="color:{color}; font-weight:bold;">{probability_percentage}% {icon}</span></p>
            </div>
        """, unsafe_allow_html=True)

else:
        # Define a function to format titles for consistency
    # Define a function to format titles with more style
    def format_title(text, color='#0c6596'):
        st.markdown(f"<h2 style='text-align: left; color: {color};'>{text}</h2>", unsafe_allow_html=True)

    # Custom styles for the submit button
    def style_submit_button():
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    font-size: 20px;
                    width: 100%;
                    height: 3em;
                    border-radius: 5px;
                    border: none;
                    background-color: #1976D2;
                    color: white;
                }
            </style>""", unsafe_allow_html=True)
    months = {
    'January': 1, 'February': 2, 'March': 3,
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12
    }


    # Main form
    with st.form("property_details"):
        st.title("Property Management Dashboard", anchor=None)

        # Initialize a dictionary to hold inputs
        property_inputs = {}

        # Property Specifications Section
        format_title("Property Specifications", color='#0c6596')
        col1, col2 = st.columns(2)
        
        with col1:
            property_inputs['n_rooms'] = st.number_input('Number of Rooms', min_value=1)
            property_inputs['n_baths'] = st.number_input('Number of Bathrooms', min_value=1)
            property_inputs['max_guests'] = st.number_input('Maximum Guests', min_value=1)
            property_inputs['min_stay'] = st.number_input('Minimum Stay (days)', min_value=1)
            property_inputs['n_photos'] = st.number_input('Number of Photos', min_value=0)
        
        with col2:
            property_inputs['property_type'] = st.selectbox("Property Type", ('multi_family', 'room', 'hotel', 'single_family', 'unknown'))
            property_inputs['property_subtype'] = st.selectbox("Property Subtype", ('flat', 'loft', 'unknown', 'special_house', 'detached_house','flat_house_room', 'hospitality_room', 'studio'))
            property_inputs['property_use'] = st.selectbox("Property Use", ('residential', 'unknown', 'hotel_hospitality'))
            property_inputs['amenities'] = st.multiselect("Select Amenities", ['Kitchen', 'Washer', 'TV', 'Essentials', 'Wireless Internet', 'Heating', 'AC', 'Pool', 'Hair-Dryer', 'Free Parking', 'Hot Water', 'Elevator', 'Laptop-Friendly'])

        # Distinct Section for Month of Rental
        st.markdown("---")
        format_title("Rental Period", color='#d89614')
        month_name = st.selectbox("Select the rental month", list(months.keys()))
        property_inputs['month'] = months[month_name]  # Translate month name to number


        # Pricing and Availability Section
        format_title("Pricing and Availability", color='#0c6596')
        col3, col4 = st.columns([3, 2])
        
        with col3:
            if model_choice == 'Occupancy Rate':
                property_inputs['adr_usd'] = st.number_input('Desired Price (USD)', min_value=0.0, format="%.2f")
            else:
                property_inputs['occupancy_rate'] = st.slider('Desired Occupancy Rate', 0.0, 1.0, 0.5)

        with col4:
            property_inputs['deposit_usd'] = st.number_input('Deposit (USD)', min_value=0.0, format="%.2f")
            property_inputs['cleaning_fee_usd'] = st.number_input('Cleaning Fee (USD)', min_value=0.0, format="%.2f")
            property_inputs['extra_people_fee_usd'] = st.number_input('Extra People Fee (USD)', min_value=0.0, format="%.2f")

        # Rental History Section
        format_title("Rental History - Last 12 Months", color='#0c6596')
        col5, col6 = st.columns(2)
        
        with col5:
            property_inputs['n_reviews_ltm'] = st.number_input('Number of Reviews', min_value=0)
            property_inputs['available_days_ltm'] = st.number_input('Available Days', min_value=0, max_value=365)
            property_inputs['reservation_days_ltm'] = st.number_input('Reservation Days', min_value=0, max_value=365)
        
        with col6:
            property_inputs['blocked_days_ltm'] = st.number_input('Blocked Days', min_value=0, max_value=365)
            property_inputs['n_bookings_ltm'] = st.number_input('Number of Bookings', min_value=0)
            property_inputs['annual_revenue_usd'] = st.number_input('Annual Revenue (USD)', min_value=0.0, format="%.2f")
            property_inputs['occupancy_rate_ltm'] = st.slider('Occupancy Rate (%)', 0.0, 1.0, 0.05)

        # Rating and Policies Section
        format_title("Rating and Policies", color='#0c6596')
        col7, col8 = st.columns(2)
        
        with col7:
            property_inputs['rating'] = st.slider('Rating (Last 12 Months)', 0, 100, 5)
        
        with col8:
            property_inputs['policy_category'] = st.selectbox("Policy Category", ['Flexible', 'Moderate', 'Strict'])
            property_inputs['is_instantbookable'] = st.checkbox('Is Instantly Bookable')
            property_inputs['is_superhost'] = st.checkbox('Is Superhost')

        # Fancy Submit Button
        st.markdown("---")
        # Submit Button at the end of the form
        style_submit_button()  # Call to style the submit button
        submit_details = st.form_submit_button("Submit Property Details")


    if submit_details:
        # Process the submitted property details
        print(property_inputs)
    # Retrieve and store municipality-specific data
    if 'province' and 'municipality' in st.session_state:
        if st.session_state.municipality in municipality_data.index:
            municipality_info = municipality_data.loc[st.session_state.municipality]
            property_inputs.update(municipality_info.to_dict())

    # Prepare the DataFrame
    if submit_details:
        property_df = pd.DataFrame([property_inputs])
        st.write("Data ready for model input. Model fitting would be implemented here.")
        # Here you would pr


    # Assuming the Airbnb dataset is stored as "airbnb_data.csv"
    import pandas as pd

    import pandas as pd


    if submit_details:
        # Construct the data dictionary from user inputs and fetched municipality info
        new_property_data = {
            'n_rooms': property_inputs['n_rooms'],
            'n_baths': property_inputs['n_baths'],
            'max_guests': property_inputs['max_guests'],
            'min_stay': property_inputs['min_stay'],
            'n_photos': property_inputs['n_photos'],
            'deposit_usd': property_inputs['deposit_usd'],
            'cleaning_fee_usd': property_inputs['cleaning_fee_usd'],
            'extra_people_fee_usd': property_inputs['extra_people_fee_usd'],
            'is_instantbookable': 1 if property_inputs['is_instantbookable'] else 0,
            'is_superhost': 1 if property_inputs['is_superhost'] else 0,
            'property_type': property_inputs['property_type'],
            'property_subtype': property_inputs['property_subtype'],
            'property_use': property_inputs['property_use'],
            'lat': municipality_info['lat'],
            'lon': municipality_info['lon'],
            # Demographic and geographic data from municipality
            'Población residente (A)': municipality_info['Población residente (A)'],
            'Población residente que se localiza durante el día en su área de residencia (B)': municipality_info['Población residente que se localiza durante el día en su área de residencia (B)'],
            'Población residente encontrada durante el día en otra área (C)': municipality_info['Población residente encontrada durante el día en otra área (C)'],
            'Población no residente que se localiza durante el día en esta área (D)': municipality_info['Población no residente que se localiza durante el día en esta área (D)'],
            'Población total que se localiza durante el día en el área (E=B+D)': municipality_info['Población total que se localiza durante el día en el área (E=B+D)'],
            'Saldo población entra y sale de esta área (F=D-C)': municipality_info['Saldo población entra y sale de esta área (F=D-C)'],
            'Porcentaje de población residente que se localiza durante el día en su área de residencia (B*100/A)': municipality_info['Porcentaje de población residente que se localiza durante el día en su área de residencia (B*100/A)'],
            'Porcentaje de población residente que sale de su área (C*100/A)': municipality_info['Porcentaje de población residente que sale de su área (C*100/A)'],
            'Porcentaje de población no residente que se localiza durante el día en esta área (D*100/A)': municipality_info['Porcentaje de población no residente que se localiza durante el día en esta área (D*100/A)'],
            'Cociente entre población total que se localiza durante el día y población residente (E*100/A)': municipality_info['Cociente entre población total que se localiza durante el día y población residente (E*100/A)'],
            'Porcentaje de población que gana o pierde durante el día (F*100/A)': municipality_info['Porcentaje de población que gana o pierde durante el día (F*100/A)'],
            'Month': property_inputs['month'],
            # Map amenities to binary columns
            'kitchen': 1 if 'Kitchen' in property_inputs['amenities'] else 0,
            'washer': 1 if 'Washer' in property_inputs['amenities'] else 0,
            'tv': 1 if 'TV' in property_inputs['amenities'] else 0,
            'heating': 1 if 'Heating' in property_inputs['amenities'] else 0,
            'ac': 1 if 'AC' in property_inputs['amenities'] else 0,
            'essentials': 1 if 'Essentials' in property_inputs['amenities'] else 0,
            'wireless_internet': 1 if 'Wireless Internet' in property_inputs['amenities'] else 0,
            'hangers': 1 if 'Hangers' in property_inputs['amenities'] else 0,
            'iron': 1 if 'Iron' in property_inputs['amenities'] else 0,
            'pool': 1 if 'Pool' in property_inputs['amenities'] else 0,
            'hair-dryer': 1 if 'Hair Dryer' in property_inputs['amenities'] else 0,
            'free_parking': 1 if 'Free Parking' in property_inputs['amenities'] else 0,
            'hot_water': 1 if 'Hot Water' in property_inputs['amenities'] else 0,
            'elevator': 1 if 'Elevator' in property_inputs['amenities'] else 0,
            'laptop-friendly': 1 if 'Laptop Friendly' in property_inputs['amenities'] else 0,
            'n_reviews_ltm': property_inputs['n_reviews_ltm'],
            'n_bookings_ltm': property_inputs['n_bookings_ltm'],
            'available_days_ltm': property_inputs['available_days_ltm'],
            'reservation_days_ltm': property_inputs['reservation_days_ltm'],
            'blocked_days_ltm': property_inputs['blocked_days_ltm'],
            'annual_revenue_usd': property_inputs['annual_revenue_usd'],
            'occupancy_rate_ltm': property_inputs['occupancy_rate_ltm'],
            'rating': property_inputs['rating'],
            'policy_category': property_inputs['policy_category'],
            'occupancy_rate': property_inputs['occupancy_rate']
        }
        # Convert the dictionary into a DataFrame
        new_property_df = pd.DataFrame([new_property_data])


        import pandas as pd
        from geopy.distance import great_circle

        # Load beaches data; assuming 'Latitude' and 'Longitude' columns exist in this data
        df_beaches = pd.read_excel('df_beaches.xlsx')
        # Function to calculate distance to the closest beach
        def get_distance_to_closest_beach(lat, lon):
            airbnb_location = (lat, lon)
            closest_beach = min(
                df_beaches.itertuples(), 
                key=lambda beach: great_circle(airbnb_location, (beach.Latitude, beach.Longitude)).meters
            )
            distance_to_closest_beach = great_circle(airbnb_location, (closest_beach.Latitude, closest_beach.Longitude)).meters
            return distance_to_closest_beach
        # Assuming 'new_property_df' is your DataFrame with the new property details
        # And it already includes 'lat' and 'lon' columns

        # Calculate distance to the closest beach
        new_property_df['distance_to_closest_beach'] = new_property_df.apply(
            lambda row: get_distance_to_closest_beach(row['lat'], row['lon']), axis=1
        )

        # Determine if it is a coastal municipality (within 2000 meters of the nearest beach)
        new_property_df['coastal_municipality'] = (new_property_df['distance_to_closest_beach'] < 2000).astype(int)
        # Print the DataFrame to see all features including the newly added ones
        new_property_df['Province'] = province
        new_property_df['Municipality'] = selected_municipality

        # Add the code for transforming the DataFrame with dummy variables here
        # List of categorical columns
        categorical_cols = ['is_instantbookable', 'is_superhost', 'property_type', 
                            'property_subtype', 'property_use', 'kitchen', 'washer', 
                            'tv', 'essentials', 'wireless_internet', 'hangers', 
                            'iron', 'heating', 'ac', 'pool', 'hair-dryer', 
                            'free_parking', 'hot_water', 'elevator', 'laptop-friendly', 
                            'policy_category', 'Month', 
                            'Municipality', 'Province', 'coastal_municipality']

        # Create dummy variables for categorical columns
        dummy_df = pd.get_dummies(new_property_df, columns=categorical_cols)

        # Display the DataFrame with dummy variables
        def prepare_input(dummy_df, model):
            # Get feature names from the model
            model_features = model.get_booster().feature_names
            
            # Ensure all model features are in the DataFrame, add them if not with a default value
            for feature in model_features:
                if feature not in dummy_df.columns:
                    dummy_df[feature] = 0  # Assigning default value 0 for missing features
            
            # Reorder the DataFrame columns according to the model's feature order
            dummy_df = dummy_df[model_features]
            return dummy_df
        
        prepared_df = prepare_input(dummy_df, model_adr)
        prediction = model_adr.predict(prepared_df)
        formatted_prediction = "{:.2f}".format(prediction[0])

        # Create a styled prompt for the prediction output
        st.write(f"""
            <div style="background-color:#f0f0f0;padding:10px;border-radius:10px;">
                <h2 style="color:#1f77b4;">🏠 Recommended Price</h2>
                <p style="font-size:20px;">The recommended price per night for your property under those conditions is: <span style="color:#ff7f0e;">${formatted_prediction}/night</span></p>
            </div>
        """, unsafe_allow_html=True)


        # def display_feature_importance(model_adr, prepared_df):
        #     # Get feature importances
        #     importance = model_adr.get_booster().get_score(importance_type='weight')
        #     importance_df = pd.DataFrame({
        #     'Feature': importance.keys(),
        #     'Importance': importance.values()
        #     }).sort_values(by='Importance', ascending=False).head(10)  
        
        #     # Display the feature importances
        #     st.write("Key Features Influencing Your Listing's Performance:")
        #     st.bar_chart(importance_df.set_index('Feature'))
        
        #     # Optionally, give specific suggestions based on the importance
        #     top_features = importance_df.head(3)['Feature'].tolist()
        #     st.write(f"The top factors affecting your listing are: {', '.join(top_features)}. Consider optimizing these aspects to improve your results.")
        
        #     # Example of usage within your Streamlit app
        #     if submit_details:
        #         prediction = model_adr.predict(prepared_df)  # Assuming `data_df` is prepared for prediction
        # display_feature_importance(model_adr, prepared_df)
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        import streamlit as st
        
        # Load and prepare the data
        # Load preprocessed and grouped data
        df = pd.read_pickle('preprocessed_grouped_data.pkl')
        
        # Define function for preprocessing data
        def preprocess_data(df, month, max_adr_room):
            df_filtered = df.loc[(df['adr_usd'] > 1) & (df['occupancy_rate'] > 0) & (df['occupancy_rate'] < 1) & (df['date_in'].dt.month == month)]
            df_filtered['adr_room'] = df_filtered['adr_usd'] / df_filtered['n_rooms']
            df_filtered = df_filtered.loc[df_filtered['adr_room'] <= max_adr_room]
            df_filtered[['occupancy_rate_log', 'adr_usd_log', 'adr_room_log', 'n_rooms_log', 'n_bookings_log']] = np.log1p(df_filtered[['occupancy_rate', 'adr_usd', 'adr_room', 'n_rooms', 'n_bookings']])
            return df_filtered
        
        # Define function for grouping data
        def group_data(df):
            df["labels"] = pd.cut(df["adr_room"], bins=1000, labels=False)
            df_grouped = df.groupby(by=["property_subtype", "labels"]).agg({"adr_room": "mean", "occupancy_rate": ["mean", "count"], "adr_room_log": "mean", "occupancy_rate_log": "mean"}).reset_index()
            df_grouped.columns = ["property_subtype", "labels", "Mean ADR room", "Mean Occupancy Rate", "Observations per Bin", "Mean ADR room log", "Mean Occupancy Rate log"]
            return df_grouped
        
        # Define function for plotting
        def plot_data(df, property_subtype):
            df_aux = df.loc[df["property_subtype"] == property_subtype]
            sns.relplot(x="Mean ADR room", y="Mean Occupancy Rate", size="Observations per Bin", alpha=.5, palette="muted", height=6, data=df_aux)
        
        # Define function for modeling
        def model_data(df_aux):
            lb = 2
            ub = 400
            kr = GridSearchCV(KernelRidge(kernel="poly", degree=3), cv=10, param_grid={"alpha": [100, 10, 1, 0.1, 0.001], "gamma": np.logspace(-5, 10, 1)})
            grid = np.r_[lb:ub:100j].reshape(-1,1)
            kr.fit(X=df_aux["Mean ADR room"].values.reshape(-1,1), y=df_aux["Mean Occupancy Rate"], sample_weight=df_aux["Observations per Bin"])
            plt.plot(grid, kr.predict(grid), linewidth=2)
            ingresos = grid.reshape(1,-1) * kr.predict(grid)*30
            max_ingresos = max(ingresos[0])
            x = grid.reshape(1,-1)[0]
            best_price = x[ingresos[0] >= max_ingresos]
            return best_price, max_ingresos
        
        # Streamlit app
        def main():
            st.title("Your Streamlit App Title")
        
            # Sidebar inputs
            property_subtype = st.sidebar.text_input("Enter Property Subtype")
            month = st.sidebar.number_input("Enter Month", min_value=1, max_value=12)
            max_adr_room = st.sidebar.number_input("Enter Maximum ADR Room")
        
            # Preprocess data
            df_filtered = preprocess_data(df, month, max_adr_room)
        
            # Group data
            df_grouped = group_data(df_filtered)
        
            # Plot data
            plot_data(df_grouped, property_subtype)
            st.pyplot()
        
            # Model data
            best_price, max_income = model_data(df_grouped)
            st.write("Optimal price:", best_price)
            st.write("Max income per month:", max_income)
        
        if __name__ == "__main__":
            main()



