
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Required for joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GDM Risk Assessment",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LOAD ASSETS ---
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    explainer = joblib.load('shap_explainer.pkl')
except FileNotFoundError:
    st.error("Error: Model, scaler, or explainer files not found. Please ensure all .pkl files are in the directory.")
    st.stop()

# --- DIETARY RECOMMENDATIONS DATABASE (GDM FOCUSED) ---
food_recommendations = {
    'High Risk': {
        'Description': 'This suggests a high probability of GDM. It is essential to consult your doctor promptly. The following diet focuses on strict blood sugar control, which is crucial for a healthy pregnancy under these conditions.',
        'Indian (Veg)': {
            'Breakfast': ['Vegetable Oats or Dalia (Porridge)', 'Ragi Idli/Dosa with minimal chutney', 'Moong Dal Chilla with vegetables'],
            'Lunch': ['Large Salad with grilled Paneer/Tofu', 'Karela (Bitter Gourd) Sabzi with 1-2 Millet (Bajra/Jowar) Rotis', 'Lentil soup (Dal) without rice'],
            'Dinner': ['Cauliflower Rice with Vegetable Curry', 'Steamed vegetables with a small portion of quinoa', 'Clear vegetable and lentil soup'],
            'Foods to Avoid': ['White Rice, Sugar, Maida (refined flour)', 'Fried Foods, Processed Snacks', 'Sweet Fruits (e.g., Mango, Grapes, Banana)', 'Root Vegetables (e.g., Potato, Sweet Potato)']
        },
        'Indian (Non-Veg)': {
            'Breakfast': ['Egg white omelette with spinach and mushrooms', 'Boiled Eggs (2)'],
            'Lunch': ['Grilled Fish with sautéed vegetables', 'Skinless Chicken curry (thin, soupy gravy) with a large salad', 'Clear chicken and vegetable broth'],
            'Dinner': ['Baked Chicken breast with greens', 'Fish Tikka (baked, not fried)', 'Scrambled eggs with vegetables'],
            'Foods to Avoid': ['Red Meat (Mutton)', 'Fried Fish/Chicken', 'Sugary Drinks & Fruit Juices', 'Full-fat dairy']
        }
    },
    'Moderate Risk': {
        'Description': 'The model indicates a significant risk of GDM. Discussing these results with your healthcare provider is highly recommended. This diet plan focuses on proactive management of blood sugar.',
        'Indian (Veg)': {
            'Breakfast': ['Oats Upma with vegetables', 'Moong Dal Chilla', 'Greek Yogurt with a few berries'],
            'Lunch': ['Quinoa with Mixed Vegetable Curry', 'Brown Rice (small portion) with Dal Tadka and a large Salad', 'Millet Roti with Paneer Bhurji'],
            'Dinner': ['Grilled Tofu with Sautéed Vegetables', 'Lentil Soup (Dal) with a side of steamed greens', 'Besan Kadhi with Brown Rice (small portion)'],
            'Foods to Limit': ['Potatoes and other root vegetables', 'White Bread/Rice (switch to whole grains)', 'Sweets and desserts', 'Sweetened beverages']
        },
        'Indian (Non-Veg)': {
            'Breakfast': ['Scrambled Eggs (whole eggs) with Spinach', 'Oats with small pieces of chicken'],
            'Lunch': ['Grilled Fish with Brown Rice and Sabzi', 'Chicken Curry (less oil) with Millet Roti', 'Egg Bhurji with whole wheat toast'],
            'Dinner': ['Chicken Tikka (baked) with Mint Chutney and Salad', 'Fish Curry with limited brown rice', 'Clear Chicken Soup'],
            'Foods to Limit': ['High-fat gravies', 'Processed meats (sausages, salami)', 'Full-fat dairy products']
        }
    },
    'Low Risk': {
        'Description': 'The model suggests a low likelihood of GDM. This is a great status to maintain. The following suggestions focus on a preventative, healthy pregnancy diet.',
        'Indian (Veg)': {
            'Breakfast': ['Idli/Dosa with Sambhar', 'Poha with vegetables', 'Whole wheat paratha with curd'],
            'Lunch': ['Standard Thali: Roti, Sabzi, Dal, Rice (portion control), Salad', 'Rajma Chawal with a side of raita'],
            'Dinner': ['Paneer Tikka Masala with whole wheat roti', 'Vegetable Pulao with Raita', 'Dal Makhani (homemade, less butter)'],
            'Good Habits': ['Stay hydrated with water', 'Include a variety of colorful vegetables', 'Choose whole fruits over juices']
        },
        'Indian (Non-Veg)': {
            'Breakfast': ['Omelette with whole wheat bread', 'Keema Paratha (limited oil)'],
            'Lunch': ['Chicken Biryani (homemade, less oil) with Raita', 'Fish Fry (pan-fried) with Dal and Rice'],
            'Dinner': ['Butter Chicken (homemade, healthier version) with Naan', 'Mutton Curry (lean cuts, occasional) with Roti'],
            'Good Habits': ['Choose lean cuts of meat', 'Prefer grilling, baking, or stir-frying over deep-frying']
        }
    },
    'Minimal Risk': {
        'Description': 'Excellent! The model indicates a minimal risk of GDM. Your focus should be on continuing a balanced and nutritious diet for a healthy pregnancy.',
        'Indian (Veg)': {
            'General Advice': [
                'Continue eating a balanced diet rich in fiber, vitamins, and minerals.',
                'Ensure you are getting adequate protein from sources like paneer, dal, and legumes.',
                'There are no specific restrictions, but moderation is always key for long-term health and a healthy pregnancy.'
            ]
        },
        'Indian (Non-Veg)': {
            'General Advice': [
                'Your current dietary patterns appear healthy for your pregnancy.',
                'Continue to include lean proteins like chicken, fish, and eggs.',
                'Focus on maintaining a healthy pregnancy weight as advised by your doctor.'
            ]
        }
    }
}

# --- HELPER FUNCTIONS ---
def get_risk_level(probability):
    """Maps a probability score to a risk category and color."""
    prob_percent = probability * 100
    if prob_percent >= 75:
        return 'High Risk', 'red'
    elif 50 <= prob_percent < 75:
        return 'Moderate Risk', 'orange'
    elif 20 <= prob_percent < 50:
        return 'Low Risk', 'gold'
    else:
        return 'Minimal Risk', 'green'

def predict_diabetes(data):
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([data], columns=feature_names)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    return prediction[0], prediction_proba[0], scaled_input

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- UI LAYOUT ---
st.title('🤰 Gestational Diabetes (GDM) Risk Assessment')
st.markdown("This tool assesses the risk of GDM in expectant mothers and provides tailored dietary guidance.")
st.write('---')

# --- SIDEBAR ---
st.sidebar.header('Patient Medical Data Entry')
pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1, step=1)

#
# --- THIS IS THE CORRECTED LINE ---
#
glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0, max_value=999, value=120) # max_value changed from 250 to 999
#
# ------------------------------------
#
blood_pressure = st.sidebar.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=140, value=70)
skin_thickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
diabetes_pedigree = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.sidebar.number_input('Age', min_value=21, max_value=90, value=30)
user_data = {
    'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree, 'Age': age
}

# --- PREDICTION AND RESULTS DISPLAY ---
if st.sidebar.button('**Assess Risk & Get Diet Plan**', type="primary"):
    data_for_prediction = user_data.copy()
    cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    medians = {'Glucose': 117, 'BloodPressure': 72, 'SkinThickness': 29, 'Insulin': 125, 'BMI': 32.3}
    for col in cols_to_clean:
        if data_for_prediction[col] == 0:
            data_for_prediction[col] = medians[col]

    prediction, prediction_proba, scaled_input = predict_diabetes(data_for_prediction)
    probability_of_diabetes = float(prediction_proba[1])
    risk_name, risk_color = get_risk_level(probability_of_diabetes)

    st.subheader('1. Risk Assessment Results')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Predicted Risk Level:** <span style='color:{risk_color}; font-size: 24px; font-weight: bold;'>{risk_name}</span>", unsafe_allow_html=True)
        st.metric(label="Probability of GDM", value=f"{probability_of_diabetes:.2%}")
        st.info(food_recommendations[risk_name]['Description'])

    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = probability_of_diabetes * 100,
            domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Risk Score"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': risk_color},
                     'steps': [{'range': [0, 20], 'color': 'lightgreen'}, {'range': [20, 50], 'color': '#F9E79F'},
                               {'range': [50, 75], 'color': 'orange'}, {'range': [75, 100], 'color': 'lightcoral'}]}))
        fig.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=40))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Click here to see the factors influencing this prediction"):
        st.subheader('2. Prediction Explanation')
        shap_values = explainer.shap_values(scaled_input)
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], pd.DataFrame([user_data]), matplotlib=False)
        st_shap(force_plot, 400)
        st.markdown("""*This chart shows which factors pushed the risk score higher (red) or lower (blue). Longer bars have a bigger impact.*""")
    
    st.write("---")

    st.subheader(f"3. Dietary Guidance for a '{risk_name}' Status")
    st.warning("🚨 **Disclaimer:** This is an educational tool, not medical advice. Please consult your doctor or a registered dietitian for a personalized pregnancy nutrition plan.", icon="⚠️")
    
    recommendations = food_recommendations[risk_name]
    veg_tab, nonveg_tab = st.tabs(["🍛 Indian (Vegetarian)", "🍗 Indian (Non-Vegetarian)"])

    with veg_tab:
        for meal, items in recommendations['Indian (Veg)'].items():
            if meal not in ['Description', 'General Advice']:
                st.markdown(f"**{meal}**")
                for item in items:
                    st.markdown(f"- {item}")
        if 'General Advice' in recommendations['Indian (Veg)']:
            for item in recommendations['Indian (Veg)']['General Advice']:
                st.markdown(f"• {item}")
    
    with nonveg_tab:
        for meal, items in recommendations['Indian (Non-Veg)'].items():
            if meal not in ['Description', 'General Advice']:
                st.markdown(f"**{meal}**")
                for item in items:
                    st.markdown(f"- {item}")
        if 'General Advice' in recommendations['Indian (Non-Veg)']:
            for item in recommendations['Indian (Non-Veg)']['General Advice']:
                st.markdown(f"• {item}")

else:
    st.info('Enter the patient\'s data in the sidebar and click the button to assess GDM risk and receive guidance.')
