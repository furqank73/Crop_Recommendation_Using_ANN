import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
import time
import sklearn

# Page configuration
st.set_page_config(
    page_title="Agricultural Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful professional look
st.markdown("""
    <style>
        :root {
            --primary: #4CAF50;
            --secondary: #8BC34A;
            --accent: #FFC107;
            --dark: #2E7D32;
            --light: #F1F8E9;
        }
        .main {
            background: linear-gradient(135deg, #F1F8E9, #DCEDC8);
        }
        .stNumberInput, .stSelectbox, .stTextInput {
            border-radius: 10px;
            border: 2px solid var(--secondary);
            background-color: white;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background: linear-gradient(to right, var(--primary), var(--dark));
            color: white;
            font-weight: bold;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            background: linear-gradient(to right, var(--dark), var(--primary));
        }
        .success-box {
            border-left: 5px solid var(--primary);
            background: linear-gradient(to right, #E8F5E9, #C8E6C9);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .parameter-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 10px rgba(0,0,0,0.08);
            margin-bottom: 25px;
            border-top: 5px solid var(--accent);
        }
        .header-text {
            color: var(--dark);
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .social-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }
        .social-links a {
            color: white !important;
            background: var(--primary);
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .social-links a:hover {
            background: var(--dark);
            transform: scale(1.05);
        }
        .title-highlight {
            background: linear-gradient(to right, var(--primary), var(--accent));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-weight: bold;
            display: inline;
        }
    </style>
""", unsafe_allow_html=True)

# Vibrant App Header with gradient title
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 10px;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0;'>
            <span class='title-highlight'>🌾 Agricultural Advisor</span>
        </h1>
        <p style='font-size: 1.1rem; color: var(--dark);'>
            AI-powered precision agriculture solution
    </div>
    """, unsafe_allow_html=True)

# Decorative divider
st.markdown("""
<div style='height: 3px; background: linear-gradient(to right, var(--primary), var(--accent), var(--primary)); 
            margin: 10px 0 30px 0; border-radius: 3px;'></div>
""", unsafe_allow_html=True)

# Load model and preprocessing tools with progress indicators
@st.cache_resource(show_spinner=False)
def load_models():
    with st.spinner('Loading AI models... This may take a moment'):
        model = tf.keras.models.load_model("crop_recommendation_model.keras", compile=False)
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        time.sleep(1)  # Simulate loading for demo
        return model, scaler, label_encoder

model, scaler, label_encoder = load_models()

# Input sections with colorful cards
with st.container():
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h3 class='header-text' style='font-size: 1.5rem;'>
            Enter <span style='color: var(--primary);'>Soil</span> and 
            <span style='color: var(--accent);'>Weather</span> Parameters
        </h3>
        <p style='color: var(--dark);'>Provide accurate measurements for precise crop recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        with stylable_container(
            key="nutrient_card",
            css_styles="""
                {
                    border-radius: 15px;
                    background: linear-gradient(145deg, #FFFFFF, #F1F8E9);
                    padding: 25px;
                    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
                    border-top: 5px solid var(--primary);
                }
            """
        ):
            st.markdown("#### 🌱 Nutrient Levels")
            N = st.slider("Nitrogen (N) ppm", 0.0, 140.0, 50.0, help="Measure of nitrogen content in soil")
            P = st.slider("Phosphorus (P) ppm", 5.0, 145.0, 50.0, help="Measure of phosphorus content in soil")
            K = st.slider("Potassium (K) ppm", 5.0, 205.0, 50.0, help="Measure of potassium content in soil")
            ph = st.slider("Soil pH", 3.0, 10.0, 6.5, 0.1, help="Acidity/alkalinity level of the soil")
    
    with col2:
        with stylable_container(
            key="weather_card",
            css_styles="""
                {
                    border-radius: 15px;
                    background: linear-gradient(145deg, #FFFFFF, #E3F2FD);
                    padding: 25px;
                    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
                    border-top: 5px solid #2196F3;
                }
            """
        ):
            st.markdown("#### ☁️ Weather Conditions")
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1, help="Average daily temperature")
            humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0, help="Relative humidity level")
            rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0, help="Monthly rainfall amount")

# Predict button with enhanced UI
predict_col, _, info_col = st.columns([1, 0.1, 2])
with predict_col:
    if st.button("Get Crop Recommendation", use_container_width=True):
        with st.spinner('Analyzing conditions and predicting optimal crop...'):
            # Simulate processing for realistic feel
            time.sleep(1.5)
            
            # Preprocess input
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled_input = scaler.transform(input_data)

            # Predict and decode
            prediction = model.predict(scaled_input)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
            
            # Show animated result
            with stylable_container(
                key="success_box",
                css_styles="""
                    {
                        border-left: 5px solid var(--primary);
                        background: linear-gradient(to right, #f7faf8, #f7faf8);
                        padding: 25px;
                        border-radius: 15px;
                        margin-top: 20px;
                        animation: fadeIn 0.5s ease-in-out;
                        box-shadow: 0 6px 10px rgba(0,0,0,0.08);
                    }
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                """
            ):
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h3 style='color: var(--dark); margin-bottom: 20px;'>Recommended Crop</h3>
                    <div style='font-size: 28px; font-weight: bold; margin: 20px 0; 
                                color: var(--dark); background: linear-gradient(to right, var(--primary), var(--accent));
                                -webkit-background-clip: text; background-clip: text; color: transparent;'>
                        🌻 {predicted_label[0].capitalize()}
                    </div>
                    <p style='color: var(--dark);'>
                        This crop has the highest suitability for your provided conditions
                    </p>
                </div>
                """, unsafe_allow_html=True)

with info_col:
    with stylable_container(
        key="info_card",
        css_styles="""
            {
                border-radius: 15px;
                background: linear-gradient(145deg, #E3F2FD, #BBDEFB);
                padding: 25px;
                border-left: 5px solid #2196F3;
                box-shadow: 0 6px 10px rgba(0,0,0,0.08);
            }
        """
    ):
        st.markdown("""
        ### ℹ️ How It Works
        1. Provide accurate soil and weather measurements
        2. Our AI model analyzes 7 key parameters
        3. The system predicts the most suitable crop
        4. Get actionable recommendations
        
        **For best results:**
        - Use recent soil test data
        - Input average weather conditions
        - Consider seasonal variations
        
        **Technical Details:**
        - Powered by TensorFlow deep learning
        - Trained on agricultural datasets
        - Continuously improving accuracy
        """)

# Colorful Footer
st.markdown("""
<div style='height: 3px; background: linear-gradient(to right, var(--primary), var(--accent), var(--primary)); 
            margin: 40px 0 20px 0; border-radius: 3px;'></div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: var(--dark); font-size: 14px; padding: 20px;'>
    <p style='margin-bottom: 10px;'>© 2025 <strong>Agricultural Advisor</strong> | Precision Agriculture AI Solution</p>
    <div class='social-links' style='justify-content: center; gap: 15px; margin-bottom: 15px;'>
        <a href='https://www.linkedin.com/in/furqan-khan-256798268/' target='_blank'>LinkedIn</a>
        <a href='https://github.com/furqank73' target='_blank'>GitHub</a>
    </div>
    <p style='font-size: 12px; color: #666;'>
        Results may vary based on actual field conditions. Always consult with local agricultural experts.
    </p>
</div>
""", unsafe_allow_html=True)