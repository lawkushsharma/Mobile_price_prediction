import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

# Load Model and Dataset
model = joblib.load("mobile_price_prediction_model.pkl")
df = pd.read_csv("mobile_data_cleaned.csv")

# App Title
st.title("📱 Mobile Price Prediction & Recommendation System")

st.write(
    "Enter mobile specifications to predict its price category "
    "and get mobile recommendations."
)

st.success("Model and dataset loaded successfully!")




st.subheader("📱 Enter Mobile Specifications")

st.write("Fill the details below to predict the mobile price category.")



brand = st.selectbox(
    "Select Mobile Brand",
    sorted(df["Brand"].dropna().unique())
)

model_name = st.selectbox(
    "Select Mobile Model",
    sorted(df[df["Brand"] == brand]["Model_name"].dropna().unique())
)


ram = st.selectbox(
    "Select RAM (GB)",
    sorted(df["Ram"].dropna().unique())
)

storage = st.selectbox(
    "Select Storage (GB)",
    sorted(df["Storage"].dropna().unique())
)



processor = st.selectbox(
    "Select Processor",
    sorted(df["Processor"].dropna().unique())
)

battery_power = st.selectbox(
    "Select Battery Power",
    sorted(df["Battery_power"].dropna().unique())
)



camera = st.selectbox(
    "Select Camera",
    sorted(df["Camera"].dropna().unique())
)

display = st.selectbox(
    "Select Display",
    sorted(df["Display"].dropna().unique())
)



screen_size = st.selectbox(
    "Select Screen Size",
    sorted(df["Screen_size"].dropna().unique())
)

resolution = st.selectbox(
    "Select Resolution",
    sorted(df["Resolution"].dropna().unique())
)



operating_system = st.selectbox(
    "Select Operating System",
    sorted(df["Operating_system"].dropna().unique())
)

dual_sim = st.selectbox(
    "Dual SIM",
    sorted(df["Dual_sim"].dropna().unique())
)



colour = st.selectbox(
    "Select Colour",
    sorted(df["Colour"].dropna().unique())
)

android_version = st.selectbox(
    "Select Android Version",
    sorted(df["Android_version"].dropna().unique())
)





if st.button("Predict Mobile Price"):

    input_data = pd.DataFrame([{
        "Brand": brand,
        "Model_name": model_name,
        "Ram": ram,
        "Storage": storage,
        "Processor": processor,
        "Battery_power": battery_power,
        "Camera": camera,
        "Display": display,
        "Screen_size": screen_size,
        "Resolution": resolution,
        "Operating_system": operating_system,
        "Dual_sim": dual_sim,
        "Colour": colour,
        "Android_version": android_version
    }])

    # Prediction
    prediction = model.predict(input_data)

    # Prediction Result
    st.success("Prediction Completed Successfully!")
    st.markdown("### 📱 Predicted Price Category")
    st.title(f"🎯 {prediction[0]}")

    # Selected Mobile Details
    st.markdown("### 📋 Selected Mobile Details")

    st.write(f"**Brand:** {brand}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**RAM:** {ram} GB")
    st.write(f"**Storage:** {storage} GB")
    st.write(f"**Processor:** {processor}")
    st.write(f"**Battery:** {battery_power}")
    st.write(f"**Camera:** {camera}")
    st.write(f"**Display:** {display}")
    st.write(f"**Screen Size:** {screen_size}")
    st.write(f"**Resolution:** {resolution}")
    st.write(f"**Operating System:** {operating_system}")
    st.write(f"**Dual SIM:** {dual_sim}")
    st.write(f"**Colour:** {colour}")
    st.write(f"**Android Version:** {android_version}")

    # Mobile Recommendation
    st.markdown("---")
    st.subheader("📱 Mobile Recommendation")
    st.write("Explore mobiles available in the dataset with the same price category.")

    recommended_mobiles = df[
        (df["Price_range"] == prediction[0]) &
        (df["Brand"] != brand)
    ]

    if not recommended_mobiles.empty:
        st.dataframe(
            recommended_mobiles[
                ["Brand", "Model_name", "Ram", "Storage", "Processor", "Price_range"]
            ].head(5),
            use_container_width=True
        )
    else:
        st.info("No similar mobiles found in this category.")