
"""Main Streamlit application for ClinGuard AI."""

import requests
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ClinGuard AI",
    page_icon="🏥",
    layout="wide",
)


# --------------------------------------------------
# 2. Backend configuration
# --------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# 3. Backend connection function
# --------------------------------------------------

def check_backend_connection():
    """Check whether the FastAPI backend is running."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/",
            timeout=5,
        )

        response.raise_for_status()

        return True, response.json()

    except requests.exceptions.RequestException as error:
        return False, str(error)


# --------------------------------------------------
# 4. Application interface
# --------------------------------------------------

st.title("ClinGuard AI")

st.subheader(
    "Intelligent Patient Monitoring and "
    "Context-Aware Anomaly Detection"
)

st.write(
    "A research prototype for patient-adaptive "
    "physiological monitoring and explainable "
    "anomaly analysis."
)

st.divider()

st.header("Backend Connection")

connected, backend_response = check_backend_connection()

if connected:
    st.success("FastAPI backend is connected!")

    st.write("Backend response:")

    st.json(backend_response)

else:
    st.error("FastAPI backend is not connected.")

    st.warning(
        "Start the FastAPI server in another terminal "
        "and refresh this page."
    )

    st.caption(f"Connection details: {backend_response}")

st.divider()

st.caption(
    "ClinGuard AI is an educational research prototype. "
    "It is not a medical diagnostic system."
)
