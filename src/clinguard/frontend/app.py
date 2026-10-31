
"""Streamlit application for ClinGuard AI."""

import requests
import pandas as pd
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
# 3. API helper functions
# --------------------------------------------------

def check_backend_connection():
    """Check whether FastAPI is running."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/",
            timeout=5,
        )
        response.raise_for_status()
        return True

    except requests.exceptions.RequestException:
        return False


def register_patient(patient_id):
    """Register a new patient through FastAPI."""

    return requests.post(
        f"{API_BASE_URL}/patients",
        json={"patient_id": patient_id},
        timeout=10,
    )


def get_patient_details(patient_id):
    """Retrieve patient information."""

    return requests.get(
        f"{API_BASE_URL}/patients/{patient_id}",
        timeout=10,
    )


def get_patient_readings(patient_id):
    """Retrieve all patient readings."""

    return requests.get(
        f"{API_BASE_URL}/patients/{patient_id}/readings",
        timeout=10,
    )


def get_patient_detections(patient_id):
    """Retrieve anomaly detection results."""

    return requests.get(
        f"{API_BASE_URL}/patients/{patient_id}/detections",
        timeout=10,
    )


def get_patient_verifications(patient_id):
    """Retrieve context verification results."""

    return requests.get(
        f"{API_BASE_URL}/patients/{patient_id}/verifications",
        timeout=10,
    )


# --------------------------------------------------
# 4. Sidebar navigation
# --------------------------------------------------

st.sidebar.title("ClinGuard AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Patient Management",
        "Patient Monitoring",
        "Anomaly & Verification",
    ],
)


# --------------------------------------------------
# 5. Home page
# --------------------------------------------------

if page == "Home":

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

    if check_backend_connection():

        st.success("FastAPI backend is connected!")

    else:

        st.error("FastAPI backend is not connected.")

        st.info(
            "Start FastAPI in another terminal."
        )


# --------------------------------------------------
# 6. Patient management page
# --------------------------------------------------

elif page == "Patient Management":

    st.title("Patient Management")

    st.write(
        "Register virtual patients and retrieve "
        "their stored monitoring records."
    )

    st.divider()

    # Patient registration

    st.header("Register New Patient")

    with st.form("patient_registration_form"):

        new_patient_id = st.text_input(
            "Enter Patient ID",
            placeholder="Example: DEMO002",
        )

        register_button = st.form_submit_button(
            "Register Patient"
        )

    if register_button:

        if not new_patient_id.strip():

            st.warning("Please enter a Patient ID.")

        else:

            try:

                response = register_patient(
                    new_patient_id.strip()
                )

                if response.status_code == 201:

                    st.success(
                        "Patient registered successfully!"
                    )

                    st.json(response.json())

                elif response.status_code == 409:

                    st.warning(
                        "This Patient ID already exists."
                    )

                else:

                    st.error(
                        f"Registration failed: {response.text}"
                    )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Backend connection error: {error}"
                )

    st.divider()

    # Patient retrieval

    st.header("Retrieve Patient Information")

    patient_id = st.text_input(
        "Enter an existing Patient ID",
        placeholder="Example: DEMO001",
        key="search_patient_id",
    )

    search_button = st.button(
        "Search Patient"
    )

    if search_button:

        if not patient_id.strip():

            st.warning("Please enter a Patient ID.")

        else:

            try:

                patient_response = get_patient_details(
                    patient_id.strip()
                )

                if patient_response.status_code == 200:

                    patient_data = patient_response.json()

                    st.success("Patient found!")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Patient ID",
                            patient_data["patient_id"],
                        )

                    with col2:

                        st.metric(
                            "Registration",
                            "Registered",
                        )

                    st.write("Patient information:")

                    st.json(patient_data)

                    # Retrieve historical readings

                    readings_response = get_patient_readings(
                        patient_id.strip()
                    )

                    if readings_response.status_code == 200:

                        readings = readings_response.json()

                        st.subheader(
                            "Historical Vital Readings"
                        )

                        if readings:

                            dataframe = pd.DataFrame(readings)

                            st.metric(
                                "Total Readings",
                                len(dataframe),
                            )

                            st.dataframe(
                                dataframe,
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                "No readings available for this patient."
                            )

                    else:

                        st.error(
                            "Unable to retrieve patient readings."
                        )

                elif patient_response.status_code == 404:

                    st.warning(
                        "Patient not found. "
                        "Please check the Patient ID."
                    )

                else:

                    st.error(
                        f"Request failed: {patient_response.text}"
                    )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Backend connection error: {error}"
                )


# --------------------------------------------------
# 7. Patient monitoring page
# --------------------------------------------------

elif page == "Patient Monitoring":

    st.title("Patient Monitoring")

    st.write(
        "View historical physiological readings "
        "and observe changes over time."
    )

    st.divider()

    st.subheader("Select Patient")

    with st.form("monitoring_patient_form"):

        monitoring_patient_id = st.text_input(
            "Enter Patient ID",
            placeholder="Example: DEMO001",
        )

        monitor_button = st.form_submit_button(
            "Load Patient Monitoring"
        )

    if monitor_button:

        if not monitoring_patient_id.strip():

            st.warning("Please enter a Patient ID.")

        else:

            st.session_state["monitoring_patient_id"] = (
                monitoring_patient_id.strip()
            )

    if "monitoring_patient_id" in st.session_state:

        selected_patient = st.session_state[
            "monitoring_patient_id"
        ]

        try:

            patient_response = get_patient_details(
                selected_patient
            )

            if patient_response.status_code == 404:

                st.warning("Patient not found.")

            elif patient_response.status_code != 200:

                st.error(
                    f"Unable to retrieve patient information: "
                    f"{patient_response.text}"
                )

            else:

                readings_response = get_patient_readings(
                    selected_patient
                )

                if readings_response.status_code != 200:

                    st.error(
                        "Unable to retrieve historical readings."
                    )

                else:

                    readings = readings_response.json()

                    st.divider()

                    st.subheader(
                        f"Monitoring Overview — {selected_patient}"
                    )

                    if not readings:

                        st.info(
                            "No vital readings are available "
                            "for this patient yet."
                        )

                    else:

                        dataframe = pd.DataFrame(readings)

                        required_columns = [
                            "timestamp",
                            "heart_rate",
                            "spo2",
                            "systolic_bp",
                            "diastolic_bp",
                            "respiratory_rate",
                            "temperature",
                        ]

                        missing_columns = [
                            column
                            for column in required_columns
                            if column not in dataframe.columns
                        ]

                        if missing_columns:

                            st.error(
                                "The API response is missing "
                                f"these fields: {missing_columns}"
                            )

                        else:

                            dataframe["timestamp"] = pd.to_datetime(
                                dataframe["timestamp"],
                                errors="coerce",
                            )

                            dataframe = dataframe.dropna(
                                subset=["timestamp"]
                            )

                            dataframe = dataframe.sort_values(
                                "timestamp"
                            )

                            vital_columns = [
                                "heart_rate",
                                "spo2",
                                "systolic_bp",
                                "diastolic_bp",
                                "respiratory_rate",
                                "temperature",
                            ]

                            for column in vital_columns:

                                dataframe[column] = pd.to_numeric(
                                    dataframe[column],
                                    errors="coerce",
                                )

                            if dataframe.empty:

                                st.warning(
                                    "No valid timestamped readings "
                                    "are available."
                                )

                            else:

                                latest = dataframe.iloc[-1]

                                st.caption(
                                    "Latest recorded reading: "
                                    f"{latest['timestamp']}"
                                )

                                st.subheader(
                                    "Latest Vital-Sign Readings"
                                )

                                col1, col2, col3 = st.columns(3)

                                with col1:

                                    st.metric(
                                        "Heart Rate",
                                        (
                                            f"{latest['heart_rate']:.1f} bpm"
                                            if pd.notna(latest["heart_rate"])
                                            else "Unavailable"
                                        ),
                                    )

                                with col2:

                                    st.metric(
                                        "SpO₂",
                                        (
                                            f"{latest['spo2']:.1f}%"
                                            if pd.notna(latest["spo2"])
                                            else "Unavailable"
                                        ),
                                    )

                                with col3:

                                    st.metric(
                                        "Respiratory Rate",
                                        (
                                            f"{latest['respiratory_rate']:.1f}"
                                            if pd.notna(
                                                latest["respiratory_rate"]
                                            )
                                            else "Unavailable"
                                        ),
                                    )

                                col4, col5, col6 = st.columns(3)

                                with col4:

                                    if (
                                        pd.notna(latest["systolic_bp"])
                                        and pd.notna(latest["diastolic_bp"])
                                    ):

                                        bp_value = (
                                            f"{latest['systolic_bp']:.0f}/"
                                            f"{latest['diastolic_bp']:.0f}"
                                        )

                                    else:

                                        bp_value = "Unavailable"

                                    st.metric(
                                        "Blood Pressure",
                                        bp_value,
                                    )

                                with col5:

                                    st.metric(
                                        "Temperature",
                                        (
                                            f"{latest['temperature']:.1f} °C"
                                            if pd.notna(
                                                latest["temperature"]
                                            )
                                            else "Unavailable"
                                        ),
                                    )

                                with col6:

                                    st.metric(
                                        "Total Readings",
                                        len(dataframe),
                                    )

                                st.divider()

                                # Heart-rate chart

                                st.subheader("Heart Rate Trend")

                                heart_rate_data = dataframe[
                                    ["timestamp", "heart_rate"]
                                ].dropna().set_index("timestamp")

                                if not heart_rate_data.empty:

                                    st.line_chart(
                                        heart_rate_data,
                                        y="heart_rate",
                                    )

                                else:

                                    st.info(
                                        "No heart-rate data available."
                                    )

                                # SpO2 chart

                                st.subheader("SpO₂ Trend")

                                spo2_data = dataframe[
                                    ["timestamp", "spo2"]
                                ].dropna().set_index("timestamp")

                                if not spo2_data.empty:

                                    st.line_chart(
                                        spo2_data,
                                        y="spo2",
                                    )

                                else:

                                    st.info(
                                        "No SpO₂ data available."
                                    )

                                # Blood pressure chart

                                st.subheader("Blood Pressure Trend")

                                bp_data = dataframe[
                                    [
                                        "timestamp",
                                        "systolic_bp",
                                        "diastolic_bp",
                                    ]
                                ].dropna(
                                    subset=[
                                        "systolic_bp",
                                        "diastolic_bp",
                                    ],
                                    how="all",
                                ).set_index("timestamp")

                                if not bp_data.empty:

                                    st.line_chart(bp_data)

                                else:

                                    st.info(
                                        "No blood-pressure data available."
                                    )

                                # Respiratory rate chart

                                st.subheader("Respiratory Rate Trend")

                                respiratory_data = dataframe[
                                    ["timestamp", "respiratory_rate"]
                                ].dropna().set_index("timestamp")

                                if not respiratory_data.empty:

                                    st.line_chart(
                                        respiratory_data,
                                        y="respiratory_rate",
                                    )

                                else:

                                    st.info(
                                        "No respiratory-rate data available."
                                    )

                                # Temperature chart

                                st.subheader("Body Temperature Trend")

                                temperature_data = dataframe[
                                    ["timestamp", "temperature"]
                                ].dropna().set_index("timestamp")

                                if not temperature_data.empty:

                                    st.line_chart(
                                        temperature_data,
                                        y="temperature",
                                    )

                                else:

                                    st.info(
                                        "No temperature data available."
                                    )

                                st.divider()

                                st.subheader(
                                    "Complete Historical Readings"
                                )

                                st.dataframe(
                                    dataframe,
                                    use_container_width=True,
                                )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Backend connection error: {error}"
            )


# --------------------------------------------------
# 8. Anomaly and verification dashboard
# --------------------------------------------------

elif page == "Anomaly & Verification":

    st.title("Anomaly & Verification Dashboard")

    st.write(
        "Review the outputs of the anomaly detection "
        "models and the context verification engine."
    )

    st.info(
        "This dashboard displays stored model and verification "
        "results. It does not independently generate predictions."
    )

    st.divider()

    # Patient selection

    with st.form("anomaly_patient_form"):

        anomaly_patient_id = st.text_input(
            "Enter Patient ID",
            placeholder="Example: DEMO001",
        )

        load_anomaly_button = st.form_submit_button(
            "Load Detection Results"
        )

    if load_anomaly_button:

        if not anomaly_patient_id.strip():

            st.warning("Please enter a Patient ID.")

        else:

            st.session_state["anomaly_patient_id"] = (
                anomaly_patient_id.strip()
            )

    if "anomaly_patient_id" in st.session_state:

        selected_patient = st.session_state[
            "anomaly_patient_id"
        ]

        try:

            # Verify patient exists

            patient_response = get_patient_details(
                selected_patient
            )

            if patient_response.status_code == 404:

                st.warning("Patient not found.")

            elif patient_response.status_code != 200:

                st.error(
                    f"Unable to retrieve patient: "
                    f"{patient_response.text}"
                )

            else:

                # Retrieve detection results

                detection_response = get_patient_detections(
                    selected_patient
                )

                # Retrieve verification results

                verification_response = get_patient_verifications(
                    selected_patient
                )

                if detection_response.status_code != 200:

                    st.error(
                        "Unable to retrieve detection results: "
                        f"{detection_response.text}"
                    )

                elif verification_response.status_code != 200:

                    st.error(
                        "Unable to retrieve verification results: "
                        f"{verification_response.text}"
                    )

                else:

                    detection_data = detection_response.json()

                    verification_data = verification_response.json()

                    detections = detection_data.get(
                        "detections", []
                    )

                    verifications = verification_data.get(
                        "verifications", []
                    )

                    st.divider()

                    st.subheader(
                        f"Analysis Results — {selected_patient}"
                    )

                    if not detections and not verifications:

                        st.info(
                            "No detection or verification results "
                            "are available for this patient."
                        )

                    else:

                        detection_df = pd.DataFrame(detections)

                        verification_df = pd.DataFrame(verifications)

                        # ----------------------------------
                        # Summary metrics
                        # ----------------------------------

                        st.subheader("Detection Summary")

                        total_records = max(
                            detection_data.get("total_results", 0),
                            verification_data.get("total_results", 0),
                        )

                        statistical_count = 0
                        isolation_count = 0
                        hybrid_or_count = 0
                        hybrid_and_count = 0

                        if not detection_df.empty:

                            statistical_count = int(
                                detection_df[
                                    "statistical_anomaly_flag"
                                ].fillna(False).astype(bool).sum()
                            )

                            isolation_count = int(
                                detection_df[
                                    "isolation_forest_anomaly_flag"
                                ].fillna(False).astype(bool).sum()
                            )

                            hybrid_or_count = int(
                                detection_df[
                                    "hybrid_or_flag"
                                ].fillna(False).astype(bool).sum()
                            )

                            hybrid_and_count = int(
                                detection_df[
                                    "hybrid_and_flag"
                                ].fillna(False).astype(bool).sum()
                            )

                        quality_count = 0
                        supported_count = 0
                        review_count = 0
                        no_alert_count = 0

                        if not verification_df.empty:

                            statuses = verification_df[
                                "verification_status"
                            ]

                            quality_count = int(
                                (
                                    statuses
                                    == "quality_review_without_model_alert"
                                ).sum()
                            )

                            supported_count = int(
                                (
                                    statuses
                                    == "supported_anomaly_candidate"
                                ).sum()
                            )

                            review_count = int(
                                (
                                    statuses
                                    == "candidate_needs_review"
                                ).sum()
                            )

                            no_alert_count = int(
                                (
                                    statuses
                                    == "no_model_alert"
                                ).sum()
                            )

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:

                            st.metric(
                                "Total Records",
                                total_records,
                            )

                        with col2:

                            st.metric(
                                "Hybrid OR Flags",
                                hybrid_or_count,
                            )

                        with col3:

                            st.metric(
                                "Hybrid AND Flags",
                                hybrid_and_count,
                            )

                        with col4:

                            st.metric(
                                "Quality Reviews",
                                quality_count,
                            )

                        # ----------------------------------
                        # Detection model comparison
                        # ----------------------------------

                        st.divider()

                        st.subheader(
                            "Anomaly Detection Model Comparison"
                        )

                        model_summary = pd.DataFrame(
                            {
                                "Detection Model": [
                                    "Statistical",
                                    "Isolation Forest",
                                    "Hybrid OR",
                                    "Hybrid AND",
                                ],
                                "Flagged Records": [
                                    statistical_count,
                                    isolation_count,
                                    hybrid_or_count,
                                    hybrid_and_count,
                                ],
                            }
                        )

                        st.bar_chart(
                            model_summary,
                            x="Detection Model",
                            y="Flagged Records",
                        )

                        st.dataframe(
                            model_summary,
                            hide_index=True,
                            use_container_width=True,
                        )

                        # ----------------------------------
                        # Verification status distribution
                        # ----------------------------------

                        st.divider()

                        st.subheader(
                            "Verification Status Distribution"
                        )

                        verification_summary = pd.DataFrame(
                            {
                                "Verification Status": [
                                    "Supported anomaly candidate",
                                    "Candidate needs review",
                                    "Quality review without model alert",
                                    "No model alert",
                                ],
                                "Number of Records": [
                                    supported_count,
                                    review_count,
                                    quality_count,
                                    no_alert_count,
                                ],
                            }
                        )

                        st.bar_chart(
                            verification_summary,
                            x="Verification Status",
                            y="Number of Records",
                        )

                        st.dataframe(
                            verification_summary,
                            hide_index=True,
                            use_container_width=True,
                        )

                        # ----------------------------------
                        # Detailed detection records
                        # ----------------------------------

                        st.divider()

                        st.subheader(
                            "Individual Detection Results"
                        )

                        if not detection_df.empty:

                            detection_display = detection_df.copy()

                            if "isolation_forest_score" in detection_display.columns:

                                detection_display[
                                    "isolation_forest_score"
                                ] = detection_display[
                                    "isolation_forest_score"
                                ].apply(
                                    lambda value:
                                    "Unavailable"
                                    if pd.isna(value)
                                    else value
                                )

                            st.dataframe(
                                detection_display,
                                hide_index=True,
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                "No detection records are available."
                            )

                        # ----------------------------------
                        # Verification history
                        # ----------------------------------

                        st.divider()

                        st.subheader(
                            "Context Verification History"
                        )

                        if not verification_df.empty:

                            verification_display = verification_df[
                                [
                                    "reading_id",
                                    "timestamp",
                                    "verification_status",
                                    "verification_reason",
                                    "context_issues",
                                ]
                            ].copy()

                            st.dataframe(
                                verification_display,
                                hide_index=True,
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                "No verification records are available."
                            )

                        # ----------------------------------
                        # Evidence-based explanation
                        # ----------------------------------

                        st.divider()

                        st.subheader(
                            "Evidence-Based Explanation"
                        )

                        if not verification_df.empty:

                            explanation_options = (
                                verification_df[
                                    "reading_id"
                                ].tolist()
                            )

                            selected_reading = st.selectbox(
                                "Select Reading ID",
                                explanation_options,
                            )

                            selected_explanation = verification_df[
                                verification_df["reading_id"]
                                == selected_reading
                            ].iloc[0]

                            st.write(
                                "**Verification Status:**",
                                selected_explanation[
                                    "verification_status"
                                ],
                            )

                            st.write(
                                "**Context Issues:**",
                                selected_explanation[
                                    "context_issues"
                                ],
                            )

                            st.write(
                                "**Verification Reason:**",
                                selected_explanation[
                                    "verification_reason"
                                ],
                            )

                            st.write("**Explanation:**")

                            st.info(
                                selected_explanation[
                                    "explainable_reason"
                                ]
                            )

                        else:

                            st.info(
                                "No explanations are available."
                            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Backend connection error: {error}"
            )


# --------------------------------------------------
# 9. Safety disclaimer
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "ClinGuard AI is a research prototype. "
    "It is not a medical diagnostic system."
)
