from pathlib import Path
from urllib.request import urlopen
import inspect
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Configure the page before displaying other Streamlit elements.
st.set_page_config(
    page_title="The Backpack Doctor",
    page_icon="🎒",
    layout="wide",
)

# Apply consistent spacing and styling to the dashboard.
st.markdown(
    """
    <style>
    /* Keep the content readable on wide screens. */
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Give headings a consistent appearance. */
    h1 {
        letter-spacing: -0.035em;
        color: #183044;
    }

    h2, h3 {
        color: #183044;
        letter-spacing: -0.02em;
    }

    /* Style the existing bordered result containers. */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 16px;
    }

    /* Give metrics more breathing room. */
    [data-testid="stMetric"] {
        padding: 0.75rem 0;
    }

    /* Make the sidebar submit button fill its available width. */
    [data-testid="stSidebar"]
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    /* Hide numeric increment/decrement buttons. */
    [data-testid="stSidebar"]
    [data-testid="stNumberInput"] button {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Project files exported by the modelling notebook.
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "triage_model.joblib"

COUNTY_PATH = (
    PROJECT_DIR / "data" / "processed" / "rq2_county_ppv.csv"
)

SUBGROUP_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "rq3_subgroup_performance.csv"
)

LOCAL_BOUNDARIES = (
    PROJECT_DIR / "data" / "geo" / "counties.geojson"
)

BOUNDARY_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "master/geojson-counties-fips.json"
)


STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona",
    "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}


# File versions invalidate caches when project exports change.
def file_version(path):
    information = path.stat()
    return information.st_mtime_ns, information.st_size


# Load the fitted model once instead of on every interaction.
@st.cache_resource(show_spinner=False)
def load_model_bundle(path, version):
    bundle = joblib.load(path)

    required = {"model", "features", "threshold"}
    missing = required - set(bundle)

    if missing:
        raise ValueError(
            "Missing model information: "
            + ", ".join(sorted(missing))
        )

    if not 0 <= float(bundle["threshold"]) <= 1:
        raise ValueError(
            "The model threshold must be between 0 and 1."
        )

    return bundle


# Load and prepare the county results once per file version.
@st.cache_data(show_spinner=False)
def load_county_results(path, version):
    counties = pd.read_csv(
        path,
        dtype={"fips": "string"},
    )

    required = {
        "fips",
        "locationname",
        "stateabbr",
        "diagnosed_diabetes_pct",
        "screening_prevalence",
        "PPV",
        "NPV",
    }

    missing = required - set(counties.columns)

    if missing:
        raise ValueError(
            "Missing county columns: "
            + ", ".join(sorted(missing))
        )

    # Preserve leading zeros in county identifiers.
    counties["fips"] = (
        counties["fips"]
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    counties["stateabbr"] = (
        counties["stateabbr"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    for column in [
        "diagnosed_diabetes_pct",
        "screening_prevalence",
        "PPV",
        "NPV",
    ]:
        counties[column] = pd.to_numeric(
            counties[column],
            errors="coerce",
        )

    original_count = len(counties)

    counties = counties.dropna(
        subset=["fips", "locationname", "stateabbr", "PPV"]
    ).copy()

    excluded_count = original_count - len(counties)

    if counties.empty:
        raise ValueError("No usable county results were found.")

    if not counties["fips"].str.fullmatch(r"\d{5}").all():
        raise ValueError(
            "County FIPS identifiers must contain five digits."
        )

    if counties["fips"].duplicated().any():
        raise ValueError(
            "The county table contains duplicate FIPS identifiers."
        )

    if not counties["PPV"].between(0, 1).all():
        raise ValueError(
            "PPV must be stored as a fraction between 0 and 1."
        )

    counties["PPV_percent"] = counties["PPV"] * 100

    counties["county_label"] = (
        counties["locationname"] + ", " + counties["stateabbr"]
    )

    return counties, excluded_count


# Preserve subgroup results exactly as exported.
@st.cache_data(show_spinner=False)
def load_subgroup_results(path, version):
    return pd.read_csv(path)


# Prefer archived boundaries when they are available.
def get_boundary_source():
    if LOCAL_BOUNDARIES.exists():
        return (
            str(LOCAL_BOUNDARIES),
            file_version(LOCAL_BOUNDARIES),
        )

    return BOUNDARY_URL, "remote"


# The shared boundary index is only read, never modified.
@st.cache_resource(show_spinner=False)
def load_boundary_index(source, version):
    if source.startswith("https://"):
        with urlopen(source, timeout=20) as response:
            boundaries = json.load(response)
    else:
        with open(source, encoding="utf-8") as handle:
            boundaries = json.load(handle)

    if boundaries.get("type") != "FeatureCollection":
        raise ValueError(
            "The map requires a GeoJSON FeatureCollection."
        )

    return {
        str(feature["id"]): feature
        for feature in boundaries["features"]
    }


# Cache the base map independently of the selected county outline.
@st.cache_data(show_spinner=False, max_entries=16)
def build_base_map(
    map_data,
    source_version,
    national_view,
    _boundary_index,
):
    # Send only the county geometries used in the current view.
    geometry = {
        "type": "FeatureCollection",
        "features": [
            _boundary_index[fips]
            for fips in map_data["fips"]
        ],
    }

    figure = px.choropleth(
        map_data,
        geojson=geometry,
        featureidkey="id",
        locations="fips",
        color="PPV_percent",
        scope="usa",
        color_continuous_scale="Blues",
        range_color=(0, 100),
        hover_name="county_label",
        hover_data={
            "fips": True,
            "PPV_percent": ":.1f",
            "diagnosed_diabetes_pct": ":.1f",
        },
        labels={
            "fips": "County FIPS",
            "PPV_percent": "Estimated PPV (%)",
            "diagnosed_diabetes_pct": (
                "Diagnosed diabetes prevalence (%)"
            ),
        },
    )

    figure.update_traces(
        marker_line_width=0.25,
        marker_line_color="#FFFFFF",
    )

    if not national_view:
        figure.update_geos(fitbounds="locations")

    figure.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showlakes=False,
    )

    figure.update_layout(
        height=510,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="PPV (%)",
            thickness=12,
            len=0.7,
        ),
        uirevision=(
            "US"
            if national_view
            else str(map_data["stateabbr"].iloc[0])
        ),
    )

    return figure


# Support both older and newer Streamlit chart-width arguments.
def display_map(figure):
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(
            figure,
            width="stretch",
            key="county_map",
        )
    else:
        st.plotly_chart(
            figure,
            use_container_width=True,
            key="county_map",
        )


# Panel 1: collect measurements and display the submitted prediction.
def show_patient_prediction(bundle, model_version):
    # Clear a stored prediction if the fitted model file changes.
    if st.session_state.get("model_version") != model_version:
        st.session_state.pop("patient_prediction", None)
        st.session_state["model_version"] = model_version

    with st.sidebar:
        st.header("Patient measurements")
        st.caption(
            "Review all measurements before calculating the prediction."
        )

        with st.form("patient_form"):
            age = st.number_input(
                "Age (years)",
                min_value=20,
                max_value=100,
                value=30,
                step=1,
            )

            sex = st.selectbox(
                "Sex",
                ("Male", "Female"),
            )

            waist = st.number_input(
                "Waist circumference (cm)",
                min_value=40.0,
                max_value=200.0,
                value=90.0,
                step=0.5,
            )

            systolic = st.number_input(
                "Systolic blood pressure (mmHg)",
                min_value=60.0,
                max_value=260.0,
                value=120.0,
                step=1.0,
            )

            diastolic = st.number_input(
                "Diastolic blood pressure (mmHg)",
                min_value=30.0,
                max_value=160.0,
                value=80.0,
                step=1.0,
            )

            submitted = st.form_submit_button(
                "Calculate prediction",
                type="primary",
            )

    if submitted:
        # Clear the previous result before validating a new submission.
        st.session_state.pop("patient_prediction", None)

        patient = pd.DataFrame([{
            "age": age,
            "female": int(sex == "Female"),
            "systolic": systolic,
            "diastolic": diastolic,
            "waist": waist,
        }])

        missing_features = (
            set(bundle["features"]) - set(patient.columns)
        )

        if missing_features:
            st.error(
                "Required model inputs are missing: "
                + ", ".join(sorted(missing_features))
            )

        elif systolic <= diastolic:
            st.error(
                "Please check the blood pressure: "
                "systolic pressure must exceed diastolic pressure."
            )

        else:
            try:
                model = bundle["model"]

                positive_class_index = list(
                    model.classes_
                ).index(1)

                risk = float(
                    model.predict_proba(
                        patient[bundle["features"]]
                    )[0, positive_class_index]
                )

                if not 0 <= risk <= 1:
                    raise ValueError(
                        "The model returned an invalid probability."
                    )

            except Exception as error:
                st.error(
                    f"The prediction could not be calculated: {error}"
                )

            else:
                # Store only the submitted measurements and their result.
                st.session_state["patient_prediction"] = {
                    "risk": risk,
                    "threshold": float(bundle["threshold"]),
                    "measurements": patient.iloc[0].to_dict(),
                    "sex": sex,
                }

    st.subheader("1 · Patient prediction")

    result = st.session_state.get("patient_prediction")

    if result is None:
        st.info(
            "Enter the measurements in the sidebar and "
            "click Calculate prediction."
        )
        return

    risk = result["risk"]
    threshold = result["threshold"]
    measurements = result["measurements"]

    with st.container(border=True):
        risk_column, threshold_column, flag_column = st.columns(
            [1.3, 1, 1.4]
        )

        risk_column.metric(
            "Estimated probability of A1C ≥ 5.7%",
            f"{risk:.1%}",
        )

        threshold_column.metric(
            "Screening threshold",
            f"{threshold:.1%}",
        )

        flag_column.metric(
            "Model screening flag",
            "Positive" if risk >= threshold else "Below threshold",
        )

        st.caption(
            f"Last submitted patient: "
            f"age {measurements['age']:.0f} · "
            f"{result['sex']} · "
            f"waist {measurements['waist']:.1f} cm · "
            f"BP {measurements['systolic']:.0f}/"
            f"{measurements['diastolic']:.0f} mmHg. "
            "Submit again after changing measurements."
        )

        if risk >= threshold:
            st.warning(
                "Positive screening flag: the model suggests "
                "referral for a laboratory A1C test."
            )
        else:
            st.info(
                "Below the screening threshold: the model "
                "does not flag this patient for referral."
            )

    st.warning(
        "This is a screening estimate, not a diagnosis. "
        "A result below the threshold does not rule out "
        "prediabetes or diabetes."
    )


# Panel 2: map and numerical results from the county reliability export.
def show_county_reliability(counties, excluded_count):
    st.divider()
    st.subheader("2 · Regional reliability")

    st.write(
        "Explore how often a positive screening flag is expected "
        "to be correct in the selected county."
    )

    if excluded_count:
        st.caption(
            f"{excluded_count} rows with missing identifiers or PPV "
            "were excluded. Missing values are not treated as zero."
        )

    state_column, county_column = st.columns([1, 2])

    state_options = sorted(
        counties["stateabbr"].unique(),
        key=lambda code: STATE_NAMES.get(code, code),
    )

    selected_state = state_column.selectbox(
        "State",
        options=state_options,
        format_func=lambda code: STATE_NAMES.get(code, code),
        key="rq2_state",
    )

    state_counties = counties[
        counties["stateabbr"] == selected_state
    ].copy()

    county_labels = state_counties.set_index(
        "fips"
    )["locationname"].to_dict()

    selected_fips = county_column.selectbox(
        "County or county equivalent",
        options=sorted(
            county_labels,
            key=lambda code: (county_labels[code], code),
        ),
        format_func=lambda code: (
            f"{county_labels[code]} ({code})"
        ),
        key=f"rq2_county_{selected_state}",
    )

    selected_county = state_counties[
        state_counties["fips"] == selected_fips
    ].iloc[0]

    with st.container(border=True):
        map_column, detail_column = st.columns([2.4, 1])

        with detail_column:
            state_name = STATE_NAMES.get(
                selected_state,
                selected_state,
            )

            st.markdown(
                f"**{selected_county['locationname']}, "
                f"{state_name}**"
            )

            ppv = float(selected_county["PPV"])

            st.metric(
                "Estimated PPV of a positive flag",
                f"{ppv:.1%}",
            )

            st.info(
                f"Under the RQ2 assumptions, about "
                f"{ppv * 100:.0f} out of 100 people flagged "
                "positive would have A1C ≥ 5.7%."
            )

            st.caption(
                "County PPV describes positive screening flags "
                "in the region. It does not replace the "
                "patient's individual risk estimate."
            )

            with st.expander("Prevalence assumptions"):
                diagnosed_pct = selected_county[
                    "diagnosed_diabetes_pct"
                ]

                screening_prevalence = selected_county[
                    "screening_prevalence"
                ]

                st.metric(
                    "Diagnosed diabetes · PLACES",
                    (
                        f"{diagnosed_pct:.1f}%"
                        if pd.notna(diagnosed_pct)
                        else "Unavailable"
                    ),
                )

                st.metric(
                    "Assumed prevalence · A1C ≥ 5.7%",
                    (
                        f"{screening_prevalence:.1%}"
                        if pd.notna(screening_prevalence)
                        and 0 <= screening_prevalence <= 1
                        else "Unavailable"
                    ),
                )

                st.caption(
                    "The target prevalence is derived in RQ2. "
                    "It is not directly measured by PLACES."
                )

        with map_column:
            national_view = st.checkbox(
                "US overview (slower)",
                value=False,
                key="national_map",
            )

            show_map = st.checkbox(
                "Show interactive map",
                value=True,
                key="show_map",
            )

            if show_map:
                try:
                    source, version = get_boundary_source()

                    with st.spinner("Preparing county map…"):
                        boundary_index = load_boundary_index(
                            source,
                            version,
                        )

                        view = (
                            counties
                            if national_view
                            else state_counties
                        )

                        matched = view[
                            view["fips"].isin(boundary_index)
                        ].copy()

                        unmatched = view[
                            ~view["fips"].isin(boundary_index)
                        ]

                        if matched.empty:
                            st.info(
                                "No matching boundaries are "
                                "available for this map view."
                            )

                        else:
                            # Cache the base map without the selection.
                            figure = build_base_map(
                                matched[[
                                    "fips",
                                    "stateabbr",
                                    "PPV_percent",
                                    "county_label",
                                    "diagnosed_diabetes_pct",
                                ]],
                                (source, version),
                                national_view,
                                boundary_index,
                            )

                            # Include only one geometry in the outline.
                            if selected_fips in boundary_index:
                                figure.add_trace(
                                    go.Choropleth(
                                        geojson={
                                            "type": "FeatureCollection",
                                            "features": [
                                                boundary_index[
                                                    selected_fips
                                                ]
                                            ],
                                        },
                                        featureidkey="id",
                                        locations=[selected_fips],
                                        z=[0],
                                        zmin=0,
                                        zmax=1,
                                        colorscale=[
                                            [0, "rgba(0,0,0,0)"],
                                            [1, "rgba(0,0,0,0)"],
                                        ],
                                        marker_line_color="#E87924",
                                        marker_line_width=2.5,
                                        showscale=False,
                                        hoverinfo="skip",
                                    )
                                )

                            display_map(figure)

                    if not unmatched.empty:
                        st.caption(
                            f"{len(unmatched)} counties in this "
                            "view have no matching boundary. "
                            "Their results remain available "
                            "through the county selector."
                        )

                        with st.expander(
                            "Counties without matching boundaries"
                        ):
                            st.dataframe(
                                unmatched[[
                                    "fips",
                                    "locationname",
                                    "stateabbr",
                                ]],
                                hide_index=True,
                            )

                    st.caption(
                        "Hover to inspect a county. The orange outline "
                        "marks the dropdown selection. Map clicks "
                        "do not change the selected county."
                    )

                except Exception as error:
                    st.info(
                        "The map is temporarily unavailable. "
                        "County results remain available on the right."
                    )

                    with st.expander("Map loading details"):
                        st.code(str(error))

    with st.expander("RQ2 assumptions and additional results"):
        st.write(
            "County PPV combines evaluated sensitivity and specificity "
            "with an assumed local target prevalence. It assumes model "
            "performance transfers to the county. This is not a local "
            "validation of the model."
        )

        st.write(
            "PLACES measures diagnosed diabetes. The RQ2 analysis "
            "converts this to an assumed prevalence of A1C ≥ 5.7%."
        )

        additional_columns = [
            column
            for column in counties.columns
            if column.startswith(("PPV_", "NPV_"))
        ]

        if additional_columns:
            st.caption(
                "Additional estimates retain their original "
                "export labels and units."
            )

            st.dataframe(
                selected_county[additional_columns]
                .rename("Exported value")
                .to_frame()
            )


# Panel 3: show the exported evaluation results without guessing its schema.
# Panel 3: match the submitted patient to separately evaluated groups.
def show_subgroup_performance():
    st.divider()
    st.subheader("3 · Performance for comparable patient groups")

    if not SUBGROUP_PATH.exists():
        st.warning("The subgroup performance export was not found.")
        return

    try:
        subgroups = load_subgroup_results(
            str(SUBGROUP_PATH),
            file_version(SUBGROUP_PATH),
        ).copy()

    except Exception as error:
        st.error(f"Subgroup results could not be loaded: {error}")
        return

    required_columns = {
        "dimension",
        "group",
        "n",
        "positives",
        "sensitivity",
        "sens low",
        "sens high",
        "specificity",
        "PPV",
        "missed cases (FN)",
        "interpretable",
    }

    missing_columns = required_columns - set(subgroups.columns)

    if missing_columns:
        st.error(
            "Missing subgroup columns: "
            + ", ".join(sorted(missing_columns))
        )
        return

    # Clean labels without changing the exported group definitions.
    for column in ["dimension", "group"]:
        subgroups[column] = (
            subgroups[column].astype("string").str.strip()
        )

    if subgroups.duplicated(["dimension", "group"]).any():
        st.error(
            "Multiple records exist for the same dimension and group. "
            "Check the subgroup export."
        )
        return

    # Convert exported counts and metrics to numeric values.
    numeric_columns = [
        "n",
        "positives",
        "sensitivity",
        "sens low",
        "sens high",
        "specificity",
        "PPV",
        "missed cases (FN)",
    ]

    for column in numeric_columns:
        subgroups[column] = pd.to_numeric(
            subgroups[column],
            errors="coerce",
        )

    # Parse flags explicitly: the string 'false' must not become True.
    def is_interpretable(value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1"}
        return pd.notna(value) and value == True

    subgroups["interpretable"] = subgroups[
        "interpretable"
    ].map(is_interpretable)

    # Format unavailable or invalid metrics without inventing values.
    def percentage(value):
        if pd.notna(value) and 0 <= value <= 1:
            return f"{value:.1%}"
        return "Not available"

    def count(value):
        if pd.notna(value):
            return f"{value:,.0f}"
        return "Not available"

    result = st.session_state.get("patient_prediction")

    if result is None:
        st.info(
            "Calculate a patient prediction first to display "
            "the matching sex and age groups."
        )

        with st.expander("View all exported subgroup results"):
            st.dataframe(subgroups, hide_index=True)

        return

    # Use the last submitted patient, matching the prediction panel.
    measurements = result["measurements"]
    patient_age = int(measurements["age"])
    patient_sex = result["sex"]

    # Use the exact age-band boundaries from the evaluation export.
    if patient_age < 20:
        st.warning(
            "The exported age groups cover adults aged 20 and older."
        )
        return
    elif patient_age < 40:
        patient_age_band = "20-39"
    elif patient_age < 55:
        patient_age_band = "40-54"
    elif patient_age < 70:
        patient_age_band = "55-69"
    else:
        patient_age_band = "70+"

    st.caption(
        f"Matching the last submitted patient: "
        f"{patient_sex}, age {patient_age} "
        f"(age group {patient_age_band})."
    )

    # Race/ethnicity is optional and is not passed to the risk model.
    race_options = (
        subgroups.loc[
            subgroups["dimension"] == "race",
            "group",
        ]
        .dropna()
        .tolist()
    )

    selected_race = st.selectbox(
        "Race/ethnicity (optional, self-reported)",
        options=["Not specified"] + race_options,
        key="rq3_race",
    )

    st.caption(
        "This optional selection changes the group comparison only. "
        "It does not change the patient's model prediction."
    )

    comparisons = [
        ("Sex", "sex", patient_sex),
        ("Age", "age_band", patient_age_band),
        ("Race/ethnicity", "race", selected_race),
    ]

    columns = st.columns(3)

    for column, (title, dimension, group) in zip(
        columns,
        comparisons,
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")

                if group == "Not specified":
                    st.info(
                        "Select a race/ethnicity above "
                        "to display this comparison."
                    )
                    continue

                st.write(group)

                matching_rows = subgroups[
                    (subgroups["dimension"] == dimension)
                    & (subgroups["group"] == group)
                ]

                if matching_rows.empty:
                    st.warning(
                        "No evaluation result is available "
                        "for this group."
                    )
                    continue

                row = matching_rows.iloc[0]

                st.caption(
                    f"Evaluated participants: {count(row['n'])} · "
                    f"Positive cases: {count(row['positives'])}"
                )

                # Respect the interpretation flag from the analysis.
                if not row["interpretable"]:
                    st.warning(
                        "The analysis marks this group as unsuitable "
                        "for performance interpretation. "
                        "Only sample counts are shown here."
                    )
                    continue

                st.metric(
                    "Sensitivity",
                    percentage(row["sensitivity"]),
                )

                low = row["sens low"]
                high = row["sens high"]

                if (
                    pd.notna(low)
                    and pd.notna(high)
                    and 0 <= low <= high <= 1
                ):
                    st.caption(
                        f"Exported sensitivity interval: "
                        f"{low:.1%}–{high:.1%}"
                    )
                else:
                    st.caption(
                        "Sensitivity interval not available."
                    )

                st.metric(
                    "Missed cases in evaluation",
                    count(row["missed cases (FN)"]),
                )

                st.write(
                    "**Specificity:** "
                    + percentage(row["specificity"])
                )

                st.write(
                    "**PPV in the evaluation sample:** "
                    + percentage(row["PPV"])
                )

    st.info(
        "Sensitivity describes how many people with the target "
        "condition were flagged positive in the evaluated group. "
        "Lower sensitivity means a greater proportion of cases "
        "were missed at the evaluated threshold."
    )

    st.caption(
        "Sex, age and race/ethnicity are separate evaluations. "
        "These figures do not measure performance for their combined "
        "intersection. Subgroup PPV comes from the evaluation sample "
        "and differs from the geographically adjusted county PPV."
    )

    with st.expander("View all exported subgroup results"):
        st.dataframe(subgroups, hide_index=True)

# Assemble and display the three dashboard panels.
def main():
    st.title("The Backpack Doctor")
    st.caption(
        "FIELD-BASED DYSGLYCEMIA TRIAGE · "
        "NHANES 2021–2023 · CDC PLACES"
    )

    st.write(
        "Patient risk, regional reliability and "
        "performance across patient groups."
    )

    # Panel 1: load the fitted model and display patient prediction.
    try:
        model_version = file_version(MODEL_PATH)

        bundle = load_model_bundle(
            str(MODEL_PATH),
            model_version,
        )

    except Exception as error:
        st.error(
            f"The fitted model could not be loaded: {error}"
        )

    else:
        show_patient_prediction(bundle, model_version)

    # Panel 2: load the exported county reliability results.
    try:
        counties, excluded_count = load_county_results(
            str(COUNTY_PATH),
            file_version(COUNTY_PATH),
        )

    except Exception as error:
        st.error(
            f"County results could not be loaded: {error}"
        )

    else:
        show_county_reliability(counties, excluded_count)

    # Panel 3: display the matching subgroup evaluation results.
    show_subgroup_performance()

    with st.expander("Model and evaluation"):
        st.write(
            "The deployed model was refitted on the full analytic "
            "sample. Reported model performance comes from the "
            "evaluation in the modelling notebook, not from "
            "evaluating the refitted model on its training data."
        )

        st.write(
            "The dashboard loads the fitted model and exported "
            "county and subgroup results. It does not retrain "
            "the model or repeat the evaluation."
        )

    st.divider()
    st.caption(
        "HSLU · MSc Applied Data Science & AI · "
        "CIP02 · Research dashboard"
    )


# Start the dashboard.
if __name__ == "__main__":
    main()