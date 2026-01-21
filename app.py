import streamlit as st
import joblib
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load('xgb_multi_label_loyalty_model.pkl')

model = load_model()

feature_names = [
    'CSOR_2', 'D_2_FAMILY', 'SAT_1', 'TRUST_4', 'SAT_3', 'PERF_3', 'D_3_EDUCATION',
    'CSOR_3', 'QUAL_5', 'ATTR_4', 'CSOR_5', 'CSOR_1', 'QUAL_2', 'PERF_4', 'TRUST_3',
    'SAT_2', 'A_1_AGE', 'QUA_7', 'LIKE_2', 'COMP_3', 'PERF_1', 'COMP_2', 'LIKE_1',
    'ATTR_2', 'ATTR_1', 'ATTR_3', 'D_4_EMPLOYMENT', 'QUAL_6', 'PERF_2', 'CSOR_4',
    'QUAL_1', 'COMP_1', 'D_1_GENDER', 'TRUST_2', 'TRUST_1', 'QUAL_4', 'PERF_5',
    'D_6_INCOME'
]

# Helper: Create Likert Radio Button
def likert_radio(question, key_prefix, default=4):
    return st.radio(
        question,
        options=["1", "2", "3", "4", "5", "6", "7"],
        index=default - 1,  # Convert 1-7 to 0-6 index
        horizontal=True,
        key=key_prefix
    )

st.title("🏦 CRM Intelligence: Customer Loyalty Segment Analysis")
st.write("""
This dashboard enables bank practitioners to analyze customer profiles and predict multi-dimensional loyalty segments. 
Input survey-based customer data below to generate a predictive risk score across cognitive, affective, and behavioral dimensions.
""")

st.header("Customer Profile & Demographics")
gender = st.radio("Customer Gender:", ["Female", "Male"], key="gender")
d1_gender = 0 if gender == "Female" else 1

age = st.select_slider("Customer Age Group:", options=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], key="age")
a1_age = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6}[age]

family_map = {
    "Preferred not to answer": 0,
    "Living alone": 1,
    "Living with a partner": 2,
    "Registered civil partnership": 3,
    "Married": 4,
    "Divorced": 5,
    "Widowed": 6
}
family_label = st.selectbox("Customer Marital Status:", list(family_map.keys()), key="family")
d2_family = family_map[family_label]

education_map = {
    "Preferred not to answer": 0,
    "No education": 1,
    "'Hauptschule'(completed 9ᵗʰ grade) ": 2,
    "'Mittlere Reife'(completed 10ᵗʰ grade)": 3,
    "'Fachhochschulreife'(completed 12ᵗʰ grade)": 4,
    "Abitur (High School Diploma) ": 5,
    "Vocational training": 6,
    "University degree": 7
}
edu_label = st.selectbox("Customer Highest Education Level:", list(education_map.keys()), key="education")
d3_education = education_map[edu_label]

employment_map = {
    "Preferred not to answer ": 0,
    "Unemployed": 1,
    "Retired": 2,
    "Houseman/housewife": 3,
    "In education": 4,
    "Studying at a university": 5,
    "Self-employed": 6,
    "Employed": 7
}
emp_label = st.selectbox("Customer Employment Status:", list(employment_map.keys()), key="employment")
d4_employment = employment_map[emp_label]

income_map = {
    "Preferred not to answer": 0,
    "Less than EUR 750": 1,
    "EUR 750–1250 ": 2,
    "EUR 1250–2000": 3,
    "EUR 2000–3500": 4,
    "EUR 3500–5000": 5,
    ">EUR 5000": 6
}
inc_label = st.selectbox("Monthly Household Income (after taxes):", list(income_map.keys()), key="income")
d6_income = income_map[inc_label]

st.header("Service Quality Indicators")
st.subheader("Customer Perceptions")
qual_1 = likert_radio("Bank attention to personal concerns.", "qual_1", 4)
qual_2 = likert_radio("Service range alignment with needs.", "qual_2", 4)
qual_4 = likert_radio("General trustworthiness of the institution.", "qual_4", 4)
qual_5 = likert_radio("Quality of products and services", "qual_5", 4)
qual_6 = likert_radio("Value-for-money of products/services.", "qual_6", 4)
qual_7 = likert_radio("Bank as a market pioneer vs. follower.", "qual_7", 4)

st.header("Operational Performance Assessments")
st.subheader("Assessments of Institutional Competence")
perf_1 = likert_radio("Economic stability of the institution.", "perf_1", 4)
perf_2 = likert_radio("Quality of company management.", "perf_2", 4)
perf_3 = likert_radio("Competitive economic performance level.", "perf_3", 4)
perf_4 = likert_radio("Clarity of corporate future vision.", "perf_4", 4)
perf_5 = likert_radio("Growth potential of the institution.", "perf_5", 4)

st.header("CSR & Ethical Reputation")
st.subheader("Impressions of Corporate Social Responsibility")
csor_1 = likert_radio("Interest in factors beyond profit maximization.", "csor_1", 4)
csor_2 = likert_radio("Commitment to environmental preservation.", "csor_2", 4)
csor_3 = likert_radio("Responsible behavior toward society.", "csor_3", 4)
csor_4 = likert_radio("Honesty in public information disclosure.", "csor_4", 4)
csor_5 = likert_radio("Fair treatment of market competitors.", "csor_5", 4)

st.header("Brand Attractiveness & Likeability")
st.subheader("Brand Perception Indicators")
attr_1 = likert_radio("General attractiveness of the company.", "attr_1", 4)
attr_2 = likert_radio("Visual appeal (branches, digital presence).", "attr_2", 4)
attr_3 = likert_radio("Qualification level of bank staff.", "attr_3", 4)
attr_4 = likert_radio("Attractiveness as a potential employer.", "attr_4", 4)

st.header("Emotional Affinity (Likeability)")
st.subheader("Emotional Connection with the Brand")
like_1 = likert_radio("Identification with bank vs. competitors.", "like_1", 4)
like_2 = likert_radio("Regret if the bank ceased to exist.", "like_2", 4)

st.header("Perceived Market Competence")
st.subheader("Market Positioning Factors")
comp_1 = likert_radio("Perception as a leading provider.", "comp_1", 4)
comp_2 = likert_radio("External reputation of the institution.", "comp_2", 4)
comp_3 = likert_radio("Adherence to highest service standards.", "comp_3", 4)

st.header("Customer Satisfaction")
st.subheader("Evaluative Judgments of Satisfaction")
sat_1 = likert_radio("Fulfillment of customer expectations.", "sat_1", 4)
sat_2 = likert_radio("General positive attitude toward bank.", "sat_2", 4)
sat_3 = likert_radio("Preference for bank over competitors.", "sat_3", 4)

st.header("Relational Trust Dimensions")
st.subheader("Trustworthiness Factors")
trust_1 = likert_radio("Listening to customer concerns/problems.", "trust_1", 4)
trust_2 = likert_radio("Provision of constructive solutions.", "trust_2", 4)
trust_3 = likert_radio("Alignment of values between bank and customer.", "trust_3", 4)
trust_4 = likert_radio("Acting in accordance with customer wishes.", "trust_4", 4)


# Predict Button
if st.button("Generate Loyalty Diagnostic"):
    # Assemble input row in correct feature order
    input_data = [
        int(csor_2), int(d2_family), int(sat_1), int(trust_4), int(sat_3), int(perf_3), int(d3_education),
        int(csor_3), int(qual_5), int(attr_4), float(csor_5), int(csor_1), int(qual_2), int(perf_4), int(trust_3),
        int(sat_2), int(a1_age), int(qual_7), int(like_2), int(comp_3), int(perf_1), int(comp_2), int(like_1),
        int(attr_2), int(attr_1), int(attr_3), int(d4_employment), int(qual_6), int(perf_2), int(csor_4),
        int(qual_1), int(comp_1), int(d1_gender), int(trust_2), int(trust_1), int(qual_4), int(perf_5),
        int(d6_income)
    ]

    input_df = pd.DataFrame([input_data], columns=feature_names)

    pred = model.predict(input_df)[0]

    class_map = {0: "Low", 1: "Neutral", 2: "High"}
    loy1_class = class_map[pred[0]]
    loy2_class = class_map[pred[1]]
    loy3_class = class_map[pred[2]]

    # Display Results
    st.success("✅ Diagnostic Analysis Complete!")
    st.subheader("Predicted Loyalty Segment Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Likelihood of customer to remain as customer of the bank**")
        st.metric("", loy1_class)
    with col2:
        st.markdown("**Likelihood of customer to purchase new banking products in future**")
        st.metric("", loy2_class)
    with col3:
        st.markdown("**Likelihood of customer to make use of other banking products or services offered by the bank**")
        st.metric("", loy3_class)

    st.info("""
        If results indicate 'Low' or 'Neutral', prioritize Trust and Satisfaction drivers.
    """)

st.markdown("---")
st.caption("Bank CRM Intelligence System • By: Ireti Samson Komolafe • © 2026")