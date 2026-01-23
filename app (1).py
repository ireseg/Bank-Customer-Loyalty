import streamlit as st
import joblib
import numpy as np
import pandas as pd
import graphviz

# ----------------------------
# Load Model & Setup
# ----------------------------
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
        index=default - 1,
        horizontal=True,
        key=key_prefix
    )

# ----------------------------
# App Header & Inputs
# ----------------------------
st.title("🏦 CRM Intelligence: Customer Loyalty Segment Analysis")
st.write("""
This dashboard enables bank practitioners to analyze customer profiles and predict multi-dimensional loyalty segments. 
Input survey-based customer data below to generate a predictive risk score.
""")

st.header("1. Customer Profile")
gender = st.radio("Customer Gender:", ["Female", "Male"], key="gender")
d1_gender = 0 if gender == "Female" else 1

age = st.select_slider("Customer Age Group:", options=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], key="age")
a1_age = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6}[age]

family_map = {"Preferred not to answer": 0, "Living alone": 1, "Living with a partner": 2, "Registered civil partnership": 3, "Married": 4, "Divorced": 5, "Widowed": 6}
family_label = st.selectbox("Marital Status:", list(family_map.keys()), key="family")
d2_family = family_map[family_label]

education_map = {"Preferred not to answer": 0, "No education": 1, "'Hauptschule'": 2, "'Mittlere Reife'": 3, "'Fachhochschulreife'": 4, "Abitur": 5, "Vocational training": 6, "University degree": 7}
edu_label = st.selectbox("Education Level:", list(education_map.keys()), key="education")
d3_education = education_map[edu_label]

employment_map = {"Preferred not to answer ": 0, "Unemployed": 1, "Retired": 2, "Houseman/housewife": 3, "In education": 4, "Studying at a university": 5, "Self-employed": 6, "Employed": 7}
emp_label = st.selectbox("Employment Status:", list(employment_map.keys()), key="employment")
d4_employment = employment_map[emp_label]

income_map = {"Preferred not to answer": 0, "Less than EUR 750": 1, "EUR 750–1250 ": 2, "EUR 1250–2000": 3, "EUR 2000–3500": 4, "EUR 3500–5000": 5, ">EUR 5000": 6}
inc_label = st.selectbox("Monthly Household Income:", list(income_map.keys()), key="income")
d6_income = income_map[inc_label]

st.markdown("---")
st.header("2. Key Loyalty Drivers")

with st.expander("Service Quality & Performance (Click to Expand)", expanded=True):
    qual_1 = likert_radio("Bank attention to personal concerns.", "qual_1", 4)
    qual_2 = likert_radio("Service range alignment with needs.", "qual_2", 4)
    qual_4 = likert_radio("General trustworthiness.", "qual_4", 4)
    qual_5 = likert_radio("Quality of products/services.", "qual_5", 4)
    qual_6 = likert_radio("Value-for-money.", "qual_6", 4)
    qual_7 = likert_radio("Bank as market pioneer.", "qual_7", 4)
    perf_1 = likert_radio("Economic stability.", "perf_1", 4)
    perf_2 = likert_radio("Quality of management.", "perf_2", 4)
    perf_3 = likert_radio("Competitive performance.", "perf_3", 4)
    perf_4 = likert_radio("Corporate vision clarity.", "perf_4", 4)
    perf_5 = likert_radio("Growth potential.", "perf_5", 4)

with st.expander("Brand, CSR & Emotion (Click to Expand)"):
    csor_1 = likert_radio("Interest beyond profit.", "csor_1", 4)
    csor_2 = likert_radio("Environmental commitment.", "csor_2", 4)
    csor_3 = likert_radio("Social responsibility.", "csor_3", 4)
    csor_4 = likert_radio("Honesty in disclosure.", "csor_4", 4)
    csor_5 = likert_radio("Fairness to competitors.", "csor_5", 4)
    attr_1 = likert_radio("General attractiveness.", "attr_1", 4)
    attr_2 = likert_radio("Visual appeal.", "attr_2", 4)
    attr_3 = likert_radio("Staff qualifications.", "attr_3", 4)
    attr_4 = likert_radio("Employer attractiveness.", "attr_4", 4)
    like_1 = likert_radio("Identification with bank.", "like_1", 4)
    like_2 = likert_radio("Regret if bank closed.", "like_2", 4)

with st.expander("Trust, Competence & Satisfaction (Click to Expand)"):
    comp_1 = likert_radio("Leading provider status.", "comp_1", 4)
    comp_2 = likert_radio("External reputation.", "comp_2", 4)
    comp_3 = likert_radio("Service standards.", "comp_3", 4)
    sat_1 = likert_radio("Expectation fulfillment.", "sat_1", 4)
    sat_2 = likert_radio("Positive attitude.", "sat_2", 4)
    sat_3 = likert_radio("Preference over others.", "sat_3", 4)
    trust_1 = likert_radio("Listening to concerns.", "trust_1", 4)
    trust_2 = likert_radio("Constructive solutions.", "trust_2", 4)
    trust_3 = likert_radio("Value alignment.", "trust_3", 4)
    trust_4 = likert_radio("Acting on wishes.", "trust_4", 4)

# ----------------------------
# Logic & Display
# ----------------------------

if st.button("Generate Loyalty Diagnostic"):
    # 1. PREDICTION
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
    loy_results = [class_map[pred[0]], class_map[pred[1]], class_map[pred[2]]]
    
    # Display Prediction Results
    st.success("✅ Analysis Complete")
    st.subheader("Customer Risk Profile")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rational Loyalty (Cognitive)", loy_results[0])
    c2.metric("Emotional Loyalty (Affective)", loy_results[1])
    c3.metric("Action Loyalty (Behavioral)", loy_results[2])

    st.markdown("---")
    
    # 2. DYNAMIC STRATEGIC INTERVENTION BASED ON PREDICTION
    
    # Determine risk level
    low_count = loy_results.count("Low")
    neutral_count = loy_results.count("Neutral")
    high_count = loy_results.count("High")
    
    # CASE 1: CRITICAL RISK - Multiple Low dimensions
    if low_count >= 2:
        st.error("### 🚨 CRITICAL RISK: Multiple Loyalty Failures Detected")
        st.write(f"""
        This customer shows **{low_count} low loyalty dimension(s)**, indicating severe relationship breakdown. 
        Immediate intervention required using the Trust-Satisfaction-Loyalty chain.
        """)
        
        # Show the chain
        st.subheader("1. Understanding the Problem: The Loyalty Chain")
        st.info("""
        **How We Lost This Customer:** Loyalty follows a specific path:
        
        **Trust** (Foundation) → **Satisfaction** (Bridge) → **Loyalty** (Outcome)
        
        When multiple loyalty dimensions fail, it means the chain broke early - likely at the **Trust** stage.
        Our research shows Trust explains 70% of Satisfaction, which explains 67% of Loyalty.
        """)
        
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        dot.node('T', 'TRUST\n(Broken)', shape='box', style='filled', fillcolor='#ffcdd2')
        dot.node('S', 'SATISFACTION\n(Damaged)', shape='box', style='filled', fillcolor='#ffe082')
        dot.node('L', 'LOYALTY\n(Lost)', shape='box', style='filled', fillcolor='#ef9a9a')
        dot.edge('T', 'S', label='Impact: 84%')
        dot.edge('S', 'L', label='Impact: 68%')
        st.graphviz_chart(dot)

        # Critical intervention
        st.subheader("2. Emergency Action Plan: Rebuild Trust FIRST")
        st.write("Based on our structural equation model, here's what actually works to rebuild trust:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ PRIMARY INTERVENTIONS (Highest Impact)")
            st.success("""
            **1. Service Quality (Impact: 37%)**  
            *The strongest lever.* Focus on: faster complaint resolution, fewer errors, personalized service delivery.
            
            **2. Staff Competence (Impact: 25%)**  
            *Second priority.* Ensure expert staff handle this account. Competence rebuilds trust faster than friendliness.
            
            **3. Attractiveness (Impact: 22%)**  
            *Visual credibility matters.* Ensure professional branch appearance, modern digital interfaces, qualified staff visibility.
            """)
        
        with col2:
            st.markdown("#### ⚠️ SECONDARY INTERVENTIONS")
            st.info("""
            **4. Likeability (Impact: 16%)**  
            Emotional connection helps, but only after service quality is fixed.
            
            **5. CSR/Ethics (Impact: 13%)**  
            Corporate responsibility signals honesty, but won't overcome service failures.
            """)
        
        st.markdown("#### ❌ CRITICAL WARNING: Avoid This Mistake")
        st.warning("""
        **Performance/Future Promises (Impact: -10%)**  
        
        Counter-intuitively, emphasizing "growth potential" or "market leadership" **reduces trust** for at-risk customers.
        
        **Why?** It signals over-promising. Customers with broken trust need proof of current competence, not future vision.
        
        **Action:** Remove all marketing language. Focus only on demonstrated capability and immediate fixes.
        """)

    # CASE 2: MODERATE RISK - One Low or Multiple Neutral
    elif low_count == 1 or neutral_count >= 2:
        st.warning("### ⚠️ MODERATE RISK: Loyalty Instability Detected")
        
        # Identify which dimension(s) are problematic
        problem_dims = []
        if loy_results[0] in ["Low", "Neutral"]:
            problem_dims.append("Rational Loyalty (thinking)")
        if loy_results[1] in ["Low", "Neutral"]:
            problem_dims.append("Emotional Loyalty (feeling)")
        if loy_results[2] in ["Low", "Neutral"]:
            problem_dims.append("Action Loyalty (behavior)")
        
        st.write(f"""
        **Problem Areas:** {", ".join(problem_dims)}
        
        This customer is wavering. The Trust-Satisfaction-Loyalty chain is intact but weakening. 
        Targeted intervention can prevent escalation.
        """)
        
        st.subheader("1. The Loyalty Chain Status")
        st.info("""
        **Current State:** The chain is functioning but under strain.
        
        Trust → Satisfaction → Loyalty (84% → 68% impact)
        
        Your goal: Strengthen the weak link before it breaks completely.
        """)
        
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        dot.node('T', 'TRUST\n(Weakening)', shape='box', style='filled', fillcolor='#fff9c4')
        dot.node('S', 'SATISFACTION\n(At Risk)', shape='box', style='filled', fillcolor='#ffe082')
        dot.node('L', 'LOYALTY\n(Unstable)', shape='box', style='filled', fillcolor='#ffcc80')
        dot.edge('T', 'S', label='Impact: 84%')
        dot.edge('S', 'L', label='Impact: 68%')
        st.graphviz_chart(dot)

        st.subheader("2. Preventive Action Plan")
        
        # Dimension-specific recommendations
        if loy_results[0] in ["Low", "Neutral"]:  # Rational
            st.markdown("#### 🎯 Target: Rational Loyalty (Customer Thinking)")
            st.success("""
            **Focus on Competence & Quality:**
            - Demonstrate expertise in financial advice
            - Highlight objective service quality metrics
            - Emphasize competitive pricing and product features
            - Provide data-driven performance reports
            """)
        
        if loy_results[1] in ["Low", "Neutral"]:  # Emotional
            st.markdown("#### 🎯 Target: Emotional Loyalty (Customer Feeling)")
            st.success("""
            **Focus on Likeability & Attractiveness:**
            - Strengthen personal relationships with account managers
            - Improve branch/digital interface aesthetics
            - Create emotional connection through personalized communication
            - Build brand affinity through values alignment
            """)
        
        if loy_results[2] in ["Low", "Neutral"]:  # Behavioral
            st.markdown("#### 🎯 Target: Action Loyalty (Customer Behavior)")
            st.success("""
            **Focus on Satisfaction Drivers:**
            - Ensure consistent positive experiences
            - Reduce friction in all transactions
            - Proactively meet expectations before issues arise
            - Create reasons to increase engagement (new products, features)
            """)
        
        st.markdown("---")
        st.markdown("#### 📊 Priority Rankings (Based on SEM Model)")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.info("""
            **Top 3 Trust Drivers:**
            1. Service Quality (37%)
            2. Competence (25%)
            3. Attractiveness (22%)
            """)
        
        with col_b:
            st.warning("""
            **What NOT to Do:**
            - Don't over-promise on performance (-10% impact)
            - Don't ignore current service issues
            - Don't rely solely on marketing
            """)

    # CASE 3: STABLE - All High
    else:
        st.success("### 💎 SECURE LOYALTY: Premium Customer Segment")
        st.write("""
        **Status:** All loyalty dimensions are high. The Trust-Satisfaction-Loyalty chain is fully intact and strong.
        
        This customer is in your top tier. Our model shows they have:
        - Strong trust foundation
        - High satisfaction (84% driven by trust)
        - Robust loyalty (68% driven by satisfaction)
        """)
        st.balloons()
        
        st.subheader("Strategic Recommendations")
        
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        dot.node('T', 'TRUST\n(Strong)', shape='box', style='filled', fillcolor='#c8e6c9')
        dot.node('S', 'SATISFACTION\n(High)', shape='box', style='filled', fillcolor='#a5d6a7')
        dot.node('L', 'LOYALTY\n(Secure)', shape='box', style='filled', fillcolor='#81c784')
        dot.edge('T', 'S', label='Impact: 84%')
        dot.edge('S', 'L', label='Impact: 68%')
        st.graphviz_chart(dot)
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("#### ✅ DO THIS")
            st.success("""
            **1. Cross-Selling (Primary Strategy)**  
            High loyalty = high receptiveness to new products. Focus on revenue growth, not retention.
            
            **2. Maintain Service Quality**  
            Continue the quality that built trust (37% impact factor). Don't become complacent.
            
            **3. Leverage as Advocate**  
            Use for referrals, testimonials, case studies. Their loyalty is your competitive advantage.
            """)
        
        with col_y:
            st.markdown("#### ❌ AVOID THIS")
            st.info("""
            **1. Over-Retention Spending**  
            Don't waste retention budget here. They're already loyal.
            
            **2. Aggressive Upselling**  
            You can promote, but maintain the quality relationship. Trust is the foundation.
            
            **3. Neglecting Communication**  
            High loyalty doesn't mean low maintenance. Keep engagement consistent.
            """)

    # Universal footer with model insights
    st.markdown("---")
    st.subheader("📚 About This Analysis")
    with st.expander("Model Methodology & Research Basis"):
        st.write("""
        **Statistical Foundation:** This tool uses Partial Least Squares Structural Equation Modeling (PLS-SEM) 
        with 675 banking customers to identify causal relationships.
        
        **Key Findings:**
        - Trust explains 65% of customer variance
        - Satisfaction explains 61% of variance (70% driven by Trust)
        - Loyalty explains 67% of variance (67% driven by Satisfaction)
        
        **Path Coefficients (Strength of Relationships):**
        - Trust → Satisfaction: 0.84 (very strong)
        - Satisfaction → Loyalty: 0.68 (strong)
        - Indirect Trust → Loyalty: 0.57 (mediated through Satisfaction)
        
        **Service Quality Impact Breakdown:**
        The 37% impact of Service Quality on Trust comes from factors like complaint resolution speed, 
        error rates, personalization, and service consistency.
        
        **The Performance Paradox:**
        Marketing emphasis on "future growth" or "market leadership" has a negative -10% impact on trust 
        for customers with existing concerns. This suggests over-promising damages credibility when 
        current service quality is questioned.
        """)

st.markdown("---")
st.caption("Bank CRM Intelligence System • By: Ireti Samson Komolafe • © 2026")