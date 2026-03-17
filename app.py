import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HR Retention Portal",
    layout="wide",
    page_icon="👥",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL STYLE
# ============================================================
st.markdown("""
<style>
    .risk-high   { background:#fdecea; border-left:4px solid #e74c3c; padding:12px; border-radius:6px; }
    .risk-medium { background:#fef9e7; border-left:4px solid #f39c12; padding:12px; border-radius:6px; }
    .risk-low    { background:#eafaf1; border-left:4px solid #2ecc71; padding:12px; border-radius:6px; }
    .metric-box  { background:#f8f9fa; border-radius:8px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 0. HELPERS
# ============================================================
FEATURE_LABELS = {
    'Time_of_Work_Days':        'Tenure (days)',
    'Salary':                   'Salary ($)',
    'Absences':                 'Absences',
    'EmpSatisfaction':          'Job Satisfaction',
    'EngagementSurvey':         'Engagement Score',
    'SpecialProjectsCount':     'Special Projects',
    'Age_at_Reference':         'Age',
    'DaysLateLast30':           'Days Late (last 30)',
    'Time_after_Review_Days':   'Days Since Last Review',
    'PerfScoreID':              'Performance Score',
    'ManagerID':                'Manager',
    'PositionID':               'Position',
    'DeptID':                   'Department',
    'FromDiversityJobFairID':   'Diversity Job Fair Hire',
}

def label(col):
    for key, val in FEATURE_LABELS.items():
        if key in col:
            return val
    return col.replace('_', ' ').replace('RecruitmentSource ', 'Source: ')

def risk_badge(prob):
    if prob >= 0.6:
        return f'<div class="risk-high">🔴 <b>High Risk</b> — {prob:.1%} probability of leaving</div>'
    elif prob >= 0.3:
        return f'<div class="risk-medium">🟡 <b>Medium Risk</b> — {prob:.1%} probability of leaving</div>'
    else:
        return f'<div class="risk-low">🟢 <b>Low Risk</b> — {prob:.1%} probability of leaving</div>'

HR_ACTIONS = {
    'Salary':               '💰 Consider a compensation review — salary is below peer benchmark.',
    'EmpSatisfaction':      '💬 Schedule a 1:1 — satisfaction score is a key early warning signal.',
    'EngagementSurvey':     '📋 Run a pulse survey — engagement drop often precedes resignation.',
    'SpecialProjectsCount': '🚀 Offer new responsibilities — employee may feel under-stimulated.',
    'Absences':             '🏥 Wellbeing check-in — elevated absence pattern detected.',
    'Time_of_Work_Days':    '🏆 Tenure milestone — consider recognition or promotion path.',
    'Time_after_Review':    '📅 Performance review is overdue — schedule an evaluation.',
    'PerfScoreID':          '📈 Address performance gap — may need coaching or support.',
    'DaysLateLast30':       '⏰ Punctuality concern — investigate root cause.',
}

def get_action(feature_name):
    for key, action in HR_ACTIONS.items():
        if key in feature_name:
            return action
    return f'🔍 Investigate **{label(feature_name)}** — significant attrition driver.'

# ============================================================
# 1. DATA & MODEL
# ============================================================
ETHICS_COLS = ['Employee_Name', 'EmpID', 'RaceDesc', 'Sex', 'HispanicLatino',
               'CitizenDesc', 'GenderID', 'MaritalDesc', 'MaritalStatusID', 'MarriedID']

@st.cache_data
def load_and_split():
    df_processed = pd.read_csv('processed_hr_data.csv')
    df_raw       = pd.read_csv('HRDataset_v14.csv')

    y = df_processed['Termd']
    X = df_processed.drop(columns=[c for c in ['Termd'] + ETHICS_COLS if c in df_processed.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Keep raw sensitive columns aligned with test set for ethics audit
    df_ethics = df_raw[['Sex', 'RaceDesc', 'MaritalDesc']].loc[X_test.index]

    # Keep employee names aligned with test set for deep-dive
    df_names = df_raw[['Employee_Name']].loc[X_test.index].reset_index(drop=False)
    # rename original index so we can look it up
    df_names.columns = ['original_index', 'Employee_Name']

    return X_train, X_test, y_train, y_test, df_ethics, df_names

X_train, X_test, y_train, y_test, df_ethics, df_names = load_and_split()

@st.cache_resource
def train_model():
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_split=12,
        bootstrap=False, class_weight='balanced', random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

model = train_model()

@st.cache_resource
def compute_shap():
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(X_test)
    # Handle both old API (list) and new API (3D array)
    if isinstance(sv, list):
        return exp, sv[1], exp.expected_value[1]
    else:
        return exp, sv[:, :, 1], float(exp.expected_value[1])

explainer, sv_leave, base_value = compute_shap()

# Pre-compute probabilities once (FIX: was undefined before)
proba = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

# ============================================================
# 2. HEADER
# ============================================================
st.title("👥 HR Strategic Retention Portal")
st.caption("Powered by Random Forest + SHAP Explainability — Responsible AI principles applied")
st.markdown("---")

# KPI strip
total     = len(X_test)
high_risk = (proba >= 0.6).sum()
med_risk  = ((proba >= 0.3) & (proba < 0.6)).sum()
low_risk  = (proba < 0.3).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Employees Analysed",  total)
k2.metric("🔴 High Risk",         high_risk,  delta=f"{high_risk/total:.0%} of workforce",  delta_color="inverse")
k3.metric("🟡 Medium Risk",       med_risk,   delta=f"{med_risk/total:.0%} of workforce",   delta_color="off")
k4.metric("🟢 Low Risk",          low_risk,   delta=f"{low_risk/total:.0%} of workforce",   delta_color="normal")

st.markdown("---")

# ============================================================
# 3. TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Company Overview",
    "🔍 Employee Deep-Dive",
    "✨ Retention Simulator",
    "⚖️ Fairness Audit",
    "🆕 New Employee Prediction"
])

# ============================================================
# TAB 1 — GLOBAL OVERVIEW
# ============================================================
with tab1:
    st.header("What Drives Turnover Company-Wide?")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Global feature importance from SHAP
        mean_shap = pd.DataFrame({
            'Feature':    [label(c) for c in X_test.columns],
            'Importance': np.abs(sv_leave).mean(axis=0)
        }).sort_values('Importance', ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(mean_shap['Feature'], mean_shap['Importance'],
                       color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(mean_shap))))
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 10 Attrition Drivers (SHAP)", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.subheader("How to read this chart")
        st.info("""
        **Mean |SHAP value|** measures how much each feature 
        shifts the model's prediction on average across all employees.
        
        A longer bar = stronger influence on attrition risk, 
        regardless of direction (push toward leaving or staying).
        """)

        st.subheader("Risk Distribution")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        risk_counts = pd.Series({'🔴 High': high_risk, '🟡 Medium': med_risk, '🟢 Low': low_risk})
        ax2.pie(risk_counts, labels=risk_counts.index,
                colors=['#e74c3c', '#f39c12', '#2ecc71'],
                autopct='%1.0f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax2.set_title("Workforce Risk Breakdown", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # At-risk employees table
    st.subheader("🔴 Employees Requiring Immediate Attention")
    risk_df = df_names.copy()
    risk_df['Attrition Probability'] = proba
    risk_df['Risk Level'] = pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                                    labels=['🟢 Low', '🟡 Medium', '🔴 High'])
    risk_df['Actual Status'] = ['Left' if v == 1 else 'Active' for v in y_test.values]
    high_risk_df = risk_df[risk_df['Risk Level'] == '🔴 High'].sort_values(
        'Attrition Probability', ascending=False
    )[['Employee_Name', 'Attrition Probability', 'Risk Level', 'Actual Status']]

    st.dataframe(
        high_risk_df.style.format({'Attrition Probability': '{:.1%}'}),
        use_container_width=True
    )

# ============================================================
# TAB 2 — EMPLOYEE DEEP-DIVE
# ============================================================
with tab2:
    st.header("Individual Risk Analysis")

    selected_name = st.selectbox(
        "Select an employee:",
        options=df_names['Employee_Name'].tolist()
    )

    # Get position in X_test (integer index)
    row_info  = df_names[df_names['Employee_Name'] == selected_name].iloc[0]
    test_pos  = X_test.index.get_loc(row_info['original_index'])

    emp_prob  = proba[test_pos]
    emp_pred  = preds[test_pos]
    emp_actual = y_test.iloc[test_pos]

    # Risk banner
    st.markdown(risk_badge(emp_prob), unsafe_allow_html=True)
    st.markdown("")

    m1, m2, m3 = st.columns(3)
    m1.metric("Attrition Probability", f"{emp_prob:.1%}")
    m2.metric("Model Prediction",      "Will Leave" if emp_pred == 1 else "Will Stay")
    m3.metric("Actual Outcome",        "Left" if emp_actual == 1 else "Active")

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("What drives this score?")

        # SHAP waterfall-style bar chart (FIX: use ax.barh with explicit color list)
        contrib = pd.DataFrame({
            'Feature': [label(c) for c in X_test.columns],
            'SHAP':    sv_leave[test_pos]
        }).sort_values('SHAP').tail(8)   # top 8 most impactful

        bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in contrib['SHAP']]

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.barh(contrib['Feature'], contrib['SHAP'], color=bar_colors)
        ax3.axvline(0, color='black', linewidth=0.8)
        ax3.set_xlabel("SHAP Value (impact on attrition probability)")
        ax3.set_title(f"Key Factors — {selected_name}", fontweight='bold')
        ax3.spines[['top', 'right']].set_visible(False)

        red_patch   = mpatches.Patch(color='#e74c3c', label='Pushes toward leaving')
        green_patch = mpatches.Patch(color='#2ecc71', label='Pushes toward staying')
        ax3.legend(handles=[red_patch, green_patch], loc='lower right', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_b:
        st.subheader("💡 Recommended HR Actions")

        contrib_full = pd.Series(sv_leave[test_pos], index=X_test.columns)
        top_risk_features = contrib_full.nlargest(3).index.tolist()

        for feat in top_risk_features:
            st.markdown(get_action(feat))
            st.markdown("")

        st.subheader("Employee Profile")
        emp_data = X_test.iloc[[test_pos]].T.reset_index()
        emp_data.columns = ['Feature', 'Value']
        emp_data['Feature'] = emp_data['Feature'].apply(label)
        st.dataframe(emp_data, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3 — RETENTION SIMULATOR
# ============================================================
with tab3:
    st.header("✨ Retention Strategy Simulator")
    st.write("Select an at-risk employee and simulate what happens to their risk if HR takes action.")

    sim_name = st.selectbox("Choose employee to simulate:", df_names['Employee_Name'].tolist(), key='sim')
    sim_info = df_names[df_names['Employee_Name'] == sim_name].iloc[0]
    sim_pos  = X_test.index.get_loc(sim_info['original_index'])

    # Use that employee's ACTUAL row as the simulation base (FIX: was using random train row)
    sim_base = X_test.iloc[[sim_pos]].copy()
    current_prob = proba[sim_pos]

    st.markdown(f"**Current risk:** {risk_badge(current_prob)}", unsafe_allow_html=True)
    st.markdown("---")

    with st.form("simulator"):
        st.subheader("Adjust the levers HR can control:")
        c1, c2, c3 = st.columns(3)

        new_salary = c1.number_input(
            "New Salary ($)",
            value=float(sim_base['Salary'].iloc[0]),
            step=1000.0
        )
        new_sat = c2.slider(
            "Satisfaction Score (1-5)",
            1, 5,
            int(sim_base['EmpSatisfaction'].iloc[0])
        )
        new_projects = c3.slider(
            "Special Projects Assigned",
            0, 10,
            int(sim_base['SpecialProjectsCount'].iloc[0])
        )

        submitted = st.form_submit_button("▶ Run Simulation", use_container_width=True)

    if submitted:
        sim_row = sim_base.copy()
        sim_row['Salary']               = new_salary
        sim_row['EmpSatisfaction']      = new_sat
        sim_row['SpecialProjectsCount'] = new_projects

        new_prob = model.predict_proba(sim_row)[0][1]
        delta    = new_prob - current_prob

        col_curr, col_arrow, col_new = st.columns([2, 1, 2])
        col_curr.metric("Before", f"{current_prob:.1%}")
        col_arrow.markdown("<h2 style='text-align:center;margin-top:20px'>→</h2>", unsafe_allow_html=True)
        col_new.metric("After", f"{new_prob:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")

        st.markdown(risk_badge(new_prob), unsafe_allow_html=True)

        if new_prob < 0.2:
            st.balloons()
            st.success("🎉 These actions would strongly reduce this employee's risk of leaving.")
        elif delta < -0.1:
            st.success(f"✅ Significant improvement — risk reduced by {abs(delta):.1%}.")
        elif delta < 0:
            st.info(f"ℹ️ Modest improvement — risk reduced by {abs(delta):.1%}. Consider bolder actions.")
        else:
            st.warning("⚠️ These changes don't reduce risk. Try adjusting other levers.")

# ============================================================
# TAB 4 — FAIRNESS AUDIT
# ============================================================
with tab4:
    st.header("⚖️ Fairness & Bias Audit")
    st.info("""
    **Responsible AI principle:** Even though sensitive attributes (sex, race, marital status) 
    were removed from training, proxy variables may still encode indirect bias. 
    This audit checks whether model predictions are equally fair across demographic groups.
    """)

    attr = st.selectbox(
        "Audit predictions by:",
        options=['Sex', 'RaceDesc', 'MaritalDesc'],
        format_func=lambda x: {'Sex': 'Gender', 'RaceDesc': 'Race', 'MaritalDesc': 'Marital Status'}[x]
    )

    # FIX: align by reset_index to avoid index mismatch
    audit_df = df_ethics[[attr]].reset_index(drop=True).copy()
    audit_df['Actual']    = y_test.values
    audit_df['Predicted'] = preds

    def group_metrics(g):
        if len(g) < 2 or g['Actual'].nunique() < 2:
            return pd.Series({'False Negative Rate': np.nan,
                              'False Positive Rate': np.nan,
                              'Predicted Attrition Rate': np.nan,
                              'Sample Size': len(g)})
        tn, fp, fn, tp = confusion_matrix(g['Actual'], g['Predicted'], labels=[0, 1]).ravel()
        fnr = fn / (fn + tp)  if (fn + tp) > 0 else 0   # missed departures
        fpr = fp / (fp + tn)  if (fp + tn) > 0 else 0   # false alarms
        par = g['Predicted'].mean()                       # disparate impact indicator
        return pd.Series({'False Negative Rate': fnr,
                          'False Positive Rate': fpr,
                          'Predicted Attrition Rate': par,
                          'Sample Size': len(g)})

    results = audit_df.groupby(attr).apply(group_metrics).round(3)

    col_tbl, col_chart = st.columns([2, 3])

    with col_tbl:
        st.subheader("Metrics per Group")
        st.dataframe(
            results.style
                .format({'False Negative Rate':      '{:.1%}',
                         'False Positive Rate':      '{:.1%}',
                         'Predicted Attrition Rate': '{:.1%}',
                         'Sample Size':              '{:.0f}'})
                .background_gradient(subset=['False Negative Rate'], cmap='Reds'),
            use_container_width=True
        )
        # Disparate impact ratio
        par_vals = results['Predicted Attrition Rate'].dropna()
        if len(par_vals) >= 2:
            di_ratio = par_vals.min() / par_vals.max()
            st.metric(
                "Disparate Impact Ratio",
                f"{di_ratio:.3f}",
                help="Legal threshold > 0.8. Below 0.8 suggests potential discrimination."
            )
            if di_ratio < 0.8:
                st.error("⚠️ Below legal threshold (0.8) — investigate potential bias.")
            else:
                st.success("✅ Within acceptable fairness range.")

    with col_chart:
        st.subheader("Predicted Attrition Rate by Group")
        plot_data = results[['Predicted Attrition Rate', 'Sample Size']].dropna().reset_index()

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        bar_cols = ['#e74c3c' if v > par_vals.mean() else '#2ecc71'
                    for v in plot_data['Predicted Attrition Rate']]
        ax4.barh(plot_data[attr], plot_data['Predicted Attrition Rate'], color=bar_cols)
        ax4.axvline(par_vals.mean(), color='grey', linestyle='--', linewidth=1, label='Average')
        ax4.set_xlabel("Predicted Attrition Rate")
        ax4.set_title(f"Predicted Attrition by {attr}", fontweight='bold')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax4.spines[['top', 'right']].set_visible(False)
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")
    st.subheader("📋 Audit Conclusion")
    st.markdown("""
    **What this audit does NOT prove:** That the model is fully fair. Removing sensitive 
    columns is necessary but not sufficient — proxy variables can still encode indirect bias.
    
    **What this audit DOES provide:** Transparency. Any HR decision based on this model 
    should be reviewed by a human before action is taken. The model is a decision-support 
    tool, not a decision-making tool.
    """)

# ============================================================
# TAB 5 — NEW EMPLOYEE PREDICTION
# Manually enter any employee profile and get a risk prediction
# ============================================================
with tab5:
    st.header("🆕 Predict Risk for a New Employee")
    st.write("Enter an employee's profile manually — useful for onboarding or pre-emptive HR screening.")

    st.markdown("---")

    # ── Section 1: Profile ──────────────────────────────────
    st.subheader("👤 Employee Profile")
    p1, p2, p3 = st.columns(3)

    age = p1.number_input(
        "Age", min_value=18, max_value=70, value=35
    )
    salary = p2.number_input(
        "Annual Salary ($)", min_value=30000, max_value=250000,
        value=60000, step=1000
    )
    tenure = p3.number_input(
        "Tenure (days in company)", min_value=0, max_value=10000, value=365
    )

    # ── Section 2: Engagement ───────────────────────────────
    st.subheader("📊 Engagement & Performance")
    e1, e2, e3, e4 = st.columns(4)

    satisfaction = e1.slider("Job Satisfaction (1–5)",     1, 5, 3)
    engagement   = e2.slider("Engagement Score (1–5)",     1, 5, 3)
    perf_score   = e3.slider("Performance Score (1–4)",    1, 4, 3)
    projects     = e4.slider("Special Projects Count",     0, 10, 1)

    # ── Section 3: Behaviour ────────────────────────────────
    st.subheader("📅 Attendance & Review")
    b1, b2, b3 = st.columns(3)

    absences           = b1.number_input("Absences (last year)",       0, 30, 3)
    days_late          = b2.number_input("Days Late (last 30 days)",   0, 30, 0)
    days_since_review  = b3.number_input("Days Since Last Review",     0, 730, 90)

    # ── Section 4: Organisational ───────────────────────────
    st.subheader("🏢 Organisational Context")
    o1, o2, o3, o4 = st.columns(4)

    dept_id     = o1.selectbox("Department",          options=sorted(X_train['DeptID'].unique()))
    position_id = o2.selectbox("Position ID",         options=sorted(X_train['PositionID'].unique()))
    manager_id  = o3.selectbox("Manager ID",          options=sorted(X_train['ManagerID'].dropna().unique()))
    diversity   = o4.selectbox("Diversity Job Fair?", options=[0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")

    # ── Recruitment source dummies ───────────────────────────
    st.subheader("📣 Recruitment Source")
    rec_cols    = [c for c in X_train.columns if c.startswith('RecruitmentSource_')]
    rec_options = [c.replace('RecruitmentSource_', '').replace('_', ' ') for c in rec_cols]

    # Also pull the dropped-first category from the raw data so it appears as an option
    @st.cache_data
    def get_all_sources():
        df_raw = pd.read_csv('HRDataset_v14.csv')
        return sorted(df_raw['RecruitmentSource'].dropna().unique().tolist())

    all_sources    = get_all_sources()
    missing_source = [s for s in all_sources if s not in rec_options]  # the drop_first category
    full_options   = missing_source + rec_options  # dropped-first goes first as default

    selected_source = st.selectbox(
        "How was this employee recruited?",
        options=full_options
    )

    st.markdown("---")

    # ── Run prediction ───────────────────────────────────────
    if st.button("🔮 Predict Attrition Risk", use_container_width=True):

        # Build a row that matches X_train column structure exactly
        new_row = pd.DataFrame(columns=X_train.columns, data=np.zeros((1, len(X_train.columns))))

        # Fill known fields
        field_map = {
            'Age_at_Reference':         age,
            'Salary':                   salary,
            'Time_of_Work_Days':        tenure,
            'EmpSatisfaction':          satisfaction,
            'EngagementSurvey':         engagement,
            'PerfScoreID':              perf_score,
            'SpecialProjectsCount':     projects,
            'Absences':                 absences,
            'DaysLateLast30':           days_late,
            'Time_after_Review_Days':   days_since_review,
            'DeptID':                   dept_id,
            'PositionID':               position_id,
            'ManagerID':                manager_id,
            'FromDiversityJobFairID':   diversity,
        }
        for col, val in field_map.items():
            if col in new_row.columns:
                new_row[col] = val

        # One-hot encode recruitment source
        rec_col = f'RecruitmentSource_{selected_source.replace(" ", "_")}'
        if rec_col in new_row.columns:
            new_row[rec_col] = 1

        # Predict
        new_prob  = model.predict_proba(new_row)[0][1]
        new_pred  = model.predict(new_row)[0]

        # ── Results ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        st.markdown(risk_badge(new_prob), unsafe_allow_html=True)
        st.markdown("")

        r1, r2 = st.columns(2)
        r1.metric("Attrition Probability", f"{new_prob:.1%}")
        r2.metric("Model Decision",        "⚠️ At Risk" if new_pred == 1 else "✅ Likely to Stay")

        # ── SHAP explanation ────────────────────────────────
        st.subheader("🔍 Why? — Key Factors")

        new_sv = explainer.shap_values(new_row)
        if isinstance(new_sv, list):
            new_sv_leave = new_sv[1][0]
        else:
            new_sv_leave = new_sv[0, :, 1]

        contrib_new = pd.DataFrame({
            'Feature': [label(c) for c in X_train.columns],
            'SHAP':    new_sv_leave
        }).sort_values('SHAP')

        # Show top 5 risk + top 3 protective
        top_risk  = contrib_new.tail(5)
        top_prot  = contrib_new.head(3)
        plot_data = pd.concat([top_prot, top_risk])

        bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in plot_data['SHAP']]

        fig5, ax5 = plt.subplots(figsize=(9, 5))
        ax5.barh(plot_data['Feature'], plot_data['SHAP'], color=bar_colors)
        ax5.axvline(0, color='black', linewidth=0.8)
        ax5.set_xlabel("SHAP Value")
        ax5.set_title("Factors Driving This Prediction", fontweight='bold')
        ax5.spines[['top', 'right']].set_visible(False)

        red_patch   = mpatches.Patch(color='#e74c3c', label='Pushes toward leaving')
        green_patch = mpatches.Patch(color='#2ecc71', label='Pushes toward staying')
        ax5.legend(handles=[red_patch, green_patch], fontsize=9)

        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

        # ── HR Action Card ───────────────────────────────────
        if new_pred == 1 or new_prob >= 0.3:
            st.subheader("💡 Recommended HR Actions")

            top_risk_features = pd.Series(
                new_sv_leave, index=X_train.columns
            ).nlargest(3).index.tolist()

            for feat in top_risk_features:
                st.markdown(f"- {get_action(feat)}")

        st.markdown("---")
        st.caption(
            "⚠️ This prediction is a decision-support tool, not a final decision. "
            "All HR actions must be reviewed by a qualified human professional."
        )