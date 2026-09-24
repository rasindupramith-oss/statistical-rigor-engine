import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from google import genai

st.set_page_config(
    page_title="Statistical Rigor Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Statistical Rigor & Hypothesis Engine")
st.markdown("Automated Goodness-of-Fit and Hypothesis Testing powered by SciPy and Gemini.")

# --- SIDEBAR & SECRETS CONFIGURATION ---
secret_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuration")
    if secret_key:
        st.success("🔒 Cloud API Key Loaded")
        api_key = secret_key
    else:
        api_key = st.text_input("Enter Google AI Studio API Key:", type="password")
        
    selected_model = st.selectbox(
        "Preferred Gemini Model:",
        ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"],
        index=0
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.caption("Upload your data to run distribution fitting or hypothesis tests.")

# --- GEMINI CLIENT HELPER ---
def call_gemini(api_key, prompt, preferred_model):
    if not api_key:
        return None, None, "No API key found. Configure Secrets or enter a key."
    try:
        client = genai.Client(api_key=api_key.strip())
    except Exception as e:
        return None, None, f"Client Initialization Error: {e}"

    models_to_try = [preferred_model, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
    unique_models = list(dict.fromkeys([m for m in models_to_try if m]))

    last_error = None
    for model_name in unique_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text, model_name, None
        except Exception as e:
            last_error = e
            continue
    return None, None, f"Error communicating with Gemini: {last_error}"

# --- CANDIDATE DISTRIBUTIONS (MODULE 1) ---
DISTRIBUTIONS = {
    "Normal": stats.norm,
    "Log-Normal": stats.lognorm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull (Min)": stats.weibull_min
}

def fit_distributions(data):
    results = []
    n = len(data)
    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(data)
            ks_stat, p_value = stats.kstest(data, dist.name, args=params)
            log_lik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * log_lik
            bic = k * np.log(n) - 2 * log_lik
            results.append({
                "Distribution": name,
                "Params": params,
                "KS_Stat": ks_stat,
                "P_Value": p_value,
                "Log_Likelihood": log_lik,
                "AIC": aic,
                "BIC": bic,
                "Dist_Obj": dist
            })
        except Exception:
            continue
    return pd.DataFrame(results).sort_values(by="AIC").reset_index(drop=True)

# --- APP TABS ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head(4), use_container_width=True)

    tab1, tab2 = st.tabs(["📈 Module 1: Distribution Fitter", "🧪 Module 2: Smart Hypothesis Testing"])

    # ==========================================
    # TAB 1: DISTRIBUTION FITTER
    # ==========================================
    with tab1:
        st.subheader("Automated Continuous Distribution Fitting & Tail Risk")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found.")
        else:
            selected_col = st.selectbox("Select variable to model:", numeric_cols, key="dist_col")
            raw_data = df[selected_col].dropna().values
            
            if np.any(raw_data <= 0):
                st.warning("⚠️ Data contains non-positive values (≤ 0). Using positive subset for heavy-tail fits.")
                fit_data = raw_data[raw_data > 0]
            else:
                fit_data = raw_data
                
            if len(fit_data) < 15:
                st.error("Sample size is too small (n < 15).")
            else:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Sample Size (n)", len(fit_data))
                m2.metric("Mean", f"{np.mean(fit_data):.2f}")
                m3.metric("Std Dev", f"{np.std(fit_data, ddof=1):.2f}")
                m4.metric("Skewness", f"{stats.skew(fit_data):.3f}")
                m5.metric("Kurtosis", f"{stats.kurtosis(fit_data):.3f}")

                if st.button("🚀 Fit Distributions", type="primary", key="btn_dist"):
                    with st.spinner("Optimizing Maximum Likelihood Estimates..."):
                        results_df = fit_distributions(fit_data)
                        best_model = results_df.iloc[0]
                        
                    st.success(f"Best Fitting Model: **{best_model['Distribution']}** (AIC: {best_model['AIC']:.2f})")
                    
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=fit_data, histnorm='probability density', name="Empirical Data", opacity=0.6, marker_color="#3b82f6"))
                        x_vals = np.linspace(min(fit_data), max(fit_data), 500)
                        palette = ["#ef4444", "#10b981"]
                        for idx, row in results_df.head(2).iterrows():
                            pdf_vals = row["Dist_Obj"].pdf(x_vals, *row["Params"])
                            fig.add_trace(go.Scatter(x=x_vals, y=pdf_vals, mode='lines', name=f"{row['Distribution']} (AIC: {row['AIC']:.1f})", line=dict(color=palette[idx % 2], width=3)))
                        fig.update_layout(template="plotly_dark", xaxis_title=selected_col, yaxis_title="Density", margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with c2:
                        disp_table = results_df[["Distribution", "AIC", "BIC", "KS_Stat", "P_Value"]].copy()
                        disp_table["P_Value"] = disp_table["P_Value"].map("{:.4f}".format)
                        disp_table["AIC"] = disp_table["AIC"].map("{:.2f}".format)
                        disp_table["BIC"] = disp_table["BIC"].map("{:.2f}".format)
                        st.dataframe(disp_table, use_container_width=True)
                        
                    st.markdown("---")
                    st.subheader("🧠 Gemini Statistical Verdict & Tail Risk Analysis")
                    with st.spinner("Generating statistical interpretation..."):
                        p_dist = f"""
                        Act as a senior Applied Statistician.
                        Interpret empirical distribution results for variable '{selected_col}':
                        - Best Fit: {best_model['Distribution']} (AIC: {best_model['AIC']:.2f}, KS p-value: {best_model['P_Value']:.4f})
                        - Skewness: {stats.skew(fit_data):.3f}, Excess Kurtosis: {stats.kurtosis(fit_data):.3f}, n = {len(fit_data)}
                        1. Assessment of fit and KS test sensitivity.
                        2. Tail risk & extreme value implications.
                        3. Practical advice for simulation/modeling.
                        """
                        report, model_used, err = call_gemini(api_key, p_dist, selected_model)
                        if report:
                            st.caption(f"Generated via **{model_used}**")
                            st.markdown(report)
                        else:
                            st.error(err)

    # ==========================================
    # TAB 2: SMART HYPOTHESIS TESTING ADVISOR
    # ==========================================
    with tab2:
        st.subheader("🧪 Automated Hypothesis Testing & Assumption Verification")
        st.markdown("Select a continuous metric and a categorical grouping variable. The advisor checks assumptions and routes to the correct parametric or non-parametric test.")
        
        all_cols = df.columns.tolist()
        num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_candidates = df.select_dtypes(include=['object', 'category', 'int', 'bool']).columns.tolist()
        
        if len(num_candidates) < 1 or len(cat_candidates) < 1:
            st.error("Need at least one numerical column and one categorical column.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                target_var = st.selectbox("Continuous Dependent Variable (Target):", num_candidates, key="hypo_target")
            with col_b:
                group_var = st.selectbox("Categorical Independent Variable (Group):", cat_candidates, key="hypo_group")
                
            unique_groups = df[group_var].dropna().unique().tolist()
            
            if len(unique_groups) < 2:
                st.warning("Selected grouping variable must contain at least 2 distinct categories.")
            else:
                selected_groups = st.multiselect("Select groups to compare:", unique_groups, default=unique_groups[:2])
                
                if len(selected_groups) < 2:
                    st.info("Please select at least 2 groups to perform a comparison test.")
                else:
                    # Filter data
                    sub_df = df[df[group_var].isin(selected_groups)][[target_var, group_var]].dropna()
                    group_data = [sub_df[sub_df[group_var] == g][target_var].values for g in selected_groups]
                    
                    # Descriptive metrics per group
                    st.write("#### Group Summaries")
                    summary_list = []
                    for g in selected_groups:
                        vals = sub_df[sub_df[group_var] == g][target_var]
                        summary_list.append({
                            "Group": str(g),
                            "Count (n)": len(vals),
                            "Mean": f"{np.mean(vals):.2f}",
                            "Std Dev": f"{np.std(vals, ddof=1):.2f}",
                            "Median": f"{np.median(vals):.2f}",
                            "IQR": f"{stats.iqr(vals):.2f}"
                        })
                    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
                    
                    if st.button("🧪 Run Hypothesis Test & Diagnosis", type="primary", key="btn_hypo"):
                        st.write("---")
                        st.subheader("1. Assumption Diagnostics")
                        
                        # Check Normality per group (Shapiro-Wilk)
                        normality_passed = True
                        shapiro_results = []
                        for g in selected_groups:
                            vals = sub_df[sub_df[group_var] == g][target_var].values
                            if len(vals) >= 3:
                                s_stat, s_p = stats.shapiro(vals)
                                is_norm = s_p > 0.05
                                if not is_norm:
                                    normality_passed = False
                                shapiro_results.append(f"Group **'{g}'**: Shapiro-Wilk p = `{s_p:.4f}` ({'✅ Normal' if is_norm else '❌ Non-Normal'})")
                            else:
                                normality_passed = False
                                shapiro_results.append(f"Group **'{g}'**: n < 3 (Cannot reliably test normality)")
                                
                        for msg in shapiro_results:
                            st.markdown(f"- {msg}")
                            
                        # Check Equal Variance (Levene)
                        lev_stat, lev_p = stats.levene(*group_data)
                        equal_var = lev_p > 0.05
                        st.markdown(f"- **Homoscedasticity (Levene's Test)**: p = `{lev_p:.4f}` ({'✅ Equal Variances' if equal_var else '❌ Unequal Variances'})")
                        
                        # --- ROUTE TO STATISTICAL TEST ---
                        st.subheader("2. Test Selection & Results")
                        is_two_group = (len(selected_groups) == 2)
                        test_name = ""
                        stat_val = 0.0
                        pval = 0.0
                        effect_name = ""
                        effect_val = 0.0
                        
                        if is_two_group:
                            g1, g2 = group_data[0], group_data[1]
                            if normality_passed and equal_var:
                                test_name = "Student's Two-Sample t-test (Parametric)"
                                stat_val, pval = stats.ttest_ind(g1, g2, equal_var=True)
                                # Cohen's d
                                s_pooled = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1)) / (len(g1)+len(g2)-2))
                                effect_val = (np.mean(g1) - np.mean(g2)) / s_pooled if s_pooled > 0 else 0
                                effect_name = "Cohen's d"
                            elif normality_passed and not equal_var:
                                test_name = "Welch's t-test (Unequal Variance)"
                                stat_val, pval = stats.ttest_ind(g1, g2, equal_var=False)
                                s_pooled = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
                                effect_val = (np.mean(g1) - np.mean(g2)) / s_pooled if s_pooled > 0 else 0
                                effect_name = "Cohen's d"
                            else:
                                test_name = "Mann-Whitney U Test (Non-Parametric)"
                                stat_val, pval = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                                # Rank-biserial correlation
                                n1, n2 = len(g1), len(g2)
                                effect_val = 1 - (2 * stat_val) / (n1 * n2)
                                effect_name = "Rank-Biserial r"
                        else:
                            if normality_passed and equal_var:
                                test_name = "One-way ANOVA (Parametric)"
                                stat_val, pval = stats.f_oneway(*group_data)
                                # Eta squared
                                all_vals = np.concatenate(group_data)
                                grand_mean = np.mean(all_vals)
                                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
                                ss_total = sum((x - grand_mean)**2 for x in all_vals)
                                effect_val = ss_between / ss_total if ss_total > 0 else 0
                                effect_name = "Eta-Squared (η²)"
                            else:
                                test_name = "Kruskal-Wallis H Test (Non-Parametric)"
                                stat_val, pval = stats.kruskal(*group_data)
                                n_total = len(np.concatenate(group_data))
                                effect_val = (stat_val - len(selected_groups) + 1) / (n_total - len(selected_groups))
                                effect_name = "Epsilon-Squared (ε²)"

                        # Result Banner
                        sig_text = "Statistically Significant Difference (p < 0.05)" if pval < 0.05 else "No Statistically Significant Difference (p ≥ 0.05)"
                        st.info(f"**Applied Test:** {test_name}\n\n**Verdict:** {sig_text}")

                        r1, r2, r3 = st.columns(3)
                        r1.metric("Test Statistic", f"{stat_val:.3f}")
                        r2.metric("p-value", f"{pval:.4e}" if pval < 0.001 else f"{pval:.4f}")
                        r3.metric(f"Effect Size ({effect_name})", f"{effect_val:.3f}")

                        # Visualization: Box / Violin
                        st.subheader("3. Group Distribution Plot")
                        fig_box = px.box(
                            sub_df, 
                            x=group_var, 
                            y=target_var, 
                            color=group_var,
                            points="all",
                            template="plotly_dark",
                            title=f"{target_var} by {group_var}"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)

                        # Gemini Interpretation
                        st.markdown("---")
                        st.subheader("🧠 Gemini Statistical Verdict & Research Writeup")
                        with st.spinner("Synthesizing APA-format hypothesis report..."):
                            p_hypo = f"""
                            Act as a senior Applied Statistician.
                            Interpret this hypothesis test:
                            - Dependent Variable: {target_var}
                            - Grouping Variable: {group_var}
                            - Compared Groups: {selected_groups}
                            - Normality Check Passed?: {normality_passed}
                            - Equal Variance Passed?: {equal_var}
                            - Chosen Test: {test_name}
                            - Test Statistic: {stat_val:.3f}
                            - p-value: {pval:.4e}
                            - Effect Size ({effect_name}): {effect_val:.3f}
                            
                            Please write:
                            1. **Selection Rationale**: Why was this specific test mathematically required over the alternative?
                            2. **Substantive Conclusion**: In practical terms, does the group make a real-world difference?
                            3. **Formal APA-Style Sentence**: Provide a formal APA 7th edition statistical reporting paragraph that can be pasted directly into a thesis or research paper.
                            """
                            report_h, model_h, err_h = call_gemini(api_key, p_hypo, selected_model)
                            if report_h:
                                st.caption(f"Generated via **{model_h}**")
                                st.markdown(report_h)
                            else:
                                st.error(err_h)
else:
    st.info("👋 Upload a CSV file in the sidebar to begin analysis.")
