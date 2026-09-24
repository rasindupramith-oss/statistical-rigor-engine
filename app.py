import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from google import genai

st.set_page_config(
    page_title="Statistical Rigor Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Statistical Distribution Fitter & Tail Risk Engine")
st.markdown("Automated Goodness-of-Fit testing with statistical interpretation powered by Gemini.")

# --- SIDEBAR: Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google AI Studio API Key:", type="password")
    
    selected_model = st.selectbox(
        "Preferred Gemini Model:",
        ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"],
        index=0
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.caption("Supports numerical columns for parametric & heavy-tail distribution fitting.")

# --- CANDIDATE DISTRIBUTIONS ---
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
            # Fit distribution parameters (MLE)
            params = dist.fit(data)
            
            # Goodness-of-fit: Kolmogorov-Smirnov Test
            ks_stat, p_value = stats.kstest(data, dist.name, args=params)
            
            # Compute Log-Likelihood, AIC, and BIC
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
            
    # Rank by AIC (lower is better)
    df_results = pd.DataFrame(results).sort_values(by="AIC").reset_index(drop=True)
    return df_results

def call_gemini(api_key, prompt, preferred_model):
    """Clean and resilient function to query Gemini without syntax errors."""
    try:
        client = genai.Client(api_key=api_key.strip())
    except Exception as e:
        return None, None, f"Client Initialization Error: {e}"

    # Google's recommended models in priority order
    models_to_try = [
        preferred_model,
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-flash"
    ]
    
    # Remove duplicates while preserving order
    unique_models = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)

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

# --- MAIN APP LOGIC ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found in this file.")
    else:
        selected_col = st.selectbox("Select variable to model:", numeric_cols)
        raw_data = df[selected_col].dropna().values
        
        # Handle support domain: non-positive values
        if np.any(raw_data <= 0):
            st.warning("⚠️ Variable contains non-positive values (≤ 0). Fitting positive subset for distributions requiring strictly positive support.")
            fit_data = raw_data[raw_data > 0]
        else:
            fit_data = raw_data
            
        if len(fit_data) < 15:
            st.error("Sample size is too small for reliable distribution fitting (n < 15).")
        else:
            # Descriptive Statistics Metric Row
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Sample Size (n)", len(fit_data))
            col_m2.metric("Mean", f"{np.mean(fit_data):.2f}")
            col_m3.metric("Std Dev", f"{np.std(fit_data, ddof=1):.2f}")
            col_m4.metric("Skewness", f"{stats.skew(fit_data):.3f}")
            col_m5.metric("Excess Kurtosis", f"{stats.kurtosis(fit_data):.3f}")
            
            if st.button("🚀 Fit Distributions & Analyze", type="primary"):
                with st.spinner("Calculating Maximum Likelihood Estimates (MLE) & AIC/BIC..."):
                    results_df = fit_distributions(fit_data)
                    best_model = results_df.iloc[0]
                    
                st.success(f"Best Fitting Model: **{best_model['Distribution']}** (AIC: {best_model['AIC']:.2f}, KS p-value: {best_model['P_Value']:.4f})")
                
                # --- VISUALIZATION SECTION ---
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Fitted PDF vs Empirical Data")
                    fig = go.Figure()
                    
                    # Empirical density histogram
                    fig.add_trace(go.Histogram(
                        x=fit_data, 
                        histnorm='probability density',
                        name="Empirical Data",
                        opacity=0.6,
                        marker_color="#3b82f6"
                    ))
                    
                    # Compute x grid for smooth theoretical curves
                    x_vals = np.linspace(min(fit_data), max(fit_data), 500)
                    
                    # Plot top 2 distributions
                    palette = ["#ef4444", "#10b981"]
                    for idx, row in results_df.head(2).iterrows():
                        dist = row["Dist_Obj"]
                        params = row["Params"]
                        pdf_vals = dist.pdf(x_vals, *params)
                        fig.add_trace(go.Scatter(
                            x=x_vals, 
                            y=pdf_vals,
                            mode='lines',
                            name=f"{row['Distribution']} (AIC: {row['AIC']:.1f})",
                            line=dict(color=palette[idx % len(palette)], width=3)
                        ))
                        
                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_title=selected_col,
                        yaxis_title="Probability Density",
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.subheader("Model Comparison Ranking")
                    display_table = results_df[["Distribution", "AIC", "BIC", "KS_Stat", "P_Value"]].copy()
                    display_table["P_Value"] = display_table["P_Value"].map("{:.4f}".format)
                    display_table["AIC"] = display_table["AIC"].map("{:.2f}".format)
                    display_table["BIC"] = display_table["BIC"].map("{:.2f}".format)
                    display_table["KS_Stat"] = display_table["KS_Stat"].map("{:.4f}".format)
                    st.dataframe(display_table, use_container_width=True)
                    
                    csv_data = display_table.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results CSV", csv_data, "distribution_results.csv", "text/csv")
                    
                # --- GEMINI INTERPRETATION ---
                st.markdown("---")
                st.subheader("🧠 Gemini Statistical Verdict & Tail Risk Analysis")
                
                if not api_key:
                    st.info("💡 Paste your Google AI Studio API Key in the sidebar to generate the automated statistical report.")
                else:
                    with st.spinner("Consulting Gemini for statistical interpretation..."):
                        skewness = float(stats.skew(fit_data))
                        kurt = float(stats.kurtosis(fit_data))
                        
                        prompt = f"""
                        Act as a senior Applied Statistician.
                        Interpret the following empirical distribution fitting results for variable '{selected_col}':
                        
                        - Best Fitted Distribution: {best_model['Distribution']}
                        - AIC: {best_model['AIC']:.2f} (Lowest among all candidate distributions)
                        - Kolmogorov-Smirnov test p-value: {best_model['P_Value']:.4f}
                        - Empirical Skewness: {skewness:.3f}
                        - Empirical Excess Kurtosis: {kurt:.3f}
                        - Sample Size: {len(fit_data)}
                        
                        Provide a clean, structured statistical breakdown:
                        1. **Assessment of Fit**: Why did {best_model['Distribution']} fit best? What does the KS p-value tell us about whether the model adequately represents the observed data?
                        2. **Tail Behavior & Heavy-Tail Risk**: Interpret the skewness and excess kurtosis. Are extreme events (black swans, rare shocks) more likely than standard normal assumptions?
                        3. **Actionable Recommendations**: How should a practitioner use this fitted distribution for simulation (Monte Carlo), risk modeling, or forecasting? Mention any critical cautions.
                        """
                        
                        report_text, model_used, err_msg = call_gemini(api_key, prompt, selected_model)
                        
                        if report_text:
                            st.caption(f"Generated via **{model_used}**")
                            st.markdown(report_text)
                        else:
                            st.error(err_msg)
else:
    st.info("👋 Upload a CSV file in the sidebar to start distribution fitting.")