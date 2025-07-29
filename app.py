import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# === Page Configuration ===
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load Pre-trained Model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Styling & Animations (title, image, footer) ===
st.markdown("""
<style>
  html, body, .stApp {
    background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
    color: #E0E0E0; font-family: 'Segoe UI', sans-serif;
  }
  .title {
    text-align: center; font-size: 2.5rem;
    animation: typing 2s steps(32, end) forwards;
    overflow: hidden; width: fit-content; margin: 0 auto; white-space: nowrap;
  }
  @keyframes typing { from { width: 0; } to { width: 100%; } }
  .subtitle {
    text-align: center; color: #81D4FA; margin-bottom: 1rem;
  }
  .fade-img {
    opacity: 0; transform: scale(0.95);
    animation: fadeIn 1.6s ease-in-out forwards;
    display: block; margin: auto;
  }
  @keyframes fadeIn { to { opacity: 1; transform: scale(1); } }
  .section-header {
    color: #B2EBF2; font-size: 1.4rem; margin-top: 2rem;
  }
  footer {
    text-align: left; margin-top: 3rem;
    color: #AAAAAA; border-top: 1px solid #444; padding: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# === Title & Subtitle Display ===
st.markdown('<div class="title">EV Adoption Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Forecast EV adoption in Washington State (3‑year outlook)</div>', unsafe_allow_html=True)

# === Image and Project Overview ===
col1, col2 = st.columns([1, 2])
with col1:
    st.image("ev-car-factory.png", width=360, caption=None)
    st.markdown('<div class="fade-img"></div>', unsafe_allow_html=True)
with col2:
    st.markdown("""
        Explore county-level electric vehicle adoption trends using machine learning forecasts.<br><br>
        • Uses recent EV registration data from Washington State<br>
        • Predicts cumulative EV counts over next 36 months<br>
        • Toggle between up to 3 counties for comparison<br><br>
        Designed for policymakers, planners, and sustainability researchers.
    """, unsafe_allow_html=True)

# === Load Data Function ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
county_list = sorted(df['County'].dropna().unique())

# === County Selection ===
st.markdown('<div class="section-header">Select a County</div>', unsafe_allow_html=True)
selected = st.selectbox("County:", [""] + county_list)
if selected == "":
    st.info("Please select a county to begin.")
    st.stop()

# === Forecast Function ===
def forecast_for_county(cty_df, code):
    hist = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
    cum = list(np.cumsum(hist))
    months = cty_df['months_since_start'].max()
    last_date = cty_df['Date'].max()
    future = []
    for i in range(1, 37):
        dt = last_date + pd.DateOffset(months=i)
        months += 1
        l1, l2, l3 = hist[-1], hist[-2], hist[-3]
        rm = np.mean([l1, l2, l3])
        pc1 = (l1 - l2)/l2 if l2 != 0 else 0
        pc3 = (l1 - l3)/l3 if l3 != 0 else 0
        slope = np.polyfit(range(len(cum[-6:])), cum[-6:], 1)[0] if len(cum)>=6 else 0
        new = {
            'months_since_start': months,
            'county_encoded': code,
            'ev_total_lag1': l1, 'ev_total_lag2': l2, 'ev_total_lag3': l3,
            'ev_total_roll_mean_3': rm,
            'ev_total_pct_change_1': pc1,
            'ev_total_pct_change_3': pc3,
            'ev_growth_slope': slope
        }
        pred = model.predict(pd.DataFrame([new]))[0]
        future.append({"Date": dt, "Predicted EV Total": round(pred)})
        hist.append(pred)
        cum.append(cum[-1] + pred)
        if len(hist) > 6:
            hist.pop(0); cum.pop(0)
    return future

# === Perform Forecast ===
cty_df = df[df['County'] == selected].sort_values('Date')
code = cty_df['county_encoded'].iloc[0]
future = forecast_for_county(cty_df, code)

hist = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
hist['Source'] = 'Historical'
hist['Cumulative EV'] = hist['Electric Vehicle (EV) Total'].cumsum()

fc_df = pd.DataFrame(future)
fc_df['Source'] = 'Forecast'
fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist['Cumulative EV'].iloc[-1]

combined = pd.concat([hist[['Date','Cumulative EV','Source']], fc_df[['Date','Cumulative EV','Source']]], ignore_index=True)

# === Plot Single-County Forecast ===
st.subheader(f"📊 Cumulative EV Forecast: {selected} County")
fig, ax = plt.subplots(figsize=(12,6))
for src, grp in combined.groupby('Source'):
    ax.plot(grp['Date'], grp['Cumulative EV'], marker='o', label=src)
ax.set_facecolor('#1E1E1E'); fig.patch.set_facecolor('#121212')
ax.set_xlabel("Date", color='white'); ax.set_ylabel("Cumulative EV", color='white')
ax.tick_params(colors='white'); ax.grid(True, alpha=0.3); ax.legend()
st.pyplot(fig)

# === Growth Summary ===
hist_total = hist['Cumulative EV'].iloc[-1]
fc_total = fc_df['Cumulative EV'].iloc[-1]
if hist_total > 0:
    percent = (fc_total - hist_total) / hist_total * 100
    st.success(f"Forecasted 3-Year Growth for **{selected}**: **{percent:.2f}%**")
else:
    st.warning("Insufficient historical EV data.")

# === Multi-County Comparison ===
st.markdown('<div class="section-header">Compare up to 3 Counties</div>', unsafe_allow_html=True)
others = st.multiselect("Select counties to compare", county_list, default=[])
if others:
    comp_dfs = []
    for cty in others:
        df_ct = df[df['County'] == cty].sort_values('Date')
        fc_ct = forecast_for_county(df_ct, df_ct['county_encoded'].iloc[0])
        hist_ct = df_ct[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_ct['Cumulative EV'] = hist_ct['Electric Vehicle (EV) Total'].cumsum()
        fc_temp = pd.DataFrame(fc_ct)
        fc_temp['Cumulative EV'] = fc_temp['Predicted EV Total'].cumsum() + hist_ct['Cumulative EV'].iloc[-1]
        comb_temp = pd.concat([hist_ct[['Date','Cumulative EV']], fc_temp[['Date','Cumulative EV']]], ignore_index=True)
        comb_temp['County'] = cty
        comp_dfs.append(comb_temp)
    comp_df = pd.concat(comp_dfs, ignore_index=True)

    st.subheader("📈 Multi-County Comparison")
    fig2, ax2 = plt.subplots(figsize=(14,7))
    for cty, grp in comp_df.groupby('County'):
        ax2.plot(grp['Date'], grp['Cumulative EV'], marker='o', label=cty)
    ax2.set_facecolor('#1E1E1E'); fig2.patch.set_facecolor('#121212')
    ax2.set_xlabel("Date", color='white'); ax2.set_ylabel("Cumulative EV", color='white')
    ax2.tick_params(colors='white'); ax2.grid(True, alpha=0.3); ax2.legend()
    st.pyplot(fig2)

    growth_summaries = []
    for cty in others:
        grp = comp_df[comp_df['County'] == cty]['Cumulative EV']
        if grp.iloc[0] > 0:
            growth_summaries.append(f"{cty}: {((grp.iloc[-1] - grp.iloc[0])/grp.iloc[0])*100:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A")
    st.success("Forecasted Growth — " + " | ".join(growth_summaries))

# === Footer ===
st.markdown('<footer>@C.SAI BALA KRISHNA</footer>', unsafe_allow_html=True)
