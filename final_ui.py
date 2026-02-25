import streamlit as st
import random
import pandas as pd
from pathlib import Path

# =========================================
# IMPORT YOUR EXISTING MODULES (unchanged)
# =========================================
try:
    from ai_engine import TotoBrain, PredictionStrategy
    HAS_AI = True
except:
    HAS_AI = False

try:
    from scraper import update_toto, get_all_results_from_csv
    HAS_SCRAPER = True
except:
    HAS_SCRAPER = False


# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="SG Toto AI",
    layout="wide"
)

st.title("🎯 SG Toto AI — Prediction Engine")


# =========================================
# LOAD BRAIN (cached)
# =========================================
@st.cache_resource
def load_brain():
    if HAS_AI:
        return TotoBrain()
    return None

brain = load_brain()

# Debugging lines — add them here 
st.write("Brain loaded:", brain is not None) 
if brain: 
    st.write("History length:", len(brain.history))

# =========================================
# MAIN PREDICT BUTTON
# =========================================
if st.button("Predict"):
    st.subheader("Prediction Results")

    if brain:
        try:
            result = brain.predict(PredictionStrategy.ENSEMBLE)
            st.success(f"Recommended: {result.numbers}")
            st.write(f"Confidence: {result.confidence*100:.1f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        nums = sorted(random.sample(range(1,50),6))
        st.warning(f"Fallback random: {nums}")

# =========================================
# SIDEBAR CONTROLS
# =========================================
st.sidebar.header("Controls")

run_btn = st.sidebar.button("Run AI")
update_btn = st.sidebar.button("Update Data")
retrain_btn = st.sidebar.button("Retrain AI")
ticket_qty = st.sidebar.number_input("Tickets", 1, 50, 5)
ticket_btn = st.sidebar.button("Generate Tickets")


# =========================================
# RUN AI
# =========================================
if run_btn:
    st.subheader("Prediction Results")

    if brain:
        result = brain.predict(PredictionStrategy.ENSEMBLE)
        st.success(f"Recommended: {result.numbers}")
        st.write(f"Confidence: {result.confidence*100:.1f}%")
    else:
        nums = sorted(random.sample(range(1,50),6))
        st.warning(f"Fallback random: {nums}")


# =========================================
# UPDATE DATA
# =========================================
if update_btn:
    if HAS_SCRAPER:
        ok, msg = update_toto()
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    else:
        st.error("Scraper not installed")


# =========================================
# TICKETS
# =========================================
if ticket_btn:
    st.subheader("Generated Tickets")

    tickets = []

    if brain:
        try:
            results = brain.predict_multiple(ticket_qty)
            tickets = [r.numbers for r in results]
        except:
            pass

    while len(tickets) < ticket_qty:
        tickets.append(sorted(random.sample(range(1,50),6)))

    for i,t in enumerate(tickets,1):
        st.write(f"#{i}: {t}")


# =========================================
# HISTORY TABLE
# =========================================
st.divider()
st.subheader("History")

if HAS_SCRAPER:
    results = get_all_results_from_csv(limit=300)

    rows = []
    for r in results:
        rows.append([r.draw_date] + r.winning_numbers + [r.additional_number])

    df = pd.DataFrame(
        rows,
        columns=["Date","N1","N2","N3","N4","N5","N6","Bonus"]
    )

    st.dataframe(df, use_container_width=True)


# =========================================
# HEATMAP
# =========================================
if HAS_SCRAPER:
    import numpy as np
    import matplotlib.pyplot as plt

    st.subheader("Frequency Heatmap")

    freq = np.zeros(49)

    for r in get_all_results_from_csv():
        for n in r.winning_numbers:
            freq[n-1]+=1

    fig, ax = plt.subplots()
    ax.bar(range(1,50), freq)
    st.pyplot(fig)