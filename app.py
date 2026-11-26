import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ----------------------------------------------------------------- config
st.set_page_config(page_title="Used Car Appraiser", page_icon="🚗", layout="wide")

NEWEST_YEAR = 2020          # newest year in the training data (age = 2020 - year)
VAL_RMSE = 0.184            # blend validation RMSE on log-price -> about +-20%
OWNER_MAP = {
    "Test Drive Car": 0, "First Owner": 1, "Second Owner": 2,
    "Third Owner": 3, "Fourth & Above Owner": 4,
}

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Space Grotesk', sans-serif; color: #111; }
.stApp { background: #fafafa; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; color: #000; }
.mono { font-family: 'IBM Plex Mono', monospace; }

.hero { text-align: center; padding: 0.5rem 0 0.2rem 0; }
.hero h1 { font-size: 3rem; margin: 0; font-weight: 700; }
.hero p  { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
           letter-spacing: 0.18em; text-transform: uppercase; color: #555; margin-top: 0.2rem; }

.draw { fill: none; stroke: #000; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round;
        stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw 2.6s ease forwards; }
.thin { stroke-width: 1.2; animation-delay: 1.2s; }
@keyframes draw { to { stroke-dashoffset: 0; } }

div.stButton > button {
    background: #000; color: #fff; border: 2px solid #000; border-radius: 0;
    padding: 0.7rem 1.4rem; width: 100%;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.15em; text-transform: uppercase;
    transition: all 0.2s ease;
}
div.stButton > button:hover { background: #fff; color: #000; border-color: #000; }

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput input {
    border-radius: 0 !important; border-color: #000 !important; background: #fff !important;
}
label, .stSlider label { font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important; letter-spacing: 0.1em; text-transform: uppercase; }

.section-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.2em;
               text-transform: uppercase; border-bottom: 2px solid #000; padding-bottom: 0.3rem;
               margin-bottom: 1rem; display: inline-block; }

/* valuation ticket */
.ticket { background: #fff; border: 2px solid #000; padding: 1.6rem 1.8rem; position: relative;
          box-shadow: 8px 8px 0 #000; }
.ticket::before, .ticket::after { content: ''; position: absolute; top: 58%; width: 22px; height: 22px;
          background: #fafafa; border: 2px solid #000; border-radius: 50%; }
.ticket::before { left: -13px; } .ticket::after { right: -13px; }
.t-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.22em;
           text-transform: uppercase; color: #666; }
.t-price { font-size: 3.6rem; font-weight: 700; line-height: 1.05; letter-spacing: -0.03em; margin: 0.2rem 0; }
.t-full  { font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; color: #444; }
.t-divider { border: none; border-top: 2px dashed #000; margin: 1.3rem 0; }

.range-wrap { margin: 0.9rem 0 0.2rem 0; }
.range-track { position: relative; height: 14px; border: 2px solid #000; background:
    repeating-linear-gradient(135deg, #fff 0 5px, #e6e6e6 5px 6px); }
.range-fill { position: absolute; top: 0; bottom: 0; background: #000; }
.range-pin  { position: absolute; top: -9px; width: 4px; height: 30px; background: #000; }
.range-pin::after { content: '\u25BC'; position: absolute; top: -17px; left: -7px; font-size: 0.9rem; }
.range-labels { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace;
                font-size: 0.8rem; margin-top: 0.5rem; }
.mini { display: flex; gap: 1rem; margin-top: 0.4rem; }
.mini div { flex: 1; border: 1.5px solid #000; padding: 0.55rem 0.7rem; }
.mini b { display: block; font-size: 1.15rem; }

.placeholder { border: 2px dashed #000; padding: 2.5rem 1.5rem; text-align: center; background: #fff; }
.placeholder .mono { color: #555; font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; }
.footnote { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #666; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- helpers
@st.cache_resource
def loadArtifacts():
    xgbModel = XGBRegressor()
    xgbModel.load_model("xgbModel.json")
    rfModel = joblib.load("rfModel.pkl")
    prep = joblib.load("preprocess.pkl")
    defaults = joblib.load("defaults.pkl")
    return xgbModel, rfModel, prep, defaults


def inr(n):
    """Indian digit grouping: 1234567 -> ₹12,34,567"""
    s = str(int(round(n)))
    if len(s) > 3:
        last3, rest, parts = s[-3:], s[:-3], []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts) + "," + last3
    return "₹" + s


def short(n):
    if n >= 1e7:
        return f"₹{n / 1e7:.2f} Cr"
    if n >= 1e5:
        return f"₹{n / 1e5:.2f} Lakh"
    return inr(n)


def buildRow(cols, defaults, f):
    """Mirror the training pipeline: one row with exactly the saved feature columns."""
    row = pd.DataFrame([defaults])[cols].astype(float)
    # reset every one-hot group, then switch on the chosen value
    for c in cols:
        if c.startswith(("brand_", "fuel_", "seller_type_", "transmission_")):
            row.loc[0, c] = 0.0
    for key, prefix in [("brand", "brand_"), ("fuel", "fuel_"),
                        ("seller", "seller_type_"), ("trans", "transmission_")]:
        col = prefix + f[key]
        if col in cols:
            row.loc[0, col] = 1.0
    row.loc[0, "age"] = NEWEST_YEAR - f["year"]
    row.loc[0, "kmLog"] = np.log1p(f["km"])
    row.loc[0, "powerLog"] = np.log1p(f["power"])
    row.loc[0, "engine"] = f["engine"]
    row.loc[0, "torque"] = f["torque"]
    row.loc[0, "mileage"] = f["mileage"]
    row.loc[0, "seats"] = f["seats"]
    row.loc[0, "ownerEncoded"] = OWNER_MAP[f["owner"]]
    row.loc[0, "famEnc"] = f["famEnc"]
    return row


def predictPrice(rowOrRows, rfModel, xgbModel):
    p = 0.5 * rfModel.predict(rowOrRows) + 0.5 * xgbModel.predict(rowOrRows)
    return np.expm1(p)


CAR_SVG = """
<svg viewBox="0 0 800 270" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;">
  <line class="draw thin" pathLength="1" x1="20" y1="253" x2="780" y2="253"/>
  <path class="draw" pathLength="1" d="M70,205 L70,172 Q72,150 100,146 L205,128 Q245,80 305,72 L475,72 Q545,76 590,124 L690,138 Q735,146 740,176 L740,205 L660,205 A60,60 0 0 0 540,205 L270,205 A60,60 0 0 0 150,205 Z"/>
  <path class="draw" pathLength="1" d="M222,126 Q256,90 308,84 L385,84 L385,126 Z"/>
  <path class="draw" pathLength="1" d="M400,84 L470,84 Q520,88 562,125 L400,125 Z"/>
  <path class="draw thin" pathLength="1" d="M392,84 L392,200"/>
  <path class="draw thin" pathLength="1" d="M410,150 L445,150"/>
  <path class="draw thin" pathLength="1" d="M705,146 L735,152 L735,168 L700,162 Z"/>
  <path class="draw thin" pathLength="1" d="M72,160 L86,160 L86,182 L72,182"/>
  <path class="draw thin" pathLength="1" d="M300,175 L330,175 M300,182 L345,182 M300,189 L360,189"/>
  <path class="draw thin" pathLength="1" d="M600,170 Q650,168 700,178"/>
  <circle class="draw" pathLength="1" cx="210" cy="205" r="46"/>
  <circle class="draw thin" pathLength="1" cx="210" cy="205" r="24"/>
  <circle class="draw thin" pathLength="1" cx="210" cy="205" r="6"/>
  <circle class="draw" pathLength="1" cx="600" cy="205" r="46"/>
  <circle class="draw thin" pathLength="1" cx="600" cy="205" r="24"/>
  <circle class="draw thin" pathLength="1" cx="600" cy="205" r="6"/>
  <path class="draw thin" pathLength="1" d="M210,181 L210,229 M186,205 L234,205 M193,188 L227,222 M227,188 L193,222"/>
  <path class="draw thin" pathLength="1" d="M600,181 L600,229 M576,205 L624,205 M583,188 L617,222 M617,188 L583,222"/>
</svg>
"""

# ----------------------------------------------------------------- load
needed = ["xgbModel.json", "rfModel.pkl", "preprocess.pkl", "defaults.pkl"]
missing = [f for f in needed if not os.path.exists(f)]
if missing:
    st.error(f"Missing model files next to app.py: {', '.join(missing)}")
    st.stop()

xgbModel, rfModel, prep, defaults = loadArtifacts()
cols = prep["featureColumns"]
famMap = prep["famMap"]
keepFamilies = prep["keepFamilies"]
globalMean = prep["globalMean"]

brandOptions = sorted(c[len("brand_"):] for c in cols if c.startswith("brand_"))
fuelOptions = sorted(c[len("fuel_"):] for c in cols if c.startswith("fuel_"))
sellerOptions = sorted(c[len("seller_type_"):] for c in cols if c.startswith("seller_type_"))
transOptions = sorted(c[len("transmission_"):] for c in cols if c.startswith("transmission_"))

def pick(options, preferred):
    return options.index(preferred) if preferred in options else 0

# ----------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  {CAR_SVG}
  <h1>Used Car Appraiser</h1>
  <p>Random Forest &times; XGBoost &nbsp;|&nbsp; trained on 7,886 listings</p>
</div>
""", unsafe_allow_html=True)
st.write("")

left, right = st.columns([1, 1.15], gap="large")

# ----------------------------------------------------------------- form
with left:
    st.markdown('<span class="section-tag">01 / The car</span>', unsafe_allow_html=True)

    brand = st.selectbox("Brand", brandOptions, index=pick(brandOptions, "Maruti"))
    families = sorted(f for f in keepFamilies if f.startswith(brand))
    family = st.selectbox("Model family", families + ["Other / not listed"])

    c1, c2 = st.columns(2)
    year = c1.slider("Model year", 1995, NEWEST_YEAR, 2015)
    km = c2.number_input("Kilometres driven", min_value=1000, max_value=1_000_000,
                         value=60000, step=5000)

    c3, c4 = st.columns(2)
    fuel = c3.selectbox("Fuel", fuelOptions, index=pick(fuelOptions, "Petrol"))
    trans = c4.selectbox("Transmission", transOptions, index=pick(transOptions, "Manual"))

    c5, c6 = st.columns(2)
    owner = c5.selectbox("Owner", list(OWNER_MAP.keys()), index=1)
    seller = c6.selectbox("Seller type", sellerOptions, index=pick(sellerOptions, "Individual"))

    st.markdown('<span class="section-tag" style="margin-top:1rem">02 / Engine</span>',
                unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    engine = c7.number_input("Engine (cc)", 600, 4000, int(defaults["engine"]), step=50)
    power = c8.number_input("Max power (bhp)", 30.0, 600.0,
                            float(round(np.expm1(defaults["powerLog"]), 1)), step=5.0)

    with st.expander("Advanced specs (optional, defaults are typical values)"):
        a1, a2, a3 = st.columns(3)
        torque = a1.number_input("Torque (Nm)", 40.0, 800.0, float(round(defaults["torque"], 1)))
        mileage = a2.number_input("Mileage (kmpl)", 5.0, 45.0, float(round(defaults["mileage"], 1)))
        seats = a3.number_input("Seats", 2, 14, int(round(defaults["seats"])))

    go = st.button("Appraise this car")

# ----------------------------------------------------------------- result
with right:
    st.markdown('<span class="section-tag">03 / The valuation</span>', unsafe_allow_html=True)

    if not go:
        st.markdown("""
        <div class="placeholder">
          <div class="mono">Fill in the car on the left,<br>then press <b>Appraise</b></div>
        </div>""", unsafe_allow_html=True)
    else:
        famKey = family if family in keepFamilies else "Other"
        famEnc = famMap.get(famKey, globalMean)

        f = dict(brand=brand, fuel=fuel, trans=trans, seller=seller, owner=owner,
                 year=year, km=km, power=power, engine=engine, torque=torque,
                 mileage=mileage, seats=seats, famEnc=famEnc)
        row = buildRow(cols, defaults, f)

        pRf = float(np.expm1(rfModel.predict(row)[0]))
        pXgb = float(np.expm1(xgbModel.predict(row)[0]))
        price = float(predictPrice(row, rfModel, xgbModel)[0])

        low, high = price * np.exp(-VAL_RMSE), price * np.exp(VAL_RMSE)

        # position of the estimate inside the band (log scale -> always centred)
        pinPct = 50
        st.markdown(f"""
        <div class="ticket">
          <div class="t-label">Estimated resale value</div>
          <div class="t-price">{short(price)}</div>
          <div class="t-full">{inr(price)}</div>
          <hr class="t-divider">
          <div class="t-label">Likely range (&plusmn;1 validation error, about 20%)</div>
          <div class="range-wrap">
            <div class="range-track">
              <div class="range-fill" style="left:15%; right:15%;"></div>
              <div class="range-pin" style="left:calc({pinPct}% - 2px);"></div>
            </div>
            <div class="range-labels"><span>{short(low)}</span><span>{short(high)}</span></div>
          </div>
          <hr class="t-divider">
          <div class="t-label">Two models, two opinions</div>
          <div class="mini">
            <div><span class="t-label">Random Forest</span><b>{short(pRf)}</b></div>
            <div><span class="t-label">XGBoost</span><b>{short(pXgb)}</b></div>
            <div><span class="t-label">Gap</span><b>{abs(pRf - pXgb) / price * 100:.1f}%</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- depreciation curve for the same car across model years
        st.write("")
        st.markdown('<span class="section-tag">04 / Same car, other model years</span>',
                    unsafe_allow_html=True)
        years = list(range(2000, NEWEST_YEAR + 1))
        rows = pd.concat([buildRow(cols, defaults, {**f, "year": y}) for y in years],
                         ignore_index=True)
        curve = predictPrice(rows, rfModel, xgbModel)

        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        ax.fill_between(years, curve * np.exp(-VAL_RMSE), curve * np.exp(VAL_RMSE),
                        color="#000", alpha=0.08, linewidth=0)
        ax.plot(years, curve, color="#000", linewidth=2)
        ax.scatter([year], [price], s=90, color="#000", zorder=5)
        ax.annotate(" you", (year, price), fontsize=9, family="monospace", va="bottom")
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f"{v / 1e5:.0f}L" for v in ax.get_yticks()], family="monospace", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_family("monospace")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linestyle=":", color="#999", linewidth=0.7)
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#fafafa")
        st.pyplot(fig, use_container_width=True)

        st.markdown(
            '<div class="footnote">Curve holds kilometres and all other inputs fixed, so older years look '
            'cheaper partly because of age alone. Estimates exclude condition, accident history and '
            'location, which the dataset does not contain.</div>', unsafe_allow_html=True)