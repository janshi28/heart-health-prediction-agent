import streamlit as st
import numpy as np
import math

# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Health Check",
    page_icon="❤️",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f0eb; }
    .stApp { background-color: #f5f0eb; }
    h1 { color: #d94f3d !important; font-size: 2.2rem !important; }
    h2 { color: #1a3a5c !important; }
    h3 { color: #0d7377 !important; }
    .risk-box {
        padding: 20px; border-radius: 16px;
        text-align: center; margin: 10px 0;
    }
    .high-risk { background-color: #fde8e6; border: 2px solid #d94f3d; }
    .mid-risk  { background-color: #fef3e2; border: 2px solid #c47c1a; }
    .low-risk  { background-color: #e8f5ee; border: 2px solid #2e8b57; }
    .section-card {
        background: white; padding: 20px;
        border-radius: 16px; margin-bottom: 16px;
        border: 1.5px solid #e2d9cf;
    }
    .meal-card {
        background: #f0fafa; padding: 12px 16px;
        border-radius: 12px; margin-bottom: 8px;
        border-left: 4px solid #0d7377;
    }
    .avoid-chip {
        display: inline-block; background: #fde8e6;
        color: #d94f3d; border-radius: 8px;
        padding: 3px 10px; margin: 3px;
        font-weight: bold; font-size: 0.85rem;
    }
    .good-chip {
        display: inline-block; background: #e8f5ee;
        color: #2e8b57; border-radius: 8px;
        padding: 3px 10px; margin: 3px;
        font-weight: bold; font-size: 0.85rem;
    }
    div[data-testid="stCheckbox"] label {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    .stProgress > div > div { background-color: #d94f3d !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── FEATURE WEIGHTS (from trained Random Forest on 70k records) ──────
WEIGHTS = {
    'Age': 0.1426, 'Cold_Sweats_Nausea': 0.1164, 'Fatigue': 0.1147,
    'Dizziness': 0.0956, 'Shortness_of_Breath': 0.0946,
    'Pain_Arms_Jaw_Back': 0.0941, 'Swelling': 0.0936,
    'Chest_Pain': 0.0900, 'Palpitations': 0.0723,
    'Sedentary_Lifestyle': 0.0121, 'Family_History': 0.0117,
    'Chronic_Stress': 0.0113, 'Obesity': 0.0108,
    'Smoking': 0.0107, 'High_BP': 0.0105,
    'High_Cholesterol': 0.0104, 'Diabetes': 0.0086
}

SYMPTOM_KEYS = ['Chest_Pain','Shortness_of_Breath','Fatigue','Palpitations',
                'Dizziness','Swelling','Pain_Arms_Jaw_Back','Cold_Sweats_Nausea']
RISK_KEYS    = ['High_BP','High_Cholesterol','Diabetes','Smoking','Obesity',
                'Sedentary_Lifestyle','Family_History','Chronic_Stress']

# ── SCORE FUNCTION ───────────────────────────────────────────────────
def calc_score(age, features):
    s = WEIGHTS['Age'] * ((age - 20) / 64) * 3.5
    for k in SYMPTOM_KEYS:
        if features.get(k): s += WEIGHTS[k] * 3.2
    for k in RISK_KEYS:
        if features.get(k): s += WEIGHTS[k] * 2.8
    raw = min(1.0, s / 1.2)
    sigmoid = 1 / (1 + math.exp(-10 * (raw - 0.45)))
    return round(sigmoid * 100)

# ── DIET PLAN ────────────────────────────────────────────────────────
VEG_BREAKFAST  = ['Oats porridge with banana','Idli (2) with sambar','Upma with vegetables',
                  'Poha with peas','Whole wheat dosa','Moong dal chilla','Brown bread with peanut butter']
NONVEG_BREAKFAST = ['Egg white omelette with brown bread','Boiled eggs (2) with oats',
                    'Egg bhurji with roti','Chicken sandwich','Egg omelette with veggies',
                    'Poha with boiled eggs','Scrambled eggs with toast']
VEG_LUNCH    = ['Brown rice + dal + sabzi + salad','Roti + palak paneer + raita',
                'Millet khichdi with vegetables','Brown rice + rajma + salad',
                'Roti + chana masala + onion salad','Jowar roti + moong dal + veggies',
                'Brown rice + sambar']
NONVEG_LUNCH = ['Brown rice + grilled fish + salad','Roti + chicken curry (less oil)',
                'Brown rice + dal + boiled chicken','Roti + fish curry + cucumber salad',
                'Brown rice + egg curry + steamed veggies','Chicken soup + brown rice',
                'Roti + grilled chicken + dal']
VEG_DINNER   = ['Moong dal khichdi + veggies','Roti + mixed veg + curd',
                'Vegetable soup + brown bread','Oats khichdi + raita',
                'Roti + dal + salad','Vegetable stew + brown rice',
                'Tomato soup + roti + paneer (light)']
NONVEG_DINNER = ['Chicken soup + roti + salad','Grilled fish + steamed vegetables',
                 'Roti + egg curry (light)','Chicken broth + brown rice',
                 'Fish curry (light) + roti','Boiled chicken salad + roti',
                 'Egg white omelette + vegetable soup']
SNACKS       = ['Banana + walnuts','Apple with peanut butter','Roasted chana',
                'Coconut water + 2 dates','Cucumber and carrot sticks',
                'Curd with seeds','Green tea + 2-3 almonds']
MORNING_DRINKS = ['Warm water with lemon','Jeera water (warm)','Warm turmeric water',
                  'Green tea (no sugar)','Warm water with tulsi',
                  'Coriander seed water','Warm ginger water']
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════

# Header
st.markdown("# ❤️ Heart Health Check")
st.markdown("**Check your heart risk in 2 minutes — free, private, no login needed**")
st.markdown("---")

# ── STEP 1: ABOUT YOU ────────────────────────────────────────────────
st.markdown("## 👤 Step 1 — About You")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Your Age", min_value=20, max_value=84, value=45, step=1)
    if age < 40:
        st.success("👍 Lower risk age group")
    elif age < 55:
        st.warning("⚠️ Medium risk age group")
    else:
        st.error("⚠️ Higher risk age group")
with col2:
    gender = st.radio("Your Gender", ["♂️ Male", "♀️ Female"], horizontal=True)
    gender_val = 1 if "Male" in gender else 0

st.markdown("---")

# ── STEP 2: SYMPTOMS ─────────────────────────────────────────────────
st.markdown("## 🩺 Step 2 — How Are You Feeling?")
st.caption("Select everything you have been feeling lately")

col1, col2 = st.columns(2)
symptoms = {}
with col1:
    symptoms['Chest_Pain']          = st.checkbox("💔 Chest Pain or Tightness")
    symptoms['Shortness_of_Breath'] = st.checkbox("😮‍💨 Hard to Breathe")
    symptoms['Fatigue']             = st.checkbox("😴 Very Tired All the Time")
    symptoms['Palpitations']        = st.checkbox("💓 Heart Beating Fast")
with col2:
    symptoms['Dizziness']           = st.checkbox("🌀 Feeling Dizzy")
    symptoms['Swelling']            = st.checkbox("🦵 Swollen Legs or Feet")
    symptoms['Pain_Arms_Jaw_Back']  = st.checkbox("🦴 Pain in Arm, Jaw or Back")
    symptoms['Cold_Sweats_Nausea']  = st.checkbox("🤢 Cold Sweat or Nausea")

st.markdown("---")

# ── STEP 3: RISK FACTORS ─────────────────────────────────────────────
st.markdown("## ⚠️ Step 3 — Your Health & Lifestyle")
st.caption("Select anything that applies to you")

col1, col2 = st.columns(2)
risk_factors = {}
with col1:
    risk_factors['High_BP']             = st.checkbox("🩺 High Blood Pressure")
    risk_factors['High_Cholesterol']    = st.checkbox("🧪 High Cholesterol")
    risk_factors['Diabetes']            = st.checkbox("💉 Diabetes")
    risk_factors['Smoking']             = st.checkbox("🚬 Smoking")
with col2:
    risk_factors['Obesity']             = st.checkbox("⚖️ Overweight / Obese")
    risk_factors['Sedentary_Lifestyle'] = st.checkbox("🛋️ Not Much Exercise")
    risk_factors['Family_History']      = st.checkbox("👪 Family Heart Problems")
    risk_factors['Chronic_Stress']      = st.checkbox("😰 Always Stressed")

st.markdown("---")

# ── STEP 4: FOOD HABITS ──────────────────────────────────────────────
st.markdown("## 🍽️ Step 4 — Your Food Habits")
st.caption("Tell us how you eat — we will make a diet plan just for you")

col1, col2 = st.columns(2)
with col1:
    food_type = st.selectbox("🥗 What kind of food do you eat?",
                             ["🥦 Vegetarian", "🍗 Non-Vegetarian", "🌱 Vegan", "🥚 Eggetarian"])
    spice = st.selectbox("🌶️ How spicy or oily is your food?",
                         ["😊 Mild", "🌶️ Medium", "🔥 Very Spicy & Oily"])
with col2:
    outside = st.selectbox("🏠 How often do you eat outside?",
                           ["🏠 Rarely", "🛵 Sometimes", "🍔 Almost Daily"])
    meals = st.selectbox("🍚 How many meals per day?",
                         ["2 Meals", "3 Meals", "4+ Meals"])

st.caption("Select any bad food habits (optional)")
col1, col2, col3 = st.columns(3)
with col1:
    f_alcohol   = st.checkbox("🍺 Drink Alcohol")
    f_sweets    = st.checkbox("🍬 Eat Lots of Sweets")
with col2:
    f_salt      = st.checkbox("🧂 Add Extra Salt")
    f_fried     = st.checkbox("🍟 Eat Fried Food Often")
with col3:
    f_skips     = st.checkbox("⏭️ Skip Breakfast")
    f_latenight = st.checkbox("🌙 Eat Late at Night")

st.markdown("---")

# ── ANALYZE BUTTON ───────────────────────────────────────────────────
st.markdown("🔒 *Your answers stay on this page — nothing is sent anywhere*")
analyze = st.button("❤️ Check My Heart Health + Get My Diet Plan",
                    type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════
if analyze:

    all_features = {**symptoms, **risk_factors}
    score = calc_score(age, all_features)

    st.markdown("---")
    st.markdown("## 📊 Your Results")

    # Risk level
    if score >= 65:
        level, color_class, emoji = "High Risk", "high-risk", "🚨"
    elif score >= 40:
        level, color_class, emoji = "Moderate Risk", "mid-risk", "⚠️"
    else:
        level, color_class, emoji = "Low Risk", "low-risk", "✅"

    # Score display
    st.markdown(f"""
    <div class="risk-box {color_class}">
        <h1 style="font-size:3.5rem; margin:0">{score}%</h1>
        <h2 style="margin:0">{emoji} {level}</h2>
        <p>Based on {sum(all_features.values())} symptoms & risk factors · Age {age}</p>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.progress(score / 100)

    # Summary columns
    sx_count = sum(symptoms.values())
    rx_count = sum(risk_factors.values())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symptoms Found", f"{sx_count}/8")
    col2.metric("Risk Factors",   f"{rx_count}/8")
    col3.metric("Age Group",      "Higher" if age>=55 else "Medium" if age>=40 else "Lower")
    col4.metric("Diet Type",      food_type.split()[1] if len(food_type.split())>1 else food_type)

    st.markdown("---")

    # ── ADVICE ───────────────────────────────────────────────────────
    st.markdown("## 💡 What You Should Do")

    if score >= 65:
        st.error("🚨 **High Risk — Please See a Doctor Soon**")
        st.markdown("""
- 🏥 Book an appointment with your doctor **this week** — do not wait
- 🚑 If you have sudden chest pain or can't breathe, **call an ambulance right away**
- 💊 Tell your doctor all the symptoms and conditions you selected above
- 🚶 Try to walk for at least **10 minutes every day**, even if slowly
- 🧘 Try to reduce stress — even simple deep breathing helps your heart
- 🥗 Cut down on fried food, salt, and oily curries immediately
        """)
    elif score >= 40:
        st.warning("⚠️ **Moderate Risk — Take Action Now**")
        st.markdown("""
- 📅 Visit your doctor for a heart check-up **within the next month**
- 🥗 Eat more fruits, vegetables and home-cooked food — less fried and salty food
- 🚶 Walk for **30 minutes every day** — even a slow walk is great for the heart
- 💧 Drink plenty of water — avoid too much tea, coffee or soft drinks
- 😴 Sleep **7–8 hours** every night — poor sleep makes heart disease worse
- 📉 Check your blood pressure and cholesterol every 6 months
        """)
    else:
        st.success("✅ **Low Risk — Keep It Up!**")
        st.markdown("""
- 😊 Great news! Your heart risk looks low based on what you shared
- 🚶 Keep being active — even a 20-minute walk every day keeps your heart strong
- 🥗 Keep eating healthy food and drink enough water every day
- 🩺 Still get a regular health check **every year** — prevention is better than cure
- 😄 Keep stress low — enjoy time with family, hobbies and rest
        """)

    st.markdown("---")

    # ── FACTOR CONTRIBUTION BARS ─────────────────────────────────────
    st.markdown("## 📈 What Is Affecting Your Score")
    st.caption("These are the biggest factors in your result")

    contribs = []
    for k, w in WEIGHTS.items():
        if k == 'Age':
            val = (age - 20) / 64
        else:
            val = 1 if all_features.get(k) else 0
        c = val * w
        if c > 0.003:
            contribs.append((k, c))

    contribs.sort(key=lambda x: -x[1])
    max_c = contribs[0][1] if contribs else 1

    nice = {
        'Age':'Your Age','Cold_Sweats_Nausea':'Cold Sweats / Nausea',
        'Fatigue':'Feeling Very Tired','Dizziness':'Dizziness',
        'Shortness_of_Breath':'Hard to Breathe','Pain_Arms_Jaw_Back':'Arm / Jaw / Back Pain',
        'Swelling':'Swollen Legs','Chest_Pain':'Chest Pain',
        'Palpitations':'Fast Heartbeat','Sedentary_Lifestyle':'Not Exercising',
        'Family_History':'Family Heart History','Chronic_Stress':'Long-term Stress',
        'Obesity':'Overweight','Smoking':'Smoking',
        'High_BP':'High Blood Pressure','High_Cholesterol':'High Cholesterol','Diabetes':'Diabetes'
    }

    for k, c in contribs[:8]:
        pct = c / max_c
        st.write(f"**{nice.get(k, k)}**")
        st.progress(pct)

    st.markdown("---")

    # ── DIET PLAN ─────────────────────────────────────────────────────
    st.markdown("## 🥗 Your Personal 7-Day Diet Plan")
    st.caption("Made just for you based on your food habits and heart health")

    is_nonveg = "Non-Veg" in food_type
    is_egg    = "Eggetarian" in food_type
    has_bp    = risk_factors.get('High_BP')
    has_db    = risk_factors.get('Diabetes')
    has_ch    = risk_factors.get('High_Cholesterol')

    # Foods to avoid
    avoid = []
    if has_bp or f_salt:   avoid += ['Extra Salt','Pickles','Papad','Salty Snacks']
    if has_ch or f_fried:  avoid += ['Fried Food','Excess Ghee','Red Meat']
    if has_db or f_sweets: avoid += ['Sugar','Sweets / Mithai','White Rice (excess)','Maida']
    if f_alcohol:          avoid += ['Alcohol','Beer','Wine']
    if "Daily" in outside: avoid += ['Fast Food','Samosa','Biryani (daily)']
    if "Spicy" in spice:   avoid += ['Very Oily Curries','Extra Chilli']
    if f_latenight:        avoid += ['Heavy Dinner after 9 PM']

    # Good foods
    good = ['Oats','Brown Rice','Fruits','Vegetables','Dal','Curd']
    if is_nonveg or is_egg: good += ['Grilled Fish','Boiled Chicken']
    if has_bp:  good += ['Banana','Spinach','Beetroot']
    if has_db:  good += ['Bitter Gourd','Methi','Whole Grains']
    if has_ch:  good += ['Walnuts','Flaxseeds','Garlic']
    good += ['Coconut Water','Green Tea','Jeera Water']

    avoid_html = "".join([f'<span class="avoid-chip">❌ {f}</span>' for f in list(dict.fromkeys(avoid))])
    good_html  = "".join([f'<span class="good-chip">✅ {f}</span>'  for f in list(dict.fromkeys(good))])

    st.markdown(f"**🚫 Foods to Avoid or Eat Less:**")
    st.markdown(avoid_html if avoid else "No specific foods to avoid — eat balanced!", unsafe_allow_html=True)
    st.markdown(f"**✅ Good Foods for Your Heart:**")
    st.markdown(good_html, unsafe_allow_html=True)

    st.markdown("💧 **Drink 8 glasses of water every day** — start morning with 1 warm glass before eating")
    st.markdown("---")

    breakfasts = NONVEG_BREAKFAST if (is_nonveg or is_egg) else VEG_BREAKFAST
    lunches    = NONVEG_LUNCH     if is_nonveg             else VEG_LUNCH
    dinners    = NONVEG_DINNER    if is_nonveg             else VEG_DINNER

    tab_labels = [f"{'🌅' if i==0 else '🌤️' if i==1 else '☀️' if i==2 else '🌞' if i==3 else '🍀' if i==4 else '🌺' if i==5 else '🎯'} {d[:3]}" for i, d in enumerate(DAYS)]
    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs):
        with tab:
            st.markdown(f"### {DAYS[i]}")
            st.markdown(f"""
<div class="meal-card">🌅 <b>Morning Drink (6–7 AM)</b><br>{MORNING_DRINKS[i]}<br><small>Drink before eating anything — good for heart</small></div>
<div class="meal-card">🍳 <b>Breakfast (7–9 AM)</b><br>{breakfasts[i]}{'  ⚠️ Don\'t skip!' if f_skips else ''}<br><small>{'Use no sugar' if has_db else 'Eat within 1 hour of waking up'}</small></div>
<div class="meal-card">🍱 <b>Lunch (12–1 PM)</b><br>{lunches[i]}<br><small>{'Use very little salt in cooking' if has_bp else 'Eat slowly — stop when 80% full'}</small></div>
<div class="meal-card">🍎 <b>Evening Snack (4–5 PM)</b><br>{SNACKS[i]}{'  (no sugar)' if has_db else ''}<br><small>A light snack stops overeating at dinner</small></div>
<div class="meal-card">🌙 <b>Dinner (7–8 PM)</b><br>{dinners[i]}{'  ⚠️ Eat before 8 PM!' if f_latenight else ''}<br><small>Keep dinner light — your heart works best when you sleep light</small></div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ This is a screening tool only. It does not replace a real doctor's advice. If you feel very unwell or have sudden chest pain, call emergency services right away.")