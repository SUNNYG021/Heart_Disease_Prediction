import io
import pickle
from datetime import datetime

import numpy as np
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ---------------------------------------------------------------------------
# Model + constants
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open('heart_model.sav', 'rb') as fh:
        return pickle.load(fh)


model = load_model()  # loaded once, not on every rerun

# Order must match the training columns
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Average value of each column in the training data (heart_disease_data.csv).
# Used to measure how far THIS patient is from the "average" patient.
TRAIN_MEANS = {'age': 54.3663, 'sex': 0.6832, 'cp': 0.967, 'trestbps': 131.6238,
               'chol': 246.264, 'fbs': 0.1485, 'restecg': 0.5281, 'thalach': 149.6469,
               'exang': 0.3267, 'oldpeak': 1.0396, 'slope': 1.3993, 'ca': 0.7294,
               'thal': 2.3135}

NAMES = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest pain type', 'trestbps': 'Resting blood pressure',
         'chol': 'Cholesterol', 'fbs': 'Fasting blood sugar', 'restecg': 'Resting ECG',
         'thalach': 'Maximum heart rate', 'exang': 'Exercise-induced angina',
         'oldpeak': 'ST depression (Oldpeak)', 'slope': 'ST slope',
         'ca': 'Major vessels', 'thal': 'Thalassemia test (Thal)'}

CP_TXT = {0: 'Asymptomatic (no typical chest-pain symptoms)', 1: 'Atypical angina',
          2: 'Non-anginal pain', 3: 'Typical angina'}
CP_EXP = {0: 'You reported no classic chest-pain symptoms. This can sound reassuring, but in the data this system learned from, many people with heart disease had no typical chest pain.',
          1: 'Chest discomfort that does not fully match the classic heart-related pattern.',
          2: 'Chest pain that is usually not caused by the heart, such as muscle or stomach-related pain.',
          3: 'Classic heart-related chest pain, usually felt as pressure or tightness during effort or stress.'}
ECG_TXT = {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}
ECG_EXP = {0: 'Your resting heart tracing looks normal.',
           1: 'The resting ECG shows a change in the ST-T wave, which can be a sign that the heart muscle is under strain.',
           2: 'The ECG suggests the main pumping chamber of the heart may be thickened or enlarged.'}
SLOPE_TXT = {0: 'Downsloping', 1: 'Flat', 2: 'Upsloping'}
SLOPE_EXP = {0: 'During exercise the ST segment of the ECG slopes downward, which can be a sign of reduced blood flow to the heart.',
             1: 'During exercise the ST segment stays flat, which is somewhat less healthy than an upward slope.',
             2: 'During exercise the ST segment slopes upward, which is the most reassuring pattern.'}
THAL_TXT = {0: 'Not recorded', 1: 'Fixed defect', 2: 'Normal', 3: 'Reversible defect'}
THAL_EXP = {0: 'No result was recorded for this test.',
            1: 'Part of the heart muscle gets little or no blood flow, even at rest (a permanent area of reduced supply).',
            2: 'Blood flow to the heart muscle looks normal on this test.',
            3: 'Blood flow to part of the heart is normal at rest but reduced during exertion.'}


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
def compute_risk(values):
    """Return (risk_percent, contributions).
    risk_percent = model's probability of class 0 (the diseased class in this dataset).
    contributions[feature] > 0 means the value pushes risk UP versus the average patient."""
    x = np.array([values])
    class0 = list(model.classes_).index(0)
    risk = model.predict_proba(x)[0][class0] * 100
    coef = model.coef_[0]  # coefficients point towards class 1 (healthy), so flip the sign
    contrib = {}
    for i, f in enumerate(FEATURES):
        contrib[f] = float(-coef[i] * (values[i] - TRAIN_MEANS[f]))
    return risk, contrib


# Age and sex cannot be changed and their model weights are not reliable,
# so they are kept out of the "what is raising your risk" bars and the report's top-3 list.
BACKGROUND = ('age', 'sex')


def split_reasons(contrib):
    """Return (raising, lowering) lists of (feature, contribution), largest first, background factors excluded."""
    ups = sorted([(f, c) for f, c in contrib.items() if f not in BACKGROUND and c > 0.05],
                 key=lambda kv: kv[1], reverse=True)
    downs = sorted([(f, c) for f, c in contrib.items() if f not in BACKGROUND and c < -0.05],
                   key=lambda kv: kv[1])
    return ups, downs


def risk_level(r):
    if r <= 30:
        return 'Low Risk', 'You do not need to visit a doctor immediately. Maintain a balanced diet and regular physical exercise.'
    if r <= 70:
        return 'Medium Risk', 'You should schedule a routine consultation with a doctor and actively limit your daily salt intake.'
    return 'High Risk', 'Immediate attention recommended. You should see a cardiologist immediately and strictly limit your salt and cholesterol intake.'


def explain(f, v):
    """Return (value shown to the patient, plain-language explanation)."""
    v_int = int(v) if f not in ('oldpeak',) else v
    if f == 'age':
        return f'{v_int} years', ('Heart disease becomes more common as people get older. Age is a risk factor '
                                   'doctors always consider, but it is not something you can change.')
    if f == 'sex':
        return ('Male' if v_int == 1 else 'Female'), ('On average, men develop heart disease earlier than women. '
                                                       'In the data this system learned from, men were more often in the higher-risk group.')
    if f == 'cp':
        return CP_TXT[v_int], CP_EXP[v_int]
    if f == 'trestbps':
        if v < 120: s = 'Normal range (below 120).'
        elif v < 130: s = 'Slightly elevated (120-129).'
        elif v < 140: s = 'High - stage 1 range (130-139).'
        else: s = 'High - stage 2 range (140 or above).'
        return f'{v_int} mm Hg', ('Pressure of the blood against artery walls while you are at rest. ' + s +
                                   ' Long-term high pressure strains the heart and arteries.')
    if f == 'chol':
        if v < 200: s = 'Desirable (below 200).'
        elif v < 240: s = 'Borderline high (200-239).'
        else: s = 'High (240 or above).'
        return f'{v_int} mg/dl', ('The amount of cholesterol in your blood. ' + s +
                                   ' Too much can build up inside arteries and narrow them.')
    if f == 'fbs':
        return ('Above 120 mg/dl' if v_int == 1 else '120 mg/dl or below'), (
            'Your blood sugar after fasting. ' +
            ('A value above 120 may point to raised blood sugar or diabetes, which increases heart risk.' if v_int == 1
             else 'This is within the usual range.'))
    if f == 'restecg':
        return ECG_TXT[v_int], ECG_EXP[v_int]
    if f == 'thalach':
        return f'{v_int} beats/min', ('The highest heart rate reached during a stress test. A low maximum can mean the heart '
                                       'cannot respond to effort as well as it should.')
    if f == 'exang':
        return ('Yes' if v_int == 1 else 'No'), (
            'Whether physical effort triggers chest pain. ' +
            ('Pain triggered by exertion is a warning sign that the heart may not be getting enough blood.' if v_int == 1
             else 'No exertion-triggered chest pain is a good sign.'))
    if f == 'oldpeak':
        if v <= 1: s = 'low (0-1).'
        elif v <= 2: s = 'moderate (1-2).'
        else: s = 'high (above 2).'
        return f'{v:.1f}', ('How far the ECG line drops during exercise compared with rest. Your value is ' + s +
                            ' Higher values suggest the heart muscle is not getting enough oxygen during effort.')
    if f == 'slope':
        return SLOPE_TXT[v_int], SLOPE_EXP[v_int]
    if f == 'ca':
        return f'{v_int} vessel(s)', ('Number of major blood vessels (0-3) in which a scan shows a problem. '
                                      + ('None were highlighted, which is a good sign.' if v_int == 0
                                         else 'The more vessels affected, the higher the heart-disease risk.'))
    if f == 'thal':
        return THAL_TXT[v_int], THAL_EXP[v_int]
    return str(v), ''


def thalach_note(age, v):
    expected = 220 - age
    pct = v / expected * 100
    if pct >= 85:
        return f'Roughly expected for your age (about {expected} beats/min is the usual maximum).'
    return f'Lower than usual for your age (about {expected} beats/min is the usual maximum).'


def build_pdf(values, risk, contrib):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                            topMargin=16*mm, bottomMargin=16*mm,
                            title='Heart Disease Prediction System - Patient Report')
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle('h1', parent=ss['Title'], fontSize=20, textColor=colors.HexColor('#b00020'))
    h2 = ParagraphStyle('h2', parent=ss['Heading2'], fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle('b', parent=ss['BodyText'], fontSize=9.5, leading=12.5)
    small = ParagraphStyle('s', parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor('#555555'))
    cell = ParagraphStyle('c', parent=body, fontSize=8.8, leading=11)
    cellb = ParagraphStyle('cb', parent=cell, fontName='Helvetica-Bold')

    level, advice = risk_level(risk)
    lvl_col = {'Low Risk': '#1b7f3b', 'Medium Risk': '#b7791f', 'High Risk': '#b00020'}[level]
    vmap = dict(zip(FEATURES, values))

    S = [Paragraph('Heart Disease Prediction System', h1),
         Paragraph('Patient Report', ParagraphStyle('sub', parent=body, alignment=1, fontSize=12)),
         Paragraph(datetime.now().strftime('Generated on %d %B %Y, %H:%M'),
                   ParagraphStyle('d', parent=small, alignment=1)),
         Spacer(1, 8)]

    S.append(Paragraph('1. Your result', h2))
    S.append(Paragraph(f'Calculated risk score: <b>{risk:.1f}%</b> &nbsp; - &nbsp; '
                       f'<font color="{lvl_col}"><b>{level}</b></font>', ParagraphStyle('r', parent=body, fontSize=13, leading=17)))
    S.append(Paragraph(advice, body))
    S.append(Paragraph('The score is the model\'s estimate of how likely your values are to match patients with heart '
                       'disease in its training data. It is a screening aid, not a diagnosis.', small))

    # Top contributors
    up = split_reasons(contrib)[0][:3]
    S.append(Paragraph('2. What is pushing your score up the most', h2))
    if up:
        rows = []
        for n, (f, c) in enumerate(up, 1):
            shown, _ = explain(f, vmap[f])
            rows.append([Paragraph(f'<b>{n}. {NAMES[f]}</b>', cell), Paragraph(shown, cell)])
        t = Table(rows, colWidths=[75*mm, 95*mm])
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f6b7b7')),
                               ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f9d3d3')),
                               ('BACKGROUND', (0, 2), (-1, -1), colors.HexColor('#fcebeb')),
                               ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cc9999')),
                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        S.append(t)
        S.append(Paragraph('Factors are ranked by how far they move your score compared with an average patient in the '
                           'training data. Number 1 is the biggest contributor. Age and sex also influence the score, '
                           'but they cannot be changed, so they are not ranked here.', small))
    else:
        S.append(Paragraph('None of your values pushed the score up noticeably compared with an average patient.', body))

    # Full table
    S.append(Paragraph('3. Every value explained', h2))
    rows = [[Paragraph('Parameter', cellb), Paragraph('Your value', cellb), Paragraph('What it means', cellb)]]
    for f in FEATURES:
        shown, exp = explain(f, vmap[f])
        if f == 'thalach':
            exp = exp + ' ' + thalach_note(vmap['age'], vmap['thalach'])
        rows.append([Paragraph(f'<b>{NAMES[f]}</b>', cell), Paragraph(shown, cell), Paragraph(exp, cell)])
    t = Table(rows, colWidths=[38*mm, 34*mm, 98*mm], repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5e5e5')),
                           ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#999999')),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')])]))
    S.append(t)

    S.append(Paragraph('4. Important', h2))
    S.append(Paragraph('This report comes from a machine-learning model trained on a small public dataset (303 patients, '
                       'ages 29-77). It is a screening and awareness tool and does <b>not</b> replace a doctor\'s diagnosis. '
                       'Reference ranges above are general adult guides. Please share this report with a qualified doctor, '
                       'and seek urgent medical help if you have chest pain, breathlessness or fainting.', body))
    doc.build(S)
    return buf.getvalue()


def bar_html(label, shown, frac, rank_first=False):
    """Red bar; the bigger the contribution the deeper the red."""
    frac = max(0.05, min(1.0, frac))
    # interpolate light red (244,166,166) -> deep red (176,0,32)
    r = int(244 + (176 - 244) * frac)
    g = int(166 + (0 - 166) * frac)
    b = int(166 + (32 - 166) * frac)
    badge = (' <span style="background:#b00020;color:#fff;border-radius:6px;padding:1px 8px;'
             'font-size:0.75rem;margin-left:6px;">Biggest contributor</span>') if rank_first else ''
    return (f'<div style="margin:8px 0;">'
            f'<div style="font-size:0.9rem;margin-bottom:3px;"><b>{label}</b> '
            f'<span style="opacity:.7">({shown})</span>{badge}</div>'
            f'<div style="background:rgba(128,128,128,.15);border-radius:6px;height:16px;">'
            f'<div style="width:{frac*100:.0f}%;background:rgb({r},{g},{b});height:16px;border-radius:6px;"></div>'
            f'</div></div>')


# ---- UI ----
st.set_page_config(page_title='Heart Disease Prediction System', page_icon='❤️')
st.title("❤️ Heart Disease Prediction System")

with st.form("patient_form"):
    age = st.number_input("Age", min_value=29, max_value=77, value=30,
                          help="Patient's age in years (29-77). The model was trained on patients in this age range, "
                               "so predictions outside it would not be reliable.")

    sex = st.selectbox("Sex", options=(0, 1), format_func=lambda x: "Female" if x == 0 else "Male")

    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0,
                         help="0 = Asymptomatic (no typical chest-pain symptoms), 1 = Atypical angina, 2 = Non-anginal pain (not heart-related), 3 = Typical angina.")

    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120,
                               help="Resting blood pressure in mm Hg on admission to the hospital. Normal is around 120.")

    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200,
                           help="Serum cholesterol in mg/dl. Healthy levels are generally under 200.")

    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=(0, 1),
                       format_func=lambda x: "Normal (<= 120)" if x == 0 else "High (> 120)",
                       help="A value greater than 120 mg/dl indicates potential diabetes, a major risk factor for heart disease.")

    restecg = st.number_input("Resting ECG results (0-2)", min_value=0, max_value=2, value=1,
                              help="0 = Normal, 1 = ST-T wave abnormality, 2 = Probable or definite left ventricular hypertrophy (enlarged heart muscle).")

    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150,
                              help="The maximum heart rate the patient achieved during a stress test. Lower maximums can indicate weaker heart function.")

    exang = st.selectbox("Exercise Induced Angina", options=(0, 1), format_func=lambda x: "No" if x == 0 else "Yes",
                         help="Does the patient experience chest pain directly triggered by physical exertion?")

    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0,
                              help="ST depression induced by exercise relative to rest. A higher number indicates an abnormal heartbeat during exercise.")

    slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1,
                            help="The slope of the peak exercise ST segment. 0 = Downsloping, 1 = Flat, 2 = Upsloping (the healthiest pattern).")

    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0,
                         help="Number of major blood vessels colored by fluoroscopy. A higher number means more blood vessels are blocked.")

    thal = st.number_input("Thal (0-3)", min_value=0, max_value=3, value=2,
                           help="Thalassemia test result. 1 = Fixed defect, 2 = Normal, 3 = Reversible defect (blood flow is normal at rest but reduced during exertion).")
    submitted = st.form_submit_button("Predict")

if submitted:
    vals = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    risk, contrib = compute_risk(vals)
    st.session_state['result'] = {'vals': vals, 'risk': risk, 'contrib': contrib,
                                  'pdf': build_pdf(vals, risk, contrib)}

# Result is kept in session_state so it stays on screen after the download button is pressed
if 'result' in st.session_state:
    res = st.session_state['result']
    risk, contrib, vals = res['risk'], res['contrib'], res['vals']
    vmap = dict(zip(FEATURES, vals))

    st.subheader(f"Calculated Risk Score: {risk:.2f}%")
    level, advice = risk_level(risk)
    if level == 'Low Risk':
        st.success(f"🟢 Low Risk: {advice}")
    elif level == 'Medium Risk':
        st.warning(f"🟡 Medium Risk: {advice}")
    else:
        st.error(f"🔴 High Risk: {advice}")

    # ---- Contributors ----
    ups, downs = split_reasons(contrib)
    st.subheader("What is raising your risk?")
    if ups:
        top = ups[0][1]
        html = ''.join(bar_html(NAMES[f], explain(f, vmap[f])[0], c / top, rank_first=(i == 0))
                       for i, (f, c) in enumerate(ups))
        st.markdown(html, unsafe_allow_html=True)
        st.caption("Longer and darker red = bigger contribution to your risk score, compared with an average "
                   "patient in the training data. Age and sex also influence the score but cannot be changed, "
                   "so they are not shown here.")
    else:
        st.write("None of your values raised the score noticeably compared with an average patient.")

    if downs:
        with st.expander("Factors that lowered your risk"):
            top_d = abs(downs[0][1])
            html = ''
            for f, c in downs:
                w = abs(c) / top_d * 100
                html += (f'<div style="margin:8px 0;"><div style="font-size:0.9rem;margin-bottom:3px;"><b>{NAMES[f]}</b> '
                         f'<span style="opacity:.7">({explain(f, vmap[f])[0]})</span></div>'
                         f'<div style="background:rgba(128,128,128,.15);border-radius:6px;height:16px;">'
                         f'<div style="width:{max(w, 5):.0f}%;background:#3aa76d;height:16px;border-radius:6px;"></div></div></div>')
            st.markdown(html, unsafe_allow_html=True)

    st.download_button("📄 Download patient report (PDF)", data=res['pdf'],
                       file_name="heart_report.pdf", mime="application/pdf")
    st.caption("This tool is a screening aid and does not replace a doctor's diagnosis.")
