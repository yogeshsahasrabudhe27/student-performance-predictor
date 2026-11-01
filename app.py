import streamlit as st, os, io, base64, pickle, json
import html as _html
import time
from datetime import datetime, date, timedelta
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Website title: change these two lines whenever you want ----
TITLE_LINE1 = "Student Performance"   # small dark line
TITLE_LINE2 = "Predictor"             # big blue italic line
PREDICT_TITLE = "📊 Academic Performance Insights"   # heading above the prediction card

st.set_page_config(page_title=f"{TITLE_LINE1} {TITLE_LINE2}", page_icon="🎓", layout="wide")

# Change this to your own time zone if needed, e.g. "Asia/Kolkata", "America/New_York"
TIMEZONE = "Asia/Kolkata"
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TIMEZONE)
except Exception:      # on Windows run: pip install tzdata
    TZ = None

# ---------- model: uses regressor.pickle from train.py if present, else built-in numbers ----------
COEF = {"Hours": 5.185, "Health": 1.509, "Understanding": 2.642}
INTERCEPT = 2.916
if os.path.exists("regressor.pickle"):
    try:
        with open("regressor.pickle", "rb") as f:
            _m = pickle.load(f)
        if len(_m.coef_) == 3:
            COEF = dict(zip(["Hours", "Health", "Understanding"], [float(c) for c in _m.coef_]))
            INTERCEPT = float(_m.intercept_)
    except Exception:
        pass

DARK_CSS = """
<style>
.stApp {background:#101827 !important;}
.stApp p, .stApp label, .stApp li {color:#F8FAFC;}
.l1, .welcome, .sec-title, .card-title, .panel h3, .ex-sub, .subj-head, .lbl, .card-sub, .ptext, .tmsg, .tagline, .rname, .actrow, .subj-name, .heathead {color:#F8FAFC !important;}
.ex-date, .actdate, .legend, .subj-count, .dlabels, .months {color:#94A3B8 !important;}
.l2 {color:#60A5FA !important;} .tagline i {background:#60A5FA !important;}
[class*="st-key-card"], .panel, .exam, [class*="st-key-tile_"] {background:#1E293B !important; border-color:#334155 !important;}
[class*="st-key-card_target"] {background:#1E293B !important; border:1px solid #334155 !important;}
[class*="st-key-card_target"] [class*="st-key-tile_"] {background:#101827 !important;}
[class*="st-key-card"] [data-testid="stNumberInput"], [class*="st-key-card"] [data-testid="stTextInput"], [class*="st-key-card"] [data-testid="stDateInput"] {background:#101827 !important; border-color:#334155 !important;}
[class*="st-key-card"] input {color:#F8FAFC !important;}
[class*="st-key-card"] [data-testid="stButton"] button {background:#60A5FA !important;}
[class*="st-key-card"] [data-testid="stButton"] button p, [class*="st-key-card"] [data-testid="stButton"] button span {color:#101827 !important;}
.cchip, .chip {background:#1E293B !important; color:#F8FAFC !important; border-color:#334155 !important;}
.bar, .dash .bar {background:#334155 !important;} .bar > span, .dash .bar > span {background:#60A5FA !important;}
.dot {background:#334155;} .dot.on {background:#60A5FA;} .rtrack {stroke:#334155 !important;} .ring-pct {fill:#F8FAFC !important;}
.actrow {border-color:#334155 !important;} .result {background:#101827 !important; color:#60A5FA !important;}
[data-testid="stCheckbox"] p, [data-testid="stRadio"] p, [data-testid="stToggle"] p {color:#F8FAFC !important;}
[class*="st-key-tile_sub_"] [data-testid="stHorizontalBlock"] {border-color:#334155 !important;}
.heatpanel, .blob, .foot {background:#1E293B !important;}
.cell {background:#334155;} .cell.fut {background:transparent;}
</style>
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,400;0,500;0,600;0,700;1,700&family=Caveat:wght@600&display=swap');
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
.stApp {background:linear-gradient(180deg,#e6f1fd,#d8e8fa);}
.stApp, .stApp input, .stApp button {font-family:'Poppins',sans-serif;}
.block-container {padding:0 !important; max-width:100% !important;}
.script {font-family:'Caveat','Segoe Script',cursive; font-weight:600;}
.hdr {position:relative; text-align:center; padding:26px 16px 0;}
.plane {position:absolute; top:28px; left:30px;}
.blob {position:absolute; top:0; right:0; width:270px; height:180px; background:#14305e; border-bottom-left-radius:100% 100%; color:#fff; text-align:center; padding:28px 18px 0 70px;}
.blob .script {font-size:25px; line-height:1.15; display:block; transform:rotate(-6deg);}
.blob .line {display:block; height:3px; width:90px; background:#5aa2ff; border-radius:3px; margin:6px auto 4px;}
.title-row {display:inline-flex; align-items:center; gap:12px;}
.ttl {text-align:left; line-height:1;}
.l1 {display:block; font-size:46px; font-weight:700; color:#14305e;}
.l2 {display:block; font-size:66px; font-weight:700; font-style:italic; color:#1f5fd1; margin-top:-4px;}
.tagline {margin-top:12px; font-size:18px; color:#1c2b4a; display:flex; align-items:center; justify-content:center; gap:14px;}
.tagline i {width:28px; height:2px; background:#14305e; display:block;}
.gallery, .charts {display:grid; grid-template-columns:1fr 1fr; gap:22px; width:min(1160px,92%); margin:26px auto 0;}
.pic {position:relative; height:230px; border-radius:14px; overflow:hidden; box-shadow:0 8px 22px rgba(20,48,94,.22);}
.pic img, .pic .art {width:100%; height:100%; object-fit:cover; display:block;}
.pic .note {position:absolute; font-size:27px; line-height:1.1;}
.pic.left .note {left:7%; top:20%; color:#14305e; transform:rotate(-6deg);}
.pic.right .note {right:4%; top:5%; font-size:25px; text-align:right; color:#fff; transform:rotate(-5deg); text-shadow:0 1px 6px rgba(0,0,0,.25);}
.panel {background:rgba(255,255,255,.82); border:1px solid #d3e3f7; border-radius:18px; padding:16px 18px; box-shadow:0 10px 30px rgba(20,48,94,.12);}
.panel h3 {margin:0 0 6px; font-size:17px; color:#14305e;}
.panel img.chart {width:100%; display:block;}
[class*="st-key-card"] {margin-top:28px; background:rgba(255,255,255,.8); border:1px solid #d3e3f7; border-radius:18px; padding:26px 26px 18px; box-shadow:0 10px 30px rgba(20,48,94,.14);}
.card-title {font-size:24px; font-weight:700; color:#14305e; display:flex; align-items:center; justify-content:center; gap:10px;}
.card-sub {text-align:center; margin:4px 0 14px; font-size:15px; color:#34466b;}
.lbl {font-size:13px; font-weight:600; color:#34466b; margin:10px 0 4px 6px;}
[class*="st-key-card"] [data-testid="stNumberInput"], [class*="st-key-card"] [data-testid="stTextInput"], [class*="st-key-card"] [data-testid="stDateInput"] {background:#f4f9ff; border:1.5px solid #a9c4ea; border-radius:999px; overflow:hidden; padding:0 8px; min-height:46px;}
[class*="st-key-card"] [data-testid="stNumberInput"]:focus-within, [class*="st-key-card"] [data-testid="stTextInput"]:focus-within, [class*="st-key-card"] [data-testid="stDateInput"]:focus-within {border-color:#1447a8; box-shadow:0 0 0 3px rgba(20,71,168,.15);}
[class*="st-key-card"] [data-testid="stNumberInput"] *, [class*="st-key-card"] [data-testid="stTextInput"] *, [class*="st-key-card"] [data-testid="stDateInput"] * {background:transparent !important; border:none !important; box-shadow:none !important;}
[class*="st-key-card"] [data-testid="stNumberInput"] input, [class*="st-key-card"] [data-testid="stTextInput"] input, [class*="st-key-card"] [data-testid="stDateInput"] input {color:#1c2b4a !important; padding-left:14px !important; font-size:14px;}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {display:none !important;}
[class*="st-key-card"] [data-testid="stButton"] {width:100% !important; margin-top:12px;}
[class*="st-key-card"] [data-testid="stButton"] button {width:100%; height:46px; border-radius:999px !important; background:#1447a8 !important; border:0 !important; color:#fff !important; font-weight:600;}
[class*="st-key-card"] [data-testid="stButton"] button:hover {background:#0f3a8f !important;}
[class*="st-key-card"] [data-testid="stButton"] button p, [class*="st-key-card"] [data-testid="stButton"] button span {color:#fff !important; font-size:15px;}
.result {margin-top:14px; padding:12px; border-radius:12px; background:#d9eafd; color:#1447a8; font-weight:600; font-size:18px; text-align:center;}
.tags {margin-top:8px; font-size:13px; color:#34466b; text-align:center;}
.chips {display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:10px;}
.chip {background:#eaf3ff; border:1px solid #c5dbf7; color:#14305e; border-radius:999px; padding:5px 12px; font-size:12px;}
.hint {margin-top:8px; font-size:12px; color:#4a5c80; text-align:center;}
.err {margin-top:12px; font-size:13px; color:#d62b2b; text-align:center;}
.err b {display:inline-block; width:16px; height:16px; line-height:16px; border-radius:50%; background:#d62b2b; color:#fff; font-size:11px; margin-right:6px;}
.qbox {margin-top:26px; padding:20px 26px; border-radius:18px; text-align:center; color:#fff; background:linear-gradient(135deg,#14305e,#1f5fd1); box-shadow:0 10px 30px rgba(20,48,94,.25);}
.qtitle {font-size:12px; letter-spacing:2px; text-transform:uppercase; color:#9fd0ff; margin-bottom:6px;}
.qtext {font-size:21px; font-weight:500; line-height:1.4;}
.st-key-quote [data-testid="stButton"] {display:flex; justify-content:center; margin-top:10px;}
.st-key-quote [data-testid="stButton"] button {border-radius:999px !important; border:1.5px solid #1447a8 !important; background:#fff !important; color:#1447a8 !important; font-weight:600;}
.st-key-quote [data-testid="stButton"] button p {color:#1447a8 !important;}
.clock {display:flex; flex-wrap:wrap; justify-content:center; gap:10px; margin-top:16px; padding:0 12px;}
.cchip {background:rgba(255,255,255,.85); border:1px solid #c5dbf7; color:#14305e; border-radius:999px; padding:6px 16px; font-size:14px; font-weight:600; box-shadow:0 4px 12px rgba(20,48,94,.10);}
.sec-title {width:min(1160px,92%); margin:44px auto 0; font-size:26px; font-weight:700; color:#14305e; text-align:center;}
.bar {height:14px; border-radius:999px; background:#d9eafd; overflow:hidden; margin-top:10px;}
.bar > span {display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,#1f5fd1,#39d353);}
.ptext {margin-top:6px; font-size:13px; color:#34466b;}
.tmsg {margin-top:12px; font-size:14px; color:#14305e; line-height:1.5;}
.okmsg {margin-top:10px; font-size:13px; color:#1a7f4b; text-align:center; font-weight:600;}
.heatpanel {background:#14305e; color:#dbe8ff; border-radius:18px; padding:18px 22px; box-shadow:0 10px 30px rgba(20,48,94,.25);}
.heathead {font-size:16px; font-weight:600; margin-bottom:12px;}
.heatscroll {overflow-x:auto; padding-bottom:6px;}
.heatwrap {display:flex; gap:8px; width:max-content;}
.dlabels {display:grid; grid-template-rows:repeat(7,14px); gap:4px; padding-top:20px; font-size:11px; color:#9fb6dd;}
.months {display:grid; grid-auto-flow:column; grid-auto-columns:14px; gap:4px; height:14px; margin-bottom:6px; font-size:11px; color:#9fb6dd;}
.months span {white-space:nowrap;}
.heat {display:grid; grid-auto-flow:column; grid-template-rows:repeat(7,14px); grid-auto-columns:14px; gap:4px;}
.cell {display:block; width:14px; height:14px; border-radius:3px; background:#2b4676;}
.cell.l1 {background:#3fae5a;} .cell.l2 {background:#4cd06b;} .cell.l3 {background:#6ee887;} .cell.l4 {background:#b6ffc6;}
.cell.fut {background:transparent;}
.legend {display:flex; justify-content:flex-end; align-items:center; gap:4px; margin-top:10px; font-size:11px; color:#9fb6dd;}
.st-key-card_exam, [class*="st-key-card_target"] {width:min(1100px,92%) !important; margin:20px auto 0 !important;}
.exam {display:flex; justify-content:space-between; align-items:center; gap:10px; background:#f4f9ff; border:1px solid #c5dbf7; border-radius:14px; padding:10px 14px;}
.ex-sub {font-size:16px; font-weight:700; color:#14305e;}
.ex-date {font-size:12px; color:#4a5c80; margin-top:2px;}
.ex-badge {border-radius:999px; padding:5px 12px; font-size:12px; font-weight:700; white-space:nowrap;}
.subj {margin:14px 0 6px;}
.subj-head {display:flex; justify-content:space-between; font-weight:700; color:#14305e; font-size:16px;}
[class*="st-key-card"] [class*="st-key-rm_"] [data-testid="stButton"] {width:auto !important; margin-top:0 !important;}
[class*="st-key-card"] [class*="st-key-rm_"] [data-testid="stButton"] button {width:36px; height:36px; min-height:36px; padding:0; background:#fff !important; border:1px solid #f0b4b4 !important;}
[class*="st-key-card"] [class*="st-key-rm_"] [data-testid="stButton"] button:hover {background:#fde8e8 !important;}
[class*="st-key-card"] [class*="st-key-rm_"] [data-testid="stButton"] button p, [class*="st-key-card"] [class*="st-key-rm_"] [data-testid="stButton"] button span {color:#d62b2b !important;}
.divider {border:0; border-top:1px solid #d3e3f7; margin:22px 0 14px;}
[class*="st-key-card_target"] {background:linear-gradient(135deg,#eaf6ff,#e3eefb) !important; border:1px solid #bae6fd !important;}
[class*="st-key-card_target"] .card-title {color:#075985;}
[class*="st-key-tile_"] {background:#fff; border:1px solid #cfe8f7; border-radius:16px; padding:16px 16px 12px; box-shadow:0 4px 14px rgba(2,132,199,.08);}
[class*="st-key-card_target"] [data-testid="stButton"] button {background:#0284c7 !important;}
[class*="st-key-card_target"] [data-testid="stButton"] button:hover {background:#0369a1 !important;}
[class*="st-key-card_target"] [data-testid="stNumberInput"], [class*="st-key-card_target"] [data-testid="stTextInput"] {border-color:#bae6fd !important; background:#f4faff !important;}
[class*="st-key-card_target"] .bar {background:#e0f2fe;}
[class*="st-key-card_target"] .bar > span {background:linear-gradient(90deg,#0ea5e9,#34d399);}
[class*="st-key-card_target"] .subj-head {color:#075985;}
.welcome {text-align:center; margin-top:16px; font-size:22px; font-weight:700; color:#14305e;}
.dash {display:grid; grid-template-columns:1fr 1.3fr 1fr; gap:22px; width:min(1160px,92%); margin:20px auto 0;}
.ringbox {display:flex; align-items:center; gap:14px;}
.dots {display:flex; gap:10px; justify-content:center; margin:12px 0 8px;}
.dot {width:22px; height:22px; border-radius:50%; background:#dfe5ef; display:block;}
.dot.on {background:#2b6be6;} .dot.today {outline:3px solid #ff9f43; outline-offset:2px;}
.rings {display:flex; flex-wrap:wrap; justify-content:space-around; gap:16px; margin:14px 0 4px;}
.rcell {text-align:center;} .rname {font-weight:600; color:#14305e; margin-top:2px;}
.actrow {display:flex; justify-content:space-between; gap:12px; padding:10px 4px; border-top:1px solid #e3ebf7; font-size:14px; color:#1c2b4a;}
.actdate {color:#6b7a99; font-size:12px; white-space:nowrap;}
@media (max-width:720px) {.dash {grid-template-columns:1fr;}}
.subj-row {display:flex; align-items:center; gap:14px; margin-bottom:6px;}
.subj-name {font-size:18px; font-weight:700; color:#111827;}
.subj-count {font-size:13px; color:#6b7280;}
[class*="st-key-tile_sub_"] [data-testid="stHorizontalBlock"] {border-top:1px solid #e5e7eb; padding-top:2px; margin-top:2px;}
.dash .bar {height:12px; background:#e2e6ee;}
.dash .bar > span {background:#2b6be6 !important;}
@keyframes ringfill {from {stroke-dashoffset:var(--c);} to {stroke-dashoffset:var(--off);}}
.ring-anim {animation:ringfill 1.4s ease-out;}
.ring-pct {fill:#14305e; font-size:28px; font-weight:700;}
.ringcol {display:flex; flex-direction:column; align-items:center; text-align:center; gap:4px;}
.ptext.big {font-size:19px; font-weight:700; color:#14305e; margin-top:6px;}
.ptext.rem {font-size:14px; color:#34466b;}
.panel.wide {grid-column:1 / -1;}
.panel {transition:transform .25s ease, box-shadow .25s ease;}
.panel:hover {transform:translateY(-3px);}
.sec-title::after {content:""; display:block; width:64px; height:4px; border-radius:4px; margin:8px auto 0; background:linear-gradient(90deg,#4aa8f0,#fb923c);}
.foot {margin-top:60px; background:#14305e; color:#fff; text-align:center; font-size:15px; padding:42px 16px 26px; border-radius:50% 50% 0 0 / 46px 46px 0 0;}
@media (max-width:720px) {
  .gallery, .charts {grid-template-columns:1fr;}
  .blob {width:170px; height:120px; padding:14px 10px 0 40px;}
  .blob .script {font-size:16px;}
  .plane {display:none;}
  .l1 {font-size:28px;} .l2 {font-size:40px;} .tagline {font-size:13px;}
}
</style>
"""

HEADER = """
<div class="hdr">
<svg class="plane" width="140" height="110" viewBox="0 0 140 110" fill="none" stroke="#14305e" stroke-width="2" stroke-linejoin="round"><path d="M84 8 L52 24 L72 30 L80 50 L90 36 L112 22 Z"/><path d="M72 30 L112 22"/><path d="M60 46 C 48 66, 40 70, 22 70" stroke-dasharray="4 5"/><circle cx="14" cy="66" r="9" stroke-dasharray="3 4"/></svg>
<div class="blob"><span class="script">Small<br>Steps<br>Big Dreams</span><span class="line"></span><svg width="18" height="16" viewBox="0 0 24 22"><path d="M12 21 C-4 10 4 -2 12 5 C20 -2 28 10 12 21Z" fill="#fff"/></svg></div>
<div class="title-row">
<svg width="68" height="52" viewBox="0 0 64 48"><path d="M32 4 L62 18 L32 32 L2 18 Z" fill="#14305e"/><path d="M14 26 V38 C14 43 50 43 50 38 V26 L32 35 Z" fill="#14305e"/><path d="M58 20 V37" stroke="#14305e" stroke-width="3"/></svg>
<div class="ttl"><span class="l1">__T1__</span><span class="l2">__T2__</span></div>
</div>
<div class="tagline"><i></i>Better Study | Higher Goals | Brighter Future<i></i></div>
</div>
"""

# Built-in illustrations: shown when there are no photos in the images folder
LEFT_ART = """<svg class="art" viewBox="0 0 600 230" preserveAspectRatio="xMidYMid slice"><defs><linearGradient id="gl" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f7e5bd"/><stop offset="1" stop-color="#d9a05c"/></linearGradient></defs><rect width="600" height="230" fill="url(#gl)"/><circle cx="80" cy="20" r="80" fill="#fff7dc" opacity=".55"/><rect y="150" width="600" height="80" fill="#a8672f" opacity=".45"/><path d="M335 124 C318 90 333 62 350 40 C362 72 375 98 335 124Z" fill="#4f9440"/><path d="M342 124 C362 96 390 86 404 76 C398 108 376 122 342 124Z" fill="#7cc062"/><path d="M95 205 L215 172 L225 215 L105 232Z" fill="#fffaf0"/><path d="M215 172 L300 190 L292 228 L225 215Z" fill="#fff1d6"/><path d="M130 200 L200 184 M135 212 L205 197" stroke="#c9b48c" stroke-width="2"/><path d="M180 192 L250 207" stroke="#3b6cf0" stroke-width="5" stroke-linecap="round"/><ellipse cx="335" cy="176" rx="38" ry="9" fill="#fff"/><rect x="313" y="140" width="44" height="34" rx="9" fill="#fff"/><ellipse cx="335" cy="141" rx="20" ry="5" fill="#5a3a22"/><path d="M357 148 C372 148 372 166 356 166" fill="none" stroke="#fff" stroke-width="5"/><rect x="392" y="128" width="34" height="44" rx="5" fill="#f5f0e8"/><path d="M400 128 L396 95" stroke="#e08a2b" stroke-width="4"/><path d="M410 128 L414 92" stroke="#3b6cf0" stroke-width="4"/><path d="M420 128 L430 100" stroke="#c0392b" stroke-width="4"/><rect x="440" y="146" width="140" height="24" fill="#2e2a25"/><rect x="450" y="124" width="124" height="22" fill="#c9a227"/><rect x="446" y="102" width="130" height="22" fill="#3a342d"/><path d="M470 100 L540 92" stroke="#e08a2b" stroke-width="5" stroke-linecap="round"/><rect x="420" y="186" width="62" height="42" rx="5" fill="#d9d9d4"/><rect x="427" y="191" width="48" height="10" fill="#8fb8a0"/><g fill="#777"><circle cx="435" cy="210" r="3"/><circle cx="451" cy="210" r="3"/><circle cx="467" cy="210" r="3"/><circle cx="435" cy="221" r="3"/><circle cx="451" cy="221" r="3"/><circle cx="467" cy="221" r="3"/></g></svg>"""

RIGHT_ART = """<svg class="art" viewBox="0 0 600 230" preserveAspectRatio="xMidYMid slice"><defs><linearGradient id="gr" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1b3fa0"/><stop offset=".6" stop-color="#3b6cf0"/><stop offset="1" stop-color="#7b5cf0"/></linearGradient></defs><rect width="600" height="230" fill="url(#gr)"/><rect y="172" width="600" height="58" fill="#6f4be0" opacity=".75"/><rect x="18" y="148" width="34" height="30" rx="4" fill="#4fa3e0"/><path d="M35 148 C20 120 25 105 35 95 C45 110 50 130 35 148Z" fill="#3fae5a"/><path d="M35 148 C55 125 70 120 78 118 C72 138 55 146 35 148Z" fill="#66c47a"/><path d="M62 178 C62 128 150 128 150 178Z" fill="#1f2f78"/><circle cx="106" cy="92" r="26" fill="#f1c29a"/><path d="M80 90 C78 62 134 58 132 90 C120 78 98 76 80 90Z" fill="#35251d"/><circle cx="98" cy="94" r="2.5" fill="#222"/><circle cx="114" cy="94" r="2.5" fill="#222"/><path d="M99 104 Q106 110 113 104" fill="none" stroke="#8a4b2a" stroke-width="2"/><path d="M168 172 L184 112 L290 112 L300 172Z" fill="#a9b3cf"/><rect x="150" y="168" width="164" height="10" rx="4" fill="#d5dbea"/><circle cx="156" cy="168" r="9" fill="#f1c29a"/><rect x="205" y="40" width="52" height="52" rx="12" fill="#2a63d8" opacity=".85" stroke="#7fe3ff"/><circle cx="231" cy="66" r="14" fill="none" stroke="#7fe3ff" stroke-width="3"/><path d="M231 52 V80 M217 66 H245" stroke="#7fe3ff" stroke-width="2"/><rect x="275" y="36" width="130" height="90" rx="8" fill="#1d4fc4" stroke="#7fb4ff"/><path d="M288 112 L312 96 L334 102 L362 70 L388 52" fill="none" stroke="#8ef0ff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><path d="M386 44 L402 50 L392 63Z" fill="#8ef0ff"/><rect x="400" y="182" width="80" height="14" fill="#20a77a"/><rect x="396" y="196" width="88" height="14" fill="#3b8cf0"/><circle cx="440" cy="150" r="30" fill="#ffcf33"/><circle cx="440" cy="150" r="23" fill="#fff"/><path d="M440 150 V134 M440 150 L452 156" stroke="#14305e" stroke-width="3" stroke-linecap="round"/><rect x="512" y="160" width="30" height="34" rx="4" fill="#f5c518"/><path d="M520 160 L516 142 M530 160 L536 138 M538 160 L548 146" stroke="#ff7a59" stroke-width="4" stroke-linecap="round"/></svg>"""

GALLERY = """
<div class="gallery">
<div class="pic left">__LEFT__<span class="note script">Plan<br>Study<br>Achieve</span></div>
<div class="pic right">__RIGHT__<span class="note script">Better Study<br>Bigger Results</span></div>
</div>
"""

CARD_TOP = """
<div class="card-title"><svg width="30" height="30" viewBox="0 0 32 32" fill="none" stroke="#14305e" stroke-width="2.4"><circle cx="16" cy="16" r="12"/><circle cx="16" cy="16" r="6.5"/><circle cx="16" cy="16" r="1.5" fill="#14305e"/><path d="M16 16 L28 4"/></svg>Predict Your Exam Marks</div>
<div class="card-sub">Enter your study hours, understanding and health</div>
"""


QUOTES = {
    "general": [
        "Small steps every day lead to big results.",
        "The expert in anything was once a beginner.",
        "Don’t watch the clock; do what it does. Keep going.",
        "Study a little today, and tomorrow you will thank yourself.",
        "Your future is created by what you do today, not tomorrow.",
    ],
    "low": [
        "A low score is a starting line, not a finish line. You can change it.",
        "Add one more focused hour today. Small changes add up fast.",
        "Understand one topic deeply today; tomorrow it will feel easier.",
        "Every topper once struggled. Keep going, you are closer than you think.",
    ],
    "mid": [
        "You are on the right track. A little more effort will lift your marks.",
        "Good progress! Now fix your weakest topic and watch the score rise.",
        "Consistency beats intensity. Keep your daily routine going.",
        "You are halfway to your goal. Don’t stop now.",
    ],
    "high": [
        "Excellent! Hard work and good habits are paying off. Keep it up.",
        "Great routine! Stay healthy and keep revising, and success will follow.",
        "Discipline is what turns talent into results. Well done.",
        "Aim high, stay humble, and keep learning.",
    ],
}


@st.cache_data
def load_image(names):
    """Return an <img> tag for the first existing photo in names, else empty text."""
    for name in names:
        if os.path.exists(name):
            mime = "image/png" if name.lower().endswith(".png") else "image/jpeg"
            data = base64.b64encode(open(name, "rb").read()).decode()
            return f'<img src="data:{mime};base64,{data}" alt="">'
    return ""


# ---------- prediction and charts ----------
def predict(hours, health, understanding):
    m = INTERCEPT + COEF["Hours"] * hours + COEF["Health"] * health + COEF["Understanding"] * understanding
    return float(min(max(m, 0), 100))   # marks stay between 0 and 100


def level(v, words):
    return words[0] if v <= 3 else words[1] if v <= 6 else words[2]


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return '<img class="chart" src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def sim_chart(hours, health, understanding, marks):
    xs = np.linspace(1, 10, 60)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if os.path.exists("data.csv"):
        try:
            df = pd.read_csv("data.csv")
            ax.scatter(df["Hours"], df["Scores"], s=12, color="#9db4d6", alpha=.55, label="Sample data")
        except Exception:
            pass
    ax.plot(xs, [predict(x, 10, 10) for x in xs], "--", color="#22a06b", lw=2, label="Best health + understanding")
    ax.plot(xs, [predict(x, 3, 3) for x in xs], ":", color="#e0752d", lw=2.2, label="Low health + understanding")
    ax.plot(xs, [predict(x, health, understanding) for x in xs], color="#1447a8", lw=3, label="Your levels")
    ax.scatter([min(hours, 10)], [marks], s=150, color="#f5b301", edgecolor=CHART_TXT, zorder=5, label="You")
    ax.set_ylim(0, 105); ax.set_xlabel("Study hours per day"); ax.set_ylabel("Predicted marks")
    ax.legend(fontsize=7, loc="lower right", frameon=False, labelcolor=CHART_TXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for item in [ax.xaxis.label, ax.yaxis.label]:
        item.set_color(CHART_TXT)
    ax.tick_params(colors=CHART_TXT)
    return fig_to_img(fig)


def contrib_chart(hours, health, understanding):
    parts = {"Base": INTERCEPT, "Study hours": COEF["Hours"] * hours,
             "Health": COEF["Health"] * health, "Understanding": COEF["Understanding"] * understanding}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.barh(list(parts.keys()), list(parts.values()), color=["#9db4d6", "#f5b301", "#22a06b", "#3b6cf0"])
    ax.bar_label(bars, fmt="%.1f", padding=3, color=CHART_TXT, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Marks contributed")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.label.set_color(CHART_TXT); ax.tick_params(colors=CHART_TXT)
    return fig_to_img(fig)


@st.fragment(run_every=1)
def live_clock():
    """Shows greeting, day, date and a ticking time (refreshes every second)."""
    now = datetime.now(TZ) if TZ else datetime.now()
    hr = now.hour
    greet, icon = (("Good morning", "🌅") if hr < 12 else ("Good afternoon", "☀️") if hr < 17
                   else ("Good evening", "🌆") if hr < 21 else ("Good night", "🌙"))
    st.markdown(f'<div class="welcome">{icon} {greet}! Ready to study?</div><div class="clock">'
                f'<span class="cchip">📅 {now:%A}, {now:%d %B %Y}</span>'
                f'<span class="cchip">🕒 {now:%I:%M:%S %p}</span></div>', unsafe_allow_html=True)


# ---------- planner: exams, topics, progress ----------
PROGRESS_FILE = "progress.json"


def today_date():
    return datetime.now(TZ).date() if TZ else date.today()


def new_id():
    return str(int(time.time() * 1000)) + os.urandom(2).hex()


def load_progress():
    p = {"exams": [], "subjects": {}, "target_hours": None, "log": {}, "mocks": {}}
    try:
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for k, typ in (("exams", list), ("subjects", dict), ("log", dict), ("mocks", dict)):
            if isinstance(data.get(k), typ):
                p[k] = data[k]
        if isinstance(data.get("target_hours"), (int, float)):
            p["target_hours"] = float(data["target_hours"])
    except Exception:
        pass
    return p


def save_progress(p):
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(p, f)
    except Exception:
        pass


def show_msg(key):
    msg = st.session_state.pop(key, None)
    if msg:
        text = _html.escape(msg[1], quote=False)
        if msg[0] == "ok":
            st.markdown(f'<div class="okmsg">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="err"><b>!</b>{text}</div>', unsafe_allow_html=True)


def is_done(hours, target):
    return bool(target) and hours is not None and hours >= target


def save_target():
    th = st.session_state.get("t_hours")
    if th is None:
        st.session_state["tg_msg"] = ("err", "Enter your daily study hours.")
        return
    p = load_progress()
    p["target_hours"] = float(th)
    save_progress(p)
    st.session_state["tg_msg"] = ("ok", "🎯 Target set! Log your study hours every day.")


def log_today():
    h = st.session_state.get("log_hours")
    if h is None:
        st.session_state["lg_msg"] = ("err", "Enter the hours you studied today.")
        return
    p = load_progress()
    p["log"][today_date().isoformat()] = float(h)
    save_progress(p)
    if is_done(h, p.get("target_hours")):
        st.session_state["lg_msg"] = ("ok", "✅ Completed! You reached today’s target. Great work!")
    else:
        st.session_state["lg_msg"] = ("ok", "Saved. Study a little more to complete today’s target.")


def save_mock():
    m = st.session_state.get("mock_marks")
    if m is None:
        st.session_state["mk_msg"] = ("err", "Enter your mock test marks.")
        return
    p = load_progress()
    p["mocks"][today_date().isoformat()] = float(m)
    save_progress(p)
    st.session_state["mk_msg"] = ("ok", f"🧪 Mock test marks saved: {m:g}%")


# ----- exams -----
def add_exam():
    sub = (st.session_state.get("ex_sub") or "").strip()
    d = st.session_state.get("ex_date")
    if not sub or d is None:
        st.session_state["e_msg"] = ("err", "Enter the exam subject and choose the exam date.")
        return
    p = load_progress()
    p["exams"].append({"id": new_id(), "subject": sub, "date": d.isoformat()})
    save_progress(p)
    st.session_state["ex_sub"] = ""
    st.session_state["ex_date"] = None
    st.session_state["e_msg"] = ("ok", "📅 Exam saved. All the best!")


def remove_exam(eid):
    p = load_progress()
    p["exams"] = [e for e in p["exams"] if e.get("id") != eid]
    save_progress(p)


def exam_row_html(e, today):
    try:
        d = date.fromisoformat(e["date"])
    except Exception:
        return ""
    n = (d - today).days
    if n < 0:
        txt, bg, fg = "Finished", "#e6e9f0", "#5b6785"
    elif n == 0:
        txt, bg, fg = "Today! All the best 🍀", "#fde2e2", "#c02626"
    elif n <= 7:
        txt, bg, fg = f"{n} day{'s' if n != 1 else ''} left", "#fde2e2", "#c02626"
    elif n <= 14:
        txt, bg, fg = f"{n} days left", "#fff1cc", "#8a5a00"
    else:
        txt, bg, fg = f"{n} days left", "#d9f5e6", "#1a7f4b"
    return (f'<div class="exam"><div><div class="ex-sub">📘 {_html.escape(e.get("subject", ""))}</div>'
            f'<div class="ex-date">{d:%A, %d %B %Y}</div></div>'
            f'<span class="ex-badge" style="background:{bg};color:{fg}">{txt}</span></div>')


# ----- subjects and topics -----
def add_topic():
    sub = (st.session_state.get("ts_sub") or "").strip()
    name = (st.session_state.get("ts_topic") or "").strip()
    if not sub or not name:
        st.session_state["t_msg"] = ("err", "Enter both the subject name and the topic.")
        return
    p = load_progress()
    key = next((k for k in p["subjects"] if k.lower() == sub.lower()), sub)
    p["subjects"].setdefault(key, []).append({"id": new_id(), "name": name, "done": None})
    save_progress(p)
    st.session_state["ts_topic"] = ""
    st.session_state["t_msg"] = ("ok", f"Added “{name}” to {key}.")


def toggle_topic(tid):
    checked = bool(st.session_state.get(f"chk_{tid}"))
    p = load_progress()
    for topics in p["subjects"].values():
        for t in topics:
            if t.get("id") == tid:
                t["done"] = today_date().isoformat() if checked else None
    save_progress(p)


def remove_topic(tid):
    p = load_progress()
    for sub in list(p["subjects"]):
        p["subjects"][sub] = [t for t in p["subjects"][sub] if t.get("id") != tid]
        if not p["subjects"][sub]:
            del p["subjects"][sub]
    save_progress(p)


def done_counts(p):
    c = {}
    for topics in p["subjects"].values():
        for t in topics:
            if t.get("done"):
                c[t["done"]] = c.get(t["done"], 0) + 1
    return c


def current_streak(p, today):
    """A day counts when the hours target is reached or at least one topic is ticked."""
    counts, th, log = done_counts(p), p.get("target_hours"), p.get("log", {})

    def ok(d):
        iso = d.isoformat()
        return is_done(log.get(iso), th) or counts.get(iso, 0) > 0

    d = today if ok(today) else today - timedelta(days=1)    # today is not over yet
    n = 0
    while ok(d):
        n += 1
        d -= timedelta(days=1)
    return n


def progress_html(p, res, today):
    th, log, mocks = p.get("target_hours"), p.get("log", {}), p.get("mocks", {})
    topics = [t for ts in p["subjects"].values() for t in ts]
    if not (th or topics or mocks):
        return '<div class="panel"><h3>📊 My progress</h3><div class="tmsg">Set your target, add topics or save a mock test above to start tracking.</div></div>'
    parts = []
    if th:
        h_today = float(log.get(today.isoformat(), 0) or 0)
        pct = h_today / th * 100
        status = "✅ Completed today!" if h_today >= th else "⏳ In progress"
        parts.append(f'<div class="ptext" style="font-weight:600;color:#14305e">⏱ Study hours: {status}</div>'
                     f'<div class="bar"><span style="width:{min(pct, 100):.0f}%"></span></div>'
                     f'<div class="ptext">{h_today:g} / {th:g} hours &nbsp;•&nbsp; {pct:.0f}%</div>')
    if topics:
        done = sum(1 for t in topics if t.get("done"))
        tp = done / len(topics) * 100
        parts.append(f'<div class="ptext" style="margin-top:12px;font-weight:600;color:#14305e">📚 Topics: {done}/{len(topics)} completed</div>'
                     f'<div class="bar"><span style="width:{tp:.0f}%"></span></div>')
    chips = (f'<span class="chip">🔥 Streak: {current_streak(p, today)} days</span>'
             f'<span class="chip">⏱ Total: {sum(float(v) for v in log.values()):g} h</span>')
    lines = []
    if mocks:
        vals = [float(mocks[k]) for k in sorted(mocks)]
        latest = vals[-1]
        trend = ""
        if len(vals) >= 2:
            diff = latest - vals[-2]
            trend = f" ({'▲' if diff > 0 else '▼' if diff < 0 else '='} {abs(diff):g})"
        chips += f'<span class="chip">🧪 Latest mock: {latest:g}%</span>'
        lines.append(f"🧪 Mock tests: latest {latest:g}%{trend}, average {sum(vals) / len(vals):.1f}%, "
                     f"best {max(vals):g}% ({len(vals)} test{'s' if len(vals) != 1 else ''}).")
        if res:
            pm = predict(res[0], res[1], res[2])
            gap = pm - latest
            if gap > 3:
                lines.append(f"Your predicted marks ({pm:.1f}%) are higher than your latest mock test. Revise weak topics to close the gap.")
            elif gap < -3:
                lines.append(f"Your latest mock test beats the predicted {pm:.1f}%. Great progress!")
            else:
                lines.append(f"Your mock test marks are close to your predicted {pm:.1f}%. Keep going!")
    notes = "".join(f'<div class="tmsg">{l}</div>' for l in lines)
    return ('<div class="panel"><h3>📊 My progress</h3>' + "".join(parts)
            + f'<div class="chips">{chips}</div>{notes}</div>')


def heatmap_html(p, today, mode="topics", weeks=26):
    counts, log, th = done_counts(p), p.get("log", {}), p.get("target_hours")
    scale = th or 4.0
    start = today - timedelta(days=today.weekday()) - timedelta(weeks=weeks - 1)
    cells, months, total = [], [], 0.0
    for w in range(weeks):
        ws = start + timedelta(weeks=w)
        prev = start + timedelta(weeks=w - 1)
        months.append(f"<span>{ws:%b}</span>" if w == 0 or ws.month != prev.month else "<span></span>")
        for k in range(7):
            d = ws + timedelta(days=k)
            if d > today:
                cells.append('<i class="cell fut"></i>')
                continue
            if mode == "hours":
                h = float(log.get(d.isoformat(), 0) or 0)
                total += h
                r = h / scale
                lvl = 0 if h <= 0 else 1 if r < .5 else 2 if r < .8 else 3 if r < 1 else 4
                tip = f"{d:%a %d %b %Y}: {h:g} h studied"
            else:
                c = counts.get(d.isoformat(), 0)
                total += c
                lvl = 0 if c == 0 else 1 if c == 1 else 2 if c == 2 else 3 if c <= 4 else 4
                tip = f'{d:%a %d %b %Y}: {c} topic{"s" if c != 1 else ""} completed'
            cells.append(f'<i class="cell l{lvl}" title="{tip}"></i>')
    legend = "".join(f'<i class="cell l{i}"></i>' for i in range(5))
    head = (f"{total:g} hours studied" if mode == "hours" else f"{total:g} topics completed") + f" in the last {weeks} weeks"
    return ('<div class="heatpanel">'
            f'<div class="heathead">{head}</div>'
            '<div class="heatscroll"><div class="heatwrap">'
            '<div class="dlabels"><span>Mon</span><span></span><span>Wed</span><span></span><span>Fri</span><span></span><span></span></div>'
            '<div><div class="months">' + "".join(months) + '</div><div class="heat">' + "".join(cells) + '</div></div>'
            '</div></div>'
            f'<div class="legend">Less {legend} More</div></div>')


PALETTE = ["#2b6be6", "#ff9f43", "#0d9488", "#16a34a", "#e11d48", "#0891b2"]


def ring_svg(pct, color, size=130):
    r = 52
    c = 2 * 3.14159265 * r
    off = c * (1 - min(max(pct, 0), 100) / 100)
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 130 130">'
            f'<circle class="rtrack" cx="65" cy="65" r="{r}" fill="none" stroke="#e1e6ef" stroke-width="12"/>'
            f'<circle cx="65" cy="65" r="{r}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 65 65)"/>'
            f'<text x="65" y="74" text-anchor="middle" font-size="26" font-weight="700" fill="{color}">{pct:.0f}%</text></svg>')


def today_ring_svg(pct, size=160):
    r = 52
    c = 2 * 3.14159265 * r
    off = c * (1 - min(max(pct, 0), 100) / 100)
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 130 130">'
            '<defs><linearGradient id="tg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2b6be6"/><stop offset="1" stop-color="#34d399"/></linearGradient></defs>'
            f'<circle class="rtrack" cx="65" cy="65" r="{r}" fill="none" stroke="#e1e6ef" stroke-width="12"/>'
            f'<circle class="ring-anim" cx="65" cy="65" r="{r}" fill="none" stroke="url(#tg)" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{c:.1f}" style="--c:{c:.1f};--off:{off:.1f};stroke-dashoffset:{off:.1f}" transform="rotate(-90 65 65)"/>'
            f'<text class="ring-pct" x="65" y="75" text-anchor="middle">{pct:.0f}%</text></svg>')


def daily_html(p, today):
    th, log = p.get("target_hours"), p.get("log", {})
    counts = done_counts(p)
    h = float(log.get(today.isoformat(), 0) or 0)
    if th:
        pct = min(h / th * 100, 100)
        rem = max(th - h, 0)
        status = ('<div class="ptext rem">🎉 Daily target completed!</div>' if h >= th else
                  f'<div class="ptext rem">{rem:g} hour{"s" if rem != 1 else ""} remaining</div>')
        body = f'<div class="ptext big">{h:g} / {th:g} Hours</div>{status}'
    else:
        pct = 0
        body = '<div class="ptext big">No daily target yet</div><div class="ptext rem">Set a daily hours target above</div>'
    ok = lambda d: is_done(log.get(d.isoformat()), th) or counts.get(d.isoformat(), 0) > 0
    dots = "".join(f'<i class="dot{" on" if ok(d) else ""}{" today" if d == today else ""}"></i>'
                   for d in [today - timedelta(days=6 - i) for i in range(7)])
    best = None
    for sub, ts in p["subjects"].items():
        left = [t for t in ts if not t.get("done")]
        if ts and left:
            sp = (len(ts) - len(left)) / len(ts) * 100
            if best is None or sp > best[0]:
                best = (sp, sub, left[0]["name"])
    if best:
        cont = (f'<h3>Continue Learning</h3><div class="actrow" style="border:0;padding:4px 0"><span>{_html.escape(best[1])} — {_html.escape(best[2])}</span>'
                f'<span>{best[0]:.0f}%</span></div><div class="bar"><span style="width:{best[0]:.0f}%"></span></div>')
    else:
        cont = '<h3>Continue Learning</h3><div class="tmsg">Add subjects and topics below to see your next topic here.</div>'
    return ('<div class="dash">'
            f'<div class="panel"><h3>Today’s Progress</h3><div class="ringcol">{today_ring_svg(pct)}{body}</div></div>'
            f'<div class="panel">{cont}</div>'
            f'<div class="panel" style="text-align:center"><h3>{current_streak(p, today)} day streak</h3><div class="dots">{dots}</div>'
            '<div class="ptext">Last 7 days</div></div></div>')


def subject_rings_html(p):
    out = []
    for i, (sub, ts) in enumerate(p["subjects"].items()):
        pct = sum(1 for t in ts if t.get("done")) / len(ts) * 100 if ts else 0
        out.append(f'<div class="rcell">{ring_svg(pct, PALETTE[i % len(PALETTE)])}<div class="rname">{_html.escape(sub)}</div></div>')
    return '<div class="rings">' + "".join(out) + '</div>' if out else ""


def perf_chart(p, today, days=7):
    xs = [today - timedelta(days=days - 1 - i) for i in range(days)]
    hrs = [float(p.get("log", {}).get(d.isoformat(), 0) or 0) for d in xs]
    mk = sorted((k, float(v)) for k, v in p.get("mocks", {}).items())[-7:]
    if not any(hrs) and not mk:
        return None
    th = p.get("target_hours")
    dark = CHART_TXT == "#F8FAFC"
    BAR, TODAY, MOCK, LINE = (("#4F8FD8", "#60A5FA", "#93C5FD", "#F8FAFC") if dark
                              else ("#7fb0e8", "#1f5fd1", "#a9ccf5", "#14305e"))
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.22})
    for a_ in (ax, axm):
        a_.grid(axis="y", color=CHART_TXT, alpha=.12)
        a_.set_axisbelow(True)
        a_.tick_params(colors=CHART_TXT, length=0)
        for sp in ("top", "right", "left", "bottom"):
            a_.spines[sp].set_visible(False)
    # study hours: one simple bar per day, today highlighted
    top = max(hrs + [th or 0, 1]) * 1.25
    ax.set_ylim(0, top)
    ax.set_xlim(-.6, days + .7)
    ax.bar(range(days), hrs, color=[TODAY if d == today else BAR for d in xs], width=.58)
    for i, h in enumerate(hrs):
        if h > 0:
            ax.text(i, h + top * .02, f"{h:g}", ha="center", va="bottom", fontsize=8, color=CHART_TXT)
    if th:
        ax.axhline(th, ls="--", lw=1.4, color=LINE, alpha=.7)
        ax.text(days - .35, th, f"Goal\n{th:g} h", ha="left", va="center", fontsize=8, color=LINE)
    ax.set_xticks(range(days))
    ax.set_xticklabels([f"{d:%a}\n{d:%d %b}" for d in xs], fontsize=8)
    ax.set_ylabel("Hours studied", color=CHART_TXT, fontsize=9)
    ax.set_title(f"Study hours (last {days} days)", loc="left", fontsize=10, fontweight="bold", color=CHART_TXT)
    # mock tests: one simple bar per test
    axm.set_ylim(0, 118)
    axm.set_yticks([0, 25, 50, 75, 100])
    axm.set_title("Mock test marks (%)", loc="left", fontsize=10, fontweight="bold", color=CHART_TXT)
    if mk:
        idx = list(range(len(mk)))
        axm.bar(idx, [v for _, v in mk], color=MOCK, width=.5)
        for i, (_, v) in zip(idx, mk):
            axm.text(i, v + 2, f"{v:g}%", ha="center", va="bottom", fontsize=8, color=CHART_TXT)
        axm.set_xticks(idx)
        axm.set_xticklabels([f"{date.fromisoformat(k):%d %b}" for k, _ in mk], fontsize=8)
        axm.set_xlim(-.7, max(len(mk) - .3, .7))
    else:
        axm.set_xticks([])
        axm.text(.5, .5, "No mock tests yet", transform=axm.transAxes, ha="center", va="center",
                 color=CHART_TXT, alpha=.6, fontsize=9)
    return fig_to_img(fig)


def recent_html(p, n=8):
    ev = [(t["done"], f"{sub} — Completed {t['name']}") for sub, ts in p["subjects"].items() for t in ts if t.get("done")]
    ev += [(d, f"Studied {float(h):g} hours") for d, h in p.get("log", {}).items()]
    ev += [(d, f"Mock test — scored {float(m):g}%") for d, m in p.get("mocks", {}).items()]
    ev.sort(key=lambda e: e[0], reverse=True)
    rows = "".join(f'<div class="actrow"><span>{_html.escape(txt)}</span><span class="actdate">{date.fromisoformat(d):%d %b %Y}</span></div>'
                   for d, txt in ev[:n]) or '<div class="tmsg">Nothing yet. Log your study, tick a topic or save a mock test.</div>'
    return '<div class="panel"><h3>🕘 Recent activity</h3>' + rows + '</div>'


# ---------- page ----------
left_img = load_image(("images/study-desk.jpg", "images/study-desk.jpeg", "images/study-desk.png"))
right_img = load_image(("images/student-laptop.jpg", "images/student-laptop.jpeg", "images/student-laptop.png"))

st.markdown(CSS, unsafe_allow_html=True)
DARK_MODE = bool(st.session_state.get("dark_mode", False))
CHART_TXT = "#F8FAFC" if DARK_MODE else "#14305e"
if DARK_MODE:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown(HEADER.replace("__T1__", TITLE_LINE1).replace("__T2__", TITLE_LINE2), unsafe_allow_html=True)
live_clock()
_, _tg, _ = st.columns([2, 1, 2])
with _tg:
    st.toggle("🌙 Dark mode", key="dark_mode")
st.markdown(GALLERY.replace("__LEFT__", left_img or LEFT_ART).replace("__RIGHT__", right_img or RIGHT_ART),
            unsafe_allow_html=True)

_prog = load_progress()
_today = today_date()

# ---------- study target ----------
st.markdown('<div class="sec-title" style="color:#0369a1">🎯 Set Target &amp; Log Today’s Study</div>', unsafe_allow_html=True)
with st.container(key="card_target"):
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(key="tile_hours"):
            st.markdown('<div class="card-title" style="font-size:17px">⏱ Daily study-hours target</div>', unsafe_allow_html=True)
            st.number_input("Target hours", min_value=0.5, max_value=24.0, step=0.5, value=_prog.get("target_hours"),
                            placeholder="Example: 6", label_visibility="collapsed", key="t_hours")
            st.button("Set Target", icon=":material/flag:", on_click=save_target, key="btn_target")
            show_msg("tg_msg")
    with c2:
        with st.container(key="tile_today"):
            st.markdown(f'<div class="card-title" style="font-size:17px">📝 Hours studied today ({_today:%d %b})</div>', unsafe_allow_html=True)
            st.number_input("Hours today", min_value=0.0, max_value=24.0, step=0.5,
                            value=_prog.get("log", {}).get(_today.isoformat()),
                            placeholder="Example: 5.5", label_visibility="collapsed", key="log_hours")
            st.button("Mark Today Completed", icon=":material/check_circle:", on_click=log_today, key="btn_log")
            show_msg("lg_msg")
    with c3:
        with st.container(key="tile_mock"):
            st.markdown('<div class="card-title" style="font-size:17px">🧪 Mock test marks</div>', unsafe_allow_html=True)
            st.number_input("Mock test marks", min_value=0.0, max_value=100.0, step=0.5,
                            value=_prog.get("mocks", {}).get(_today.isoformat()),
                            placeholder="Example: 72 (out of 100)", label_visibility="collapsed", key="mock_marks")
            st.button("Save Mock Marks", icon=":material/quiz:", on_click=save_mock, key="btn_mock")
            show_msg("mk_msg")
            if _prog.get("mocks"):
                _k = max(_prog["mocks"])
                st.markdown(f'<div class="ptext">Latest: {float(_prog["mocks"][_k]):g}% on {date.fromisoformat(_k):%d %b}</div>',
                            unsafe_allow_html=True)


st.markdown('<div class="sec-title">📊 Daily Progress &amp; Study Streak</div>', unsafe_allow_html=True)
st.markdown(daily_html(_prog, _today), unsafe_allow_html=True)

st.markdown('<div class="sec-title" style="color:#0369a1">📚 Subjects &amp; Topics</div>', unsafe_allow_html=True)
with st.container(key="card_target_topics"):
    st.markdown(subject_rings_html(_prog), unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="font-size:19px">➕ Add a subject and topic</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns([3, 3, 1.4], gap="medium", vertical_alignment="bottom")
    with a1:
        st.markdown('<div class="lbl">Subject name</div>', unsafe_allow_html=True)
        st.text_input("Subject name", placeholder="Example: Mathematics", label_visibility="collapsed", key="ts_sub")
    with a2:
        st.markdown('<div class="lbl">Topic to complete</div>', unsafe_allow_html=True)
        st.text_input("Topic", placeholder="Example: Integration", label_visibility="collapsed", key="ts_topic")
    with a3:
        st.button("Add Topic", icon=":material/add:", on_click=add_topic, key="btn_topic")
    show_msg("t_msg")

    if _prog["subjects"]:
        st.markdown('<div class="lbl" style="margin-top:14px">Tick the box when you finish a topic ✅</div>', unsafe_allow_html=True)
        _cols = st.columns(2, gap="medium")
        for i, (sub, topics) in enumerate(_prog["subjects"].items()):
            with _cols[i % 2]:
                with st.container(key=f"tile_sub_{i}"):
                    n_done = sum(1 for t in topics if t.get("done"))
                    pct = n_done / len(topics) * 100
                    st.markdown(f'<div class="subj-row">{ring_svg(pct, PALETTE[i % len(PALETTE)], 72)}<div>'
                                f'<div class="subj-name">{_html.escape(sub)}</div>'
                                f'<div class="subj-count">{n_done}/{len(topics)} topics completed</div></div></div>', unsafe_allow_html=True)
                    for t in topics:
                        r1, r2 = st.columns([8, 1], vertical_alignment="center")
                        with r1:
                            if t.get("done"):
                                label = f"~~{t['name']}~~ · done {date.fromisoformat(t['done']):%d %b}"
                            else:
                                label = t["name"]
                            st.checkbox(label, value=bool(t.get("done")), key=f"chk_{t['id']}",
                                        on_change=toggle_topic, args=(t["id"],))
                        with r2:
                            st.button(":material/delete:", key=f"rm_tp_{t['id']}", on_click=remove_topic, args=(t["id"],))


# ---------- exam details ----------

st.markdown('<div class="sec-title">📅 Exam Details</div>', unsafe_allow_html=True)
with st.container(key="card_exam"):
    ea, eb = st.columns([1, 1.2], gap="large")
    with ea:
        st.markdown('<div class="card-title" style="font-size:19px">➕ Add an exam</div>', unsafe_allow_html=True)
        st.markdown('<div class="lbl">Exam subject</div>', unsafe_allow_html=True)
        st.text_input("Exam subject", placeholder="Example: Mathematics", label_visibility="collapsed", key="ex_sub")
        st.markdown('<div class="lbl">Exam date</div>', unsafe_allow_html=True)
        st.date_input("Exam date", value=None, min_value=_today, format="DD/MM/YYYY",
                      label_visibility="collapsed", key="ex_date")
        st.button("Save Exam", icon=":material/event:", on_click=add_exam, key="btn_exam")
        show_msg("e_msg")
    with eb:
        st.markdown('<div class="card-title" style="font-size:19px">⏳ Upcoming exams</div>', unsafe_allow_html=True)
        _exams = sorted(_prog["exams"], key=lambda e: (e.get("date", "") < _today.isoformat(), e.get("date", "")))
        if not _exams:
            st.markdown('<div class="tmsg" style="text-align:center">No exams added yet.</div>', unsafe_allow_html=True)
        for e in _exams:
            r1, r2 = st.columns([6, 1], vertical_alignment="center")
            with r1:
                st.markdown(exam_row_html(e, _today), unsafe_allow_html=True)
            with r2:
                st.button(":material/delete:", key=f"rm_ex_{e['id']}", on_click=remove_exam, args=(e["id"],))


st.markdown(f'<div class="sec-title">{PREDICT_TITLE}</div>', unsafe_allow_html=True)
_, middle, _ = st.columns([1, 1.3, 1])
with middle:
    with st.container(key="card"):
        st.markdown(CARD_TOP, unsafe_allow_html=True)
        st.markdown('<div class="lbl">Study hours per day (0 - 24)</div>', unsafe_allow_html=True)
        hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=None, step=0.5,
                                placeholder="Enter study hours (0 - 24)", label_visibility="collapsed")
        st.markdown('<div class="lbl">Level of understanding (1 = low, 10 = excellent)</div>', unsafe_allow_html=True)
        understand = st.number_input("Understanding level", min_value=1, max_value=10, value=None, step=1,
                                     placeholder="Enter understanding level (1 - 10)", label_visibility="collapsed")
        st.markdown('<div class="lbl">Health level (1 = poor, 10 = excellent)</div>', unsafe_allow_html=True)
        health = st.number_input("Health level", min_value=1, max_value=10, value=None, step=1,
                                 placeholder="Enter health level (1 - 10)", label_visibility="collapsed")

        if st.button("Predict Marks", icon=":material/show_chart:"):
            if hours is None or understand is None or health is None:
                st.session_state["err"], st.session_state["res"] = True, None
            else:
                st.session_state["err"], st.session_state["res"] = False, (hours, health, understand)

        if st.session_state.get("err"):
            st.markdown('<div class="err"><b>!</b>Fill all three boxes: hours 0 to 24, understanding 1 to 10, health 1 to 10</div>',
                        unsafe_allow_html=True)

        res = st.session_state.get("res")
        if res:
            h, hl, u = res
            marks = predict(h, hl, u)
            st.markdown(f'<div class="result">✦ Predicted Marks: {marks:.1f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="tags">Study {h:g} h &nbsp;•&nbsp; Understanding: {level(u, ("Low", "Medium", "High"))} ({u}/10)'
                        f' &nbsp;•&nbsp; Health: {level(hl, ("Poor", "Fair", "Good"))} ({hl}/10)</div>', unsafe_allow_html=True)
            ideas = [("+1 study hour", predict(min(h + 1, 24), hl, u) - marks),
                     ("Understanding to 10", predict(h, hl, 10) - marks),
                     ("Health to 10", predict(h, 10, u) - marks)]
            chips = "".join(f'<span class="chip">{n}: +{g:.1f} marks</span>' for n, g in ideas if g > 0.05)
            if chips:
                st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
            if h > 10:
                st.markdown('<div class="hint">The model learned from 1–9 hours of data, so higher values are only rough estimates.</div>',
                            unsafe_allow_html=True)

QUOTE_SECONDS = 8   # a new quote appears automatically every 8 seconds


@st.fragment(run_every=QUOTE_SECONDS)
def rotating_quote():
    res_now = st.session_state.get("res")
    if res_now:
        m = predict(res_now[0], res_now[1], res_now[2])
        tier = "low" if m < 50 else "mid" if m < 75 else "high"
    else:
        tier = "general"
    pool = QUOTES[tier]
    quote = pool[int(time.time() // QUOTE_SECONDS) % len(pool)]
    st.markdown(f'<div class="qbox"><div class="qtitle">💪 Daily motivation</div><div class="qtext">“{quote}”</div></div>',
                unsafe_allow_html=True)


_, qmid, _ = st.columns([1, 1.3, 1])
with qmid:
    rotating_quote()

res = st.session_state.get("res")
if res:
    h, hl, u = res
    st.markdown(
        '<div class="charts">'
        '<div class="panel"><h3>📈 What-if simulation</h3>' + sim_chart(h, hl, u, predict(h, hl, u)) + '</div>'
        '<div class="panel"><h3>🧩 What drives your marks</h3>' + contrib_chart(h, hl, u) + '</div>'
        '</div>', unsafe_allow_html=True)



# ---------- performance analytics and study history ----------
_prog = load_progress()
st.markdown('<div class="sec-title">📈 Performance Analytics &amp; Study History</div>', unsafe_allow_html=True)
_, rmid, _ = st.columns([1, 2, 1])
with rmid:
    _mode = st.radio("Calendar shows", ["Topics completed", "Study hours"], horizontal=True, key="heat_mode")
_chart = perf_chart(_prog, _today)
_perf = '<div class="panel wide"><h3>📈 Study hours &amp; mock test performance</h3>' + (
    _chart or '<div class="tmsg">Log your study hours and save mock test marks to see this graph.</div>') + '</div>'
st.markdown('<div class="charts">' + _perf + heatmap_html(_prog, _today, "hours" if _mode == "Study hours" else "topics")
            + progress_html(_prog, st.session_state.get("res"), _today) + recent_html(_prog) + '</div>', unsafe_allow_html=True)

st.markdown('<div class="foot">🎓 Study Smart | Work Hard | Achieve More</div>', unsafe_allow_html=True)