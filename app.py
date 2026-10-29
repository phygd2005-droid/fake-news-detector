import hmac
import io
import os
import re

import joblib
import numpy as np
import requests
import streamlit as st
from PIL import Image

from common import clean
from facts import check_facts
from verifier import verify_text

st.set_page_config(page_title="Fake News & Poster Detector", page_icon="🕵️")


def secret(name, default=""):
    """Read a value from Streamlit secrets (cloud) or environment variables (local)."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def login_gate():
    """Asks for a username and password when APP_USERNAME / APP_PASSWORD are set in secrets."""
    user_ok = str(secret("APP_USERNAME", ""))
    pass_ok = str(secret("APP_PASSWORD", ""))
    if not user_ok or not pass_ok:
        return                                  # no login configured (local testing)
    if st.session_state.get("logged_in"):
        with st.sidebar:
            if st.button("Log out"):
                st.session_state["logged_in"] = False
                st.rerun()
        return
    st.title("🔒 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if (hmac.compare_digest(u.encode(), user_ok.encode())
                and hmac.compare_digest(p.encode(), pass_ok.encode())):
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Wrong username or password")
    st.stop()


# ------------------------------------------------------------------ models
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_resource
def load_easyocr():
    import easyocr
    return easyocr.Reader(["en"])          # use ["en", "hi"] to also read Hindi text


def ml_style_check(text):
    """Writing-style classifier trained in train.py. Returns (verdict, confidence%)."""
    try:
        proba = load_model().predict_proba([clean(text)])[0]     # [fake, true]
    except Exception:
        return None, 0.0
    verdict = "TRUE" if proba[1] >= 0.5 else "FAKE"
    return verdict, max(proba) * 100


# ------------------------------------------------------------------ OCR
def ocr_space(img):
    """Free OCR API (used on the cloud, where EasyOCR is too heavy)."""
    key = secret("OCR_SPACE_KEY", "helloworld")     # get your own free key at ocr.space
    im = img.copy()
    im.thumbnail((1400, 1400))
    quality = 85
    while True:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        if buf.tell() < 1_000_000 or quality <= 40:    # free plan limit is 1 MB
            break
        quality -= 15
    try:
        r = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": ("poster.jpg", buf.getvalue(), "image/jpeg")},
            data={"apikey": key, "language": "eng", "OCREngine": "2", "scale": "true"},
            timeout=40).json()
        if r.get("IsErroredOnProcessing"):
            return "", str(r.get("ErrorMessage"))
        text = " ".join(p.get("ParsedText", "") for p in r.get("ParsedResults", []))
        return text, ""
    except Exception as e:
        return "", str(e)[:120]


def read_text_from_image(img):
    try:
        import easyocr  # noqa: F401  (installed only in the full local version)
        have_easyocr = True
    except ImportError:
        have_easyocr = False
    if have_easyocr:
        return " ".join(load_easyocr().readtext(np.array(img), detail=0)), ""
    return ocr_space(img)


# ------------------------------------------------------------------ ESP32 display
def publish_to_display(topic, verdict, text):
    """Sends 'VERDICT|short text' to ntfy.sh; the ESP32 reads it and shows it on the OLED."""
    if not topic:
        return False
    short = text.encode("ascii", "ignore").decode()
    short = re.sub(r"\s+", " ", short).strip()[:90]
    try:
        r = requests.post(f"https://ntfy.sh/{topic.strip()}",
                          data=f"{verdict}|{short}".encode("utf-8"), timeout=6)
        return r.status_code == 200
    except Exception:
        return False


# ------------------------------------------------------------------ logic
def analyse(text, use_web, google_key):
    # Layer 1: hand-verified facts
    v, why = check_facts(text)
    if v:
        return {"verdict": v, "method": "Verified-facts database", "reason": why,
                "evidence": [], "confidence": None}

    # Layer 2 + 3: Google Fact Check API + live web evidence
    if use_web:
        res = verify_text(text, google_key)
        res["method"] = "Online evidence check (web search + fact-check sites)"
        if res["verdict"] != "UNCERTAIN":
            return res
    else:
        res = {"verdict": "UNCERTAIN", "reason": "Online check is switched off.",
               "evidence": [], "confidence": None, "method": "-"}

    # Layer 4: last resort - writing-style classifier
    mv, mc = ml_style_check(text)
    if mv and mc >= 80:
        return {"verdict": mv, "method": "ML writing-style classifier (no online evidence found)",
                "reason": f"Confidence {mc:.1f}%. This only judges writing style, not facts.",
                "evidence": res.get("evidence", []), "confidence": None,
                "debug": res.get("debug")}
    return res


def show(res):
    v = res["verdict"]
    if v == "FAKE":
        st.error("❌ FAKE")
    elif v == "TRUE":
        st.success("✅ TRUE")
    else:
        st.warning("⚠️ UNCERTAIN - could not verify")
    st.write(f"**Decided by:** {res.get('method', '-')}")
    st.write(res.get("reason", ""))
    if res.get("confidence"):
        st.write(f"**Evidence agreement:** {res['confidence'] * 100:.0f}%")
    ev = res.get("evidence") or []
    if ev:
        with st.expander(f"Evidence found ({len(ev)})", expanded=True):
            for e in ev:
                st.markdown(f"- **{e['stance']}** - [{e['title'][:110]}]({e['url']})")
    if res.get("debug"):
        with st.expander("Technical details (for debugging)"):
            st.json(res["debug"])


def run_check(text, use_web, google_key, topic):
    with st.spinner("Searching the web and checking the claim..."):
        res = analyse(text, use_web, google_key)
    show(res)
    if topic:
        ok = publish_to_display(topic, res["verdict"], text)
        st.caption("📟 Result sent to the ESP32 OLED display." if ok
                   else "📟 Could not send the result to the display (check topic / internet).")


# ------------------------------------------------------------------ UI
login_gate()

st.title("🕵️ Fake News & Fake Poster Detector")

with st.sidebar:
    st.header("Settings")
    use_web = st.checkbox("Verify online (web search)", value=True)
    google_key = secret("GOOGLE_FACTCHECK_KEY", "")
    topic = secret("NTFY_TOPIC", "")
    if not google_key:
        google_key = st.text_input("Google Fact Check API key (optional)", type="password")
    if not topic:
        topic = st.text_input("ESP32 display topic (ntfy.sh)")
    st.caption("Needs internet. The topic must be the same one written in the ESP32 code.")

tab1, tab2 = st.tabs(["📰 News text", "🖼️ Image / Poster"])

with tab1:
    news = st.text_area("Paste any news headline, claim or article", height=200)
    if st.button("Check news") and news.strip():
        run_check(news, use_web, google_key, topic)

with tab2:
    up = st.file_uploader("Upload image / poster", type=["jpg", "jpeg", "png", "webp"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, width=500)
        if st.button("Check image"):
            with st.spinner("Reading text from the image..."):
                text, err = read_text_from_image(img)
            if not text.strip():
                st.warning("No readable text found in the image." + (f" ({err})" if err else ""))
            else:
                st.write("**Text found in image:**", text)
                run_check(text, use_web, google_key, topic)