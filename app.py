# app.py — Text Insight Lab (mejorado)
import re
import json
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator, single_detection

# ───────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Insight Lab",
    page_icon="💬",
    layout="wide",
    menu_items={"About": "Text Insight Lab – versión con traducción robusta + historial exportable"},
)

PRIMARY = "#4F46E5"
ACCENT  = "#06B6D4"
OK      = "#22C55E"
WARN    = "#F59E0B"
BAD     = "#EF4444"
MUTED   = "#94A3B8"

st.markdown(
    f"""
    <style>
      .card {{
        background: linear-gradient(180deg, rgba(255,255,255,.75), rgba(255,255,255,.55));
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(17,24,39,.08);
      }}
      @media (prefers-color-scheme: dark) {{
        .card {{ background: linear-gradient(180deg, rgba(15,23,42,.7), rgba(15,23,42,.55)); }}
      }}
      .badge {{
        display:inline-flex; align-items:center; gap:.5rem; font-weight:600;
        font-size:.85rem; padding:.3rem .65rem; border-radius:999px;
        border:1px solid rgba(148,163,184,.3); background: rgba(148,163,184,.12);
        color: #e2e8f0;
      }}
      textarea, .stTextInput > div > div > input {{ border-radius: 14px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────
# ESTADO & CACHES
# ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []

@st.cache_resource(show_spinner=False)
def get_translator():
    # No necesita endpoints manuales, deep-translator usa la API pública de Google
    return GoogleTranslator(source='auto', target='en')

translator = get_translator()

@st.cache_data(show_spinner=False)
def cached_detect(text: str) -> str:
    if not text.strip():
        return "und"
    try:
        return single_detection(text, api_key=None) or "und"
    except Exception:
        if re.search(r"[áéíóúñ¿¡]", text.lower()):
            return "es"
        return "en"

@st.cache_data(show_spinner=False)
def cached_translate(text: str, src: Optional[str], dest: str) -> str:
    if not text.strip():
        return text
    try:
        return GoogleTranslator(source=src or 'auto', target=dest).translate(text)
    except Exception:
        return text

# ───────────────────────────────────────────────────────────────
# UTILIDADES DE NLP
# ───────────────────────────────────────────────────────────────
STOP_ES = set("""
a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaban estado estaba estais estamos estan estar estara estas este estes esto estos estoy fue fuera fueron fui fuimos han hasta hay la las le les lo los mas me mi mis mucha muchos muy nada ni no nos nosotras nosotros o os otra otras otro otros para pero poco por porque que quien quienes se ser sera si sido sin sobre sois somos son soy su sus tambien te tiene tengo tenia tenian tenemos teneis tener tus tu un una uno unas unos y ya
""".split())
STOP_EN = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let let's me more most mustn't my myself no nor not of off on once only or other ought our ours  ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're  they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

def normalize_spaces(t: str) -> str:
    t = re.sub(r"https?://\S+", " ", t)  # quita URLs
    t = re.sub(r"[@#]\w+", " ", t)       # quita @menciones/#hashtags
    t = re.sub(r"[^\w\sáéíóúñÁÉÍÓÚüÜ¿¡’'-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def sentiment_blob(en_text: str) -> Tuple[float, float]:
    blob = TextBlob(en_text)
    return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)

def split_sentences(t: str) -> List[str]:
    # separador simple multi-puntuación
    return [s.strip() for s in re.split(r"[\.!?…]+", t) if s.strip()]

def count_words(text: str, lang: str) -> Dict[str, int]:
    tokens = re.findall(r"\b[\w’'-]+\b", text.lower())
    stop = STOP_EN if lang == "en" else STOP_ES
    tokens = [t for t in tokens if len(t) > 2 and not t.isnumeric() and t not in stop]
    return dict(Counter(tokens).most_common(15))

def label_from_polarity(p: float) -> Tuple[str, str, str]:
    if p >= 0.5:  return "Positivo", "😊", OK
    if p <= -0.5: return "Negativo", "😔", BAD
    return "Neutral", "😐", WARN

def add_history(payload: Dict):
    st.session_state.history.insert(0, payload)

# ───────────────────────────────────────────────────────────────
# SIDEBAR — opciones
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Opciones")
    lang_mode = st.selectbox(
        "Idioma para el análisis",
        ["Auto (detectar & traducir a EN)", "Forzar EN (sin traducir)", "Forzar ES → traducir a EN"],
        index=0,
        help="TextBlob rinde mejor en inglés; por eso traducimos por defecto.",
    )
    show_topwords = st.toggle("Mostrar top-palabras", value=True)
    per_sentence = st.toggle("Analizar por frases", value=True)
    st.divider()
    uploaded = st.file_uploader("Sube un .txt (opcional)", type=["txt"])
    if st.button("Usar texto de ejemplo"):
        st.session_state["example_text"] = (
            "Hoy me sentí increíblemente motivado. El cielo estaba claro y todo salió bien. "
            "Aunque tuve un pequeño problema con el bus, resolví la situación con calma."
        )
    clear_hist = st.button("🧹 Limpiar historial")
    if clear_hist:
        st.session_state.history.clear()
        st.toast("Historial limpiado", icon="🧹")

# ───────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ───────────────────────────────────────────────────────────────
st.title("💬 Text Insight Lab")
st.caption("Analiza sentimiento, subjetividad y palabras clave con una interfaz moderna.")

tab_an, tab_hist = st.tabs(["🔎 Analizar texto", "🗂️ Historial"])

# ───────────────────────────────────────────────────────────────
# TAB 1 — ANÁLISIS
# ───────────────────────────────────────────────────────────────
with tab_an:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    default_text = st.session_state.get("example_text", "")
    if uploaded is not None:
        try:
            default_text = uploaded.read().decode("utf-8", errors="ignore")
        except Exception:
            st.warning("No se pudo leer el archivo como UTF-8; usando fallback ISO-8859-1…")
            default_text = uploaded.read().decode("latin-1", errors="ignore")

    texto = st.text_area("✍️ Escribe tu texto (ES/EN) o carga un .txt:", height=180, value=default_text)

    if st.button("Analizar texto", type="primary"):
        if texto.strip():
            with st.spinner("Preprocesando, traduciendo (si aplica) y analizando…"):
                texto_norm = normalize_spaces(texto)

                # 1) Detección & pipeline de idioma
                src_detected = cached_detect(texto_norm) if lang_mode.startswith("Auto") else ("es" if "ES" in lang_mode else "en")

                # 2) Texto que analiza TextBlob (inglés si no forzamos EN)
                if lang_mode == "Forzar EN (sin traducir)":
                    text_for_blob = texto_norm
                    used_lang_for_blob = "en"
                    texto_en = texto_norm
                    texto_es = texto_norm if src_detected == "es" else cached_translate(texto_norm, src_detected, "es")
                elif lang_mode == "Forzar ES → traducir a EN":
                    texto_es = texto_norm
                    texto_en = cached_translate(texto_es, "es", "en")
                    text_for_blob = texto_en
                    used_lang_for_blob = "en"
                else:
                    # Auto: si no es EN, traducimos a EN; guardamos ambas vistas
                    if src_detected != "en":
                        texto_en = cached_translate(texto_norm, src_detected, "en")
                        texto_es = texto_norm if src_detected == "es" else cached_translate(texto_norm, src_detected, "es")
                        used_lang_for_blob = "en"
                        text_for_blob = texto_en
                    else:
                        texto_en = texto_norm
                        texto_es = cached_translate(texto_norm, "en", "es")
                        used_lang_for_blob = "en"
                        text_for_blob = texto_en

                # 3) Sentimiento global
                pol, sub = sentiment_blob(text_for_blob)
                label, emoji, color = label_from_polarity(pol)

                st.markdown(
                    f'<span class="badge" style="background:{color}; color:white;">{emoji} {label}</span>',
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Polaridad", f"{pol:+.2f}")
                c2.metric("Subjetividad", f"{sub:.2f}")
                c3.metric("Idioma detectado", src_detected.upper())

                st.write("**Polaridad (−1 → 1)**")
                st.progress((pol + 1) / 2)
                st.write("**Subjetividad (0 → 1)**")
                st.progress(sub)

                # Mostrar textos base del análisis
                with st.expander("Ver textos"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original (ES)**")
                        st.text(texto_es)
                    with col2:
                        st.markdown("**Usado para análisis (EN)**")
                        st.text(texto_en)

                # 4) Frases detectadas
                if per_sentence:
                    st.subheader("Frases detectadas")
                    frases_es = split_sentences(texto_es)
                    frases_en = split_sentences(texto_en)
                    filas = []
                    for i in range(min(len(frases_es), len(frases_en))):
                        p_i, _ = sentiment_blob(frases_en[i])
                        lbl, emo, _c = label_from_polarity(p_i)
                        filas.append({
                            "#": i + 1,
                            "Original (ES)": frases_es[i],
                            "Traducción (EN)": frases_en[i],
                            "Polaridad": round(p_i, 2),
                            "Etiqueta": lbl,
                            "Icono": emo
                        })
                    if filas:
                        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
                    else:
                        st.info("No se detectaron frases.")

                # 5) Top palabras (según idioma original detectado)
                if show_topwords:
                    st.subheader("Top-palabras")
                    lang_for_top = "en" if src_detected == "en" else "es"
                    base_text = texto_en if lang_for_top == "en" else texto_es
                    top_dict = count_words(base_text, lang_for_top)
                    if top_dict:
                        df_top = pd.DataFrame(
                            [{"Palabra": k, "Frecuencia": v} for k, v in top_dict.items()]
                        )
                        st.dataframe(df_top, use_container_width=True, hide_index=True)
                    else:
                        st.info("No se encontraron palabras destacadas.")

                # 6) Guardar historial
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "src_detected": src_detected,
                    "texto_es": texto_es,
                    "texto_en": texto_en,
                    "pol": round(pol, 3),
                    "sub": round(sub, 3),
                    "label": label,
                    "emoji": emoji
                }
                add_history(record)
                st.toast("Análisis guardado ✅", icon="💾")

        else:
            st.warning("Por favor, ingresa algún texto para analizar.")

    st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# TAB 2 — HISTORIAL
# ───────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Historial de análisis previos")

    if not st.session_state.history:
        st.info("No hay análisis previos aún.")
    else:
        # Exportar CSV
        df_hist = pd.DataFrame(st.session_state.history)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.download_button(
                "⬇️ Descargar historial (CSV)",
                data=df_hist.to_csv(index=False).encode("utf-8"),
                file_name="text_insight_historial.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c2:
            st.caption("Incluye timestamp, idioma detectado, polaridad, subjetividad y textos ES/EN.")

        st.divider()
        for h in st.session_state.history:
            st.markdown(f"**{h['timestamp']}** — {h['emoji']} *{h['label']}* · Pol: {h['pol']:+.2f} · Subj: {h['sub']:.2f}")
            with st.expander("Ver texto analizado"):
                st.text(h["texto_es"])
    st.markdown('</div>', unsafe_allow_html=True)
