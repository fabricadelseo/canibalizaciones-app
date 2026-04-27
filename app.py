# -*- coding: utf-8 -*-
"""
Detector de Canibalizaciones SEO — La Fábrica del SEO  (v2)
===========================================================
Conecta con Ahrefs API v3 (endpoint `organic-keywords`) y detecta
canibalizaciones REALES: keywords donde 2+ URLs del proyecto rankean
simultáneamente en posiciones relevantes (no solo "Top keyword duplicada").

Para cada grupo, calcula:
  - Severidad (Alta / Media / Baja) por reglas SEO.
  - Score de impacto (volumen × CTR esperado × nº URLs) para priorizar.
  - Recomendación de Claude con contexto (volumen, KD, intención, tráfico).

Dos entradas:
  - API Ahrefs (recomendado).
  - CSV de Top Pages (fallback, formato del Colab original).
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import date
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from anthropic import Anthropic, APIError

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Canibalizaciones SEO · La Fábrica del SEO",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
AHREFS_BASE_URL = "https://api.ahrefs.com/v3"

COUNTRIES = {
    "es": "España", "us": "Estados Unidos", "mx": "México", "ar": "Argentina",
    "co": "Colombia", "cl": "Chile", "pe": "Perú", "fr": "Francia",
    "it": "Italia", "pt": "Portugal", "de": "Alemania", "uk": "Reino Unido",
}

MODES = {
    "subdomains": "Dominio + subdominios",
    "domain": "Solo dominio principal",
    "prefix": "Prefijo (sección)",
    "exact": "URL exacta",
}

# CTR aproximado por posición (Sistrix / Advanced Web Ranking 2024).
# Usado para estimar tráfico potencial perdido por canibalización.
CTR_BY_POSITION = {
    1: 0.30, 2: 0.16, 3: 0.10, 4: 0.07, 5: 0.05,
    6: 0.04, 7: 0.03, 8: 0.025, 9: 0.02, 10: 0.018,
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def clean_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    o = urlparse(url.strip())
    if not o.scheme or not o.netloc:
        return url.strip()
    return f"{o.scheme}://{o.netloc}{o.path}"


def estimated_ctr(position: float) -> float:
    """CTR estimado para una posición (0 si >10)."""
    try:
        p = int(round(float(position)))
    except (TypeError, ValueError):
        return 0.0
    return CTR_BY_POSITION.get(p, 0.0)


def detect_intent(row: pd.Series) -> str:
    """Devuelve la intención dominante a partir de los flags de Ahrefs."""
    flags = [
        ("transaccional", row.get("is_transactional")),
        ("comercial", row.get("is_commercial")),
        ("informacional", row.get("is_informational")),
        ("navegacional", row.get("is_navigational")),
        ("branded", row.get("is_branded")),
    ]
    for name, val in flags:
        if val is True or val == 1:
            return name
    return "desconocida"


def detect_page_type(url: str) -> str:
    """Heurística simple por URL: blog / categoría / producto / servicio / home."""
    u = (url or "").lower()
    if u.endswith("/") and u.count("/") <= 3:
        return "home"
    for marker, label in [
        ("/blog/", "blog"),
        ("/noticias/", "blog"),
        ("/recursos/", "blog"),
        ("/categoria/", "categoría"),
        ("/category/", "categoría"),
        ("/producto/", "producto"),
        ("/product/", "producto"),
        ("/tienda/", "producto"),
        ("/servicios/", "servicio"),
        ("/servicio/", "servicio"),
        ("/contacto", "contacto"),
        ("/sobre-", "página fija"),
        ("/about", "página fija"),
    ]:
        if marker in u:
            return label
    return "página"


# ---------------------------------------------------------------------------
# Cliente Ahrefs API v3 — organic-keywords
# ---------------------------------------------------------------------------

ORGANIC_KW_FIELDS = (
    "keyword,best_position,best_position_url,volume,keyword_difficulty,"
    "cpc,is_branded,is_commercial,is_informational,"
    "is_navigational,is_transactional"
)


def fetch_organic_keywords(
    api_key: str,
    target: str,
    country: str,
    mode: str = "subdomains",
    limit: int = 5000,
    on_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Llama a /site-explorer/organic-keywords y devuelve un DataFrame con todas
    las parejas (keyword × URL) que rankean para el target.
    Docs: https://docs.ahrefs.com/api/reference/site-explorer/get-organic-keywords
    """
    if not api_key:
        raise ValueError("Falta la API key de Ahrefs.")
    if not target:
        raise ValueError("Falta el dominio objetivo.")

    on_date = on_date or date.today().isoformat()

    params = {
        "target": target,
        "country": country,
        "mode": mode,
        "limit": limit,
        "date": on_date,
        "select": ORGANIC_KW_FIELDS,
        "order_by": "traffic:desc",
        "output": "json",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    resp = requests.get(
        f"{AHREFS_BASE_URL}/site-explorer/organic-keywords",
        params=params,
        headers=headers,
        timeout=90,
    )

    if resp.status_code == 401:
        raise PermissionError("API key de Ahrefs inválida o sin permisos (401).")
    if resp.status_code == 403:
        raise PermissionError("Acceso denegado por Ahrefs (403). Revisa el plan o los permisos.")
    if resp.status_code == 429:
        raise RuntimeError("Ahrefs ha devuelto rate limit (429). Reintenta en un minuto.")
    if not resp.ok:
        raise RuntimeError(f"Error Ahrefs {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    rows = payload.get("keywords") or payload.get("data") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normaliza nombre de la URL (la API usa best_position_url).
    if "best_position_url" in df.columns:
        df = df.rename(columns={"best_position_url": "url"})
    elif "url" not in df.columns:
        raise RuntimeError("La respuesta de Ahrefs no contiene URL de ranking.")

    return df


def read_top_pages_csv(file_bytes: bytes) -> pd.DataFrame:
    """Fallback: lee CSV de Top Pages con formato del Colab original."""
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    # Mapeo de columnas Top Pages → formato organic-keywords interno.
    rename_map = {
        "URL": "url",
        "Top keyword": "keyword",
        "Top keyword: Position": "best_position",
        "Volume": "volume",
        "Current traffic": "traffic",
        "Top keyword KD": "keyword_difficulty",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    return df


# ---------------------------------------------------------------------------
# Detección + scoring
# ---------------------------------------------------------------------------


def detect_canibalizations(
    df: pd.DataFrame,
    max_position: int = 20,
    min_volume: int = 10,
) -> pd.DataFrame:
    """
    Filtra parejas (keyword, URL) que cumplen umbrales y agrupa por keyword
    canibalizada (>=2 URLs distintas).

    Espera columnas: url, keyword, best_position, volume.
    """
    required = {"url", "keyword", "best_position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {', '.join(missing)}. "
            "Asegúrate de usar el endpoint organic-keywords o el CSV correcto."
        )

    work = df.copy()
    work["url"] = work["url"].apply(clean_url)
    work = work.dropna(subset=["url", "keyword", "best_position"])
    work = work[work["url"] != ""]

    # Filtros: posición y volumen mínimos.
    work["best_position"] = pd.to_numeric(work["best_position"], errors="coerce")
    work = work.dropna(subset=["best_position"])
    work = work[work["best_position"] <= max_position]

    if "volume" in work.columns:
        work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
        work = work[work["volume"] >= min_volume]

    # Canibalización = misma keyword en ≥2 URLs distintas.
    grouped = work.groupby("keyword")["url"].nunique()
    cannibal_kws = grouped[grouped > 1].index
    cannibal = work[work["keyword"].isin(cannibal_kws)].copy()

    if cannibal.empty:
        return cannibal

    # Enriquecimiento: tipo de página + intención (si hay flags).
    cannibal["page_type"] = cannibal["url"].apply(detect_page_type)
    if any(c.startswith("is_") for c in cannibal.columns):
        cannibal["intent"] = cannibal.apply(detect_intent, axis=1)
    else:
        cannibal["intent"] = "desconocida"

    return cannibal.sort_values(["keyword", "best_position"])


def score_group(group: pd.DataFrame) -> dict:
    """
    Calcula severidad + score de impacto para un grupo canibalizado.

    Severidad:
      - Alta: ≥2 URLs en top 10 y volumen ≥100, o intención comercial/transaccional con vol ≥50.
      - Media: 1 URL en top 10 + otra en 11-20, o volumen 50-100.
      - Baja: posiciones distantes / branded / volumen <50.

    Score = volumen × Σ CTR(posición) × nº URLs (más URLs = más fragmentación).
    """
    n_urls = group["url"].nunique()
    volume = float(group["volume"].iloc[0]) if "volume" in group.columns else 0.0
    positions = group["best_position"].astype(float).tolist()
    in_top10 = sum(1 for p in positions if p <= 10)
    intent = group["intent"].iloc[0] if "intent" in group.columns else "desconocida"

    # Severidad
    if intent == "branded":
        severity = "Baja"
    elif in_top10 >= 2 and volume >= 100:
        severity = "Alta"
    elif intent in {"comercial", "transaccional"} and volume >= 50:
        severity = "Alta"
    elif in_top10 >= 1 and any(11 <= p <= 20 for p in positions):
        severity = "Media"
    elif 50 <= volume < 100:
        severity = "Media"
    else:
        severity = "Baja"

    # Score de impacto (tráfico estimado fragmentado)
    ctr_sum = sum(estimated_ctr(p) for p in positions)
    score = round(volume * ctr_sum * (1 + 0.1 * (n_urls - 2)), 1)

    return {"severity": severity, "score": score, "n_urls": n_urls, "in_top10": in_top10}


# ---------------------------------------------------------------------------
# Claude — prompt enriquecido
# ---------------------------------------------------------------------------

ACTION_SYSTEM_PROMPT = """Eres un consultor SEO senior español especializado en \
resolver canibalizaciones de keywords. Recibes un grupo de URLs que compiten \
por la misma keyword, con datos de Ahrefs (volumen, KD, posición, tráfico, \
intención, tipo de página).

Devuelve UNA acción entre estas (usa el texto exacto):
- "Consolidar": fusionar URLs en una y redirigir 301 las demás (misma intención).
- "Redirigir 301": una URL es claramente mejor; las otras desaparecen y se redirigen.
- "Desindexar": URLs sin tráfico ni intención propia (noindex o eliminar).
- "Diferenciar intención": las URLs cubren intenciones distintas; reescribir y \
reorientar enlazado interno.
- "Reescribir y reforzar": hay una URL principal clara pero mal optimizada o sin autoridad.
- "Mantener y monitorizar": canibalización leve (branded, posiciones distantes) sin acción.

Prioriza como URL principal la que tenga mejor combinación de: posición, tráfico, \
y coherencia con la intención de la keyword (ej. para "comprar X", una página \
de producto o categoría es mejor que un blog post).

Responde SOLO un JSON válido (sin markdown, sin texto extra):

{
  "accion": "<acción exacta de la lista>",
  "url_principal": "<URL canónica recomendada o '' si no aplica>",
  "justificacion": "<2-3 frases concretas y accionables, en español, citando \
los datos relevantes (posiciones, intención, tipo de página)>"
}

No inventes datos que no estén en el grupo. Sé específico."""


def build_user_prompt(keyword: str, group: pd.DataFrame, score_data: dict) -> str:
    volume = group["volume"].iloc[0] if "volume" in group.columns else "N/D"
    kd = group["keyword_difficulty"].iloc[0] if "keyword_difficulty" in group.columns else "N/D"
    intent = group["intent"].iloc[0] if "intent" in group.columns else "desconocida"

    rows = []
    for _, r in group.iterrows():
        parts = [f"URL: {r['url']}"]
        parts.append(f"Posición: {r['best_position']}")
        if "traffic" in group.columns and pd.notna(r.get("traffic")):
            parts.append(f"Tráfico: {r['traffic']}")
        if "page_type" in group.columns:
            parts.append(f"Tipo: {r['page_type']}")
        rows.append("- " + " | ".join(parts))

    return (
        f'Keyword canibalizada: "{keyword}"\n'
        f'Volumen: {volume} | KD: {kd} | Intención: {intent} | '
        f'Severidad calculada: {score_data["severity"]} | '
        f'URLs en top 10: {score_data["in_top10"]} de {score_data["n_urls"]}\n\n'
        f"URLs que compiten por esta keyword:\n"
        + "\n".join(rows)
        + "\n\nDevuélveme la acción recomendada en el JSON descrito."
    )


def ask_claude(client: Anthropic, keyword: str, group: pd.DataFrame, score_data: dict) -> dict:
    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=ACTION_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_prompt(keyword, group, score_data)}
            ],
        )
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", "") == "text"
        ).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        return {
            "accion": data.get("accion", "").strip(),
            "url_principal": data.get("url_principal", "").strip(),
            "justificacion": data.get("justificacion", "").strip(),
        }
    except (json.JSONDecodeError, APIError, Exception) as e:
        return {
            "accion": "Error",
            "url_principal": "",
            "justificacion": f"No se pudo generar recomendación ({type(e).__name__}).",
        }


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def build_excel(cannibal: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Resumen", index=False)

        detail_cols = [
            "keyword", "url", "best_position", "page_type", "intent",
        ]
        for opt in ("volume", "keyword_difficulty", "traffic"):
            if opt in cannibal.columns:
                detail_cols.append(opt)

        merged = cannibal[detail_cols].merge(
            summary[["keyword", "Severidad", "Score impacto", "Acción",
                     "URL principal", "Justificación"]],
            on="keyword",
            how="left",
        )
        merged.to_excel(writer, sheet_name="Detalle", index=False)

        # Auto-anchura
        for sheet in ("Resumen", "Detalle"):
            ws = writer.sheets[sheet]
            for col in ws.columns:
                length = max(
                    (len(str(c.value)) if c.value is not None else 0) for c in col
                )
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(length + 2, 12), 60
                )

        # Coloreado de severidad en el Resumen
        from openpyxl.styles import PatternFill
        ws = writer.sheets["Resumen"]
        sev_col = None
        for idx, cell in enumerate(ws[1], start=1):
            if cell.value == "Severidad":
                sev_col = idx
                break
        if sev_col:
            colors = {
                "Alta": "F8B4B4",
                "Media": "FDE68A",
                "Baja": "BBF7D0",
            }
            for row in ws.iter_rows(min_row=2, min_col=sev_col, max_col=sev_col):
                cell = row[0]
                fill = colors.get(cell.value)
                if fill:
                    cell.fill = PatternFill("solid", fgColor=fill)

    return output.getvalue()


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    :root {
        --lfs-primary: #E63946;
        --lfs-dark: #1D3557;
        --lfs-light: #F1FAEE;
        --lfs-accent: #457B9D;
    }
    .main .block-container { padding-top: 2rem; max-width: 1300px; }
    h1 { color: var(--lfs-dark); font-weight: 700; }
    h2, h3 { color: var(--lfs-dark); }
    .stButton>button {
        background: var(--lfs-primary); color: white; border: none;
        font-weight: 600; border-radius: 6px; padding: 0.5rem 1.2rem;
    }
    .stButton>button:hover { background: var(--lfs-dark); color: white; }
    .stDownloadButton>button {
        background: var(--lfs-dark); color: white; border: none;
        font-weight: 600; border-radius: 6px;
    }
    .stDownloadButton>button:hover { background: var(--lfs-accent); color: white; }
    [data-testid="stMetricValue"] { color: var(--lfs-primary); font-weight: 700; }
    [data-testid="stSidebar"] { background-color: #FAFAFA; }
    .severity-Alta { background: #F8B4B4; padding: 2px 8px; border-radius: 4px; }
    .severity-Media { background: #FDE68A; padding: 2px 8px; border-radius: 4px; }
    .severity-Baja { background: #BBF7D0; padding: 2px 8px; border-radius: 4px; }
</style>
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _get_secret(key: str) -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key] or ""
    except Exception:
        pass
    return os.environ.get(key, "")


def run_analysis(
    df: pd.DataFrame,
    client_name: str,
    use_claude: bool,
    max_groups: int,
    anthropic_key: str,
    max_position: int,
    min_volume: int,
    button_key: str,
) -> None:
    try:
        cannibal = detect_canibalizations(df, max_position=max_position, min_volume=min_volume)
    except ValueError as e:
        st.error(str(e))
        with st.expander("Columnas detectadas"):
            st.write(list(df.columns))
        return

    if cannibal.empty:
        st.success("✅ No se detectaron canibalizaciones con los umbrales actuales.")
        return

    # Calcular score y severidad por grupo.
    summary_records = []
    score_by_kw = {}
    for kw, group in cannibal.groupby("keyword"):
        s = score_group(group)
        score_by_kw[kw] = s
        summary_records.append({
            "keyword": kw,
            "Nº URLs": s["n_urls"],
            "URLs en top 10": s["in_top10"],
            "Volumen": int(group["volume"].iloc[0]) if "volume" in group.columns else None,
            "KD": int(group["keyword_difficulty"].iloc[0]) if "keyword_difficulty" in group.columns and pd.notna(group["keyword_difficulty"].iloc[0]) else None,
            "Intención": group["intent"].iloc[0] if "intent" in group.columns else "desconocida",
            "Severidad": s["severity"],
            "Score impacto": s["score"],
            "Acción": "",
            "URL principal": "",
            "Justificación": "",
        })
    summary = pd.DataFrame(summary_records).sort_values(
        by=["Severidad", "Score impacto"],
        key=lambda c: c.map({"Alta": 0, "Media": 1, "Baja": 2}) if c.name == "Severidad" else c,
        ascending=[True, False],
    ).reset_index(drop=True)

    # Métricas de cabecera
    sev_counts = summary["Severidad"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Canibalizaciones", len(summary))
    c2.metric("🔴 Severidad alta", int(sev_counts.get("Alta", 0)))
    c3.metric("🟡 Severidad media", int(sev_counts.get("Media", 0)))
    c4.metric("🟢 Severidad baja", int(sev_counts.get("Baja", 0)))

    st.divider()

    # Filtros sobre el resumen
    st.subheader("Resumen priorizado")
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        sev_filter = st.multiselect(
            "Filtrar por severidad",
            ["Alta", "Media", "Baja"],
            default=["Alta", "Media", "Baja"],
            key=f"sev_{button_key}",
        )
    with fc2:
        intent_options = sorted(summary["Intención"].dropna().unique().tolist())
        intent_filter = st.multiselect(
            "Filtrar por intención",
            intent_options,
            default=intent_options,
            key=f"int_{button_key}",
        )

    view = summary[
        summary["Severidad"].isin(sev_filter) &
        summary["Intención"].isin(intent_filter)
    ].reset_index(drop=True)
    st.dataframe(view, use_container_width=True, hide_index=True)

    with st.expander(f"Ver detalle ({len(cannibal)} filas URL × keyword)"):
        st.dataframe(cannibal, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Generar informe con acciones SEO")

    if not st.button("🚀 Procesar con Claude y generar Excel", type="primary", key=button_key):
        return

    # Solo procesamos los grupos visibles tras filtros.
    keywords_to_process = view["keyword"].tolist()
    actions_records = []

    if use_claude:
        if not anthropic_key:
            st.error("Falta la Anthropic API Key. Añádela en la barra lateral.")
            return
        try:
            client = Anthropic(api_key=anthropic_key)
        except Exception as e:
            st.error(f"No se pudo inicializar Anthropic: {e}")
            return

        progress = st.progress(0.0, text="Analizando grupos…")
        total = min(len(keywords_to_process), max_groups)

        for i, kw in enumerate(keywords_to_process):
            if i >= max_groups:
                actions_records.append({
                    "keyword": kw,
                    "Acción": "Pendiente (límite alcanzado)",
                    "URL principal": "",
                    "Justificación": "",
                })
                continue

            group = cannibal[cannibal["keyword"] == kw]
            res = ask_claude(client, kw, group, score_by_kw[kw])
            actions_records.append({
                "keyword": kw,
                "Acción": res["accion"],
                "URL principal": res["url_principal"],
                "Justificación": res["justificacion"],
            })
            progress.progress((i + 1) / total, text=f"Analizando… {i + 1}/{total}")
            time.sleep(0.2)
        progress.empty()
    else:
        for kw in keywords_to_process:
            actions_records.append({"keyword": kw, "Acción": "", "URL principal": "", "Justificación": ""})

    actions_df = pd.DataFrame(actions_records)
    final_summary = view.drop(columns=["Acción", "URL principal", "Justificación"]).merge(
        actions_df, on="keyword", how="left"
    )

    st.subheader("Acciones recomendadas")
    st.dataframe(final_summary, use_container_width=True, hide_index=True)

    # Filtramos cannibal a solo las keywords procesadas para que el detalle case.
    cannibal_filtered = cannibal[cannibal["keyword"].isin(keywords_to_process)]
    excel_bytes = build_excel(cannibal_filtered, final_summary)
    safe_name = (client_name or "cliente").strip().replace(" ", "_")
    st.download_button(
        label="⬇️ Descargar Excel",
        data=excel_bytes,
        file_name=f"Canibalizaciones_{safe_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        key=f"dl_{button_key}",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🥩 Detector de Canibalizaciones SEO")
    st.caption(
        "Detecta canibalizaciones reales en proyectos SEO usando Ahrefs API. "
        "Calcula severidad, prioriza por impacto y sugiere acciones con Claude."
    )

    # --- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.subheader("⚙️ Configuración")

        anthropic_key = st.text_input(
            "Anthropic API Key", type="password",
            value=_get_secret("ANTHROPIC_API_KEY"),
            help="En Streamlit Cloud → Secrets como ANTHROPIC_API_KEY.",
        )
        ahrefs_key = st.text_input(
            "Ahrefs API Key", type="password",
            value=_get_secret("AHREFS_API_KEY"),
            help="En Streamlit Cloud → Secrets como AHREFS_API_KEY.",
        )
        client_name = st.text_input(
            "Nombre del cliente", placeholder="Ej. Cronomía",
            help="Se usa en el nombre del Excel.",
        )

        st.divider()
        country_code = st.selectbox(
            "País",
            options=list(COUNTRIES.keys()),
            format_func=lambda c: f"{c.upper()} · {COUNTRIES[c]}",
            index=0,
        )

        st.divider()
        st.markdown("**Umbrales de detección**")
        max_position = st.slider(
            "Posición máxima a considerar", 5, 50, 20,
            help="Solo se cuentan keywords donde una URL rankea por encima de esta posición.",
        )
        min_volume = st.number_input(
            "Volumen mínimo de búsqueda", 0, 10000, 10, step=10,
            help="Filtra keywords con volumen menor (cola larga residual).",
        )

        st.divider()
        use_claude = st.toggle(
            "Generar recomendaciones con Claude", value=True,
        )
        max_groups = st.number_input(
            "Máx. grupos a analizar con Claude",
            min_value=1, max_value=500, value=50, step=10,
        )

    # --- Tabs --------------------------------------------------------------
    tab_api, tab_csv = st.tabs(["🔗 API Ahrefs (recomendado)", "📄 CSV (fallback)"])

    # --- API ---------------------------------------------------------------
    with tab_api:
        st.markdown(
            "Trae **todas las parejas keyword × URL** del informe `Organic keywords` "
            "de Ahrefs. Detecta canibalizaciones reales (la misma keyword rankeando "
            "en varias URLs simultáneamente)."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            target = st.text_input(
                "Dominio objetivo", placeholder="ej. cronomia.com",
                help="Sin http:// ni www.",
            )
        with col2:
            mode_label = st.selectbox(
                "Modo", options=list(MODES.keys()),
                format_func=lambda m: MODES[m], index=0,
            )

        col3, col4 = st.columns(2)
        with col3:
            limit = st.number_input(
                "Límite de keywords a traer",
                min_value=500, max_value=20000, value=5000, step=500,
                help="Cuantas más, más unidades de Ahrefs consume.",
            )
        with col4:
            on_date = st.date_input("Fecha del informe", value=date.today())

        if st.button("📥 Traer Organic keywords", type="primary", key="btn_fetch"):
            if not ahrefs_key:
                st.error("Falta la Ahrefs API Key.")
            elif not target:
                st.error("Introduce un dominio objetivo.")
            else:
                try:
                    with st.spinner(f"Consultando Ahrefs ({target}, {country_code.upper()}, {MODES[mode_label]})…"):
                        df = fetch_organic_keywords(
                            api_key=ahrefs_key,
                            target=target.strip(),
                            country=country_code,
                            mode=mode_label,
                            limit=int(limit),
                            on_date=on_date.isoformat(),
                        )
                    st.session_state["ahrefs_df"] = df
                    st.success(f"✅ {len(df):,} keywords recibidas para {target}.")
                except (PermissionError, ValueError, RuntimeError) as e:
                    st.error(str(e))
                except requests.RequestException as e:
                    st.error(f"Error de red: {e}")

        df_api = st.session_state.get("ahrefs_df")
        if df_api is not None and not df_api.empty:
            st.divider()
            run_analysis(
                df_api, client_name, use_claude, int(max_groups),
                anthropic_key, int(max_position), int(min_volume),
                button_key="btn_api",
            )
        elif df_api is not None:
            st.info("Ahrefs no devolvió keywords para esos parámetros.")

    # --- CSV ---------------------------------------------------------------
    with tab_csv:
        st.markdown(
            "Fallback: sube el CSV de **Top Pages** de Ahrefs (formato del Colab original). "
            "⚠️ Top Pages tiene **menos detalle** que Organic keywords: solo ve la "
            "*Top keyword* de cada URL, así que se escapan canibalizaciones de keywords "
            "secundarias. Se recomienda usar la API."
        )
        uploaded = st.file_uploader(
            "Sube el CSV de Top Pages", type=["csv"], accept_multiple_files=False,
        )
        if uploaded:
            try:
                df_csv = read_top_pages_csv(uploaded.getvalue())
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
                return
            st.success(f"CSV cargado: {len(df_csv):,} filas.")
            st.divider()
            run_analysis(
                df_csv, client_name, use_claude, int(max_groups),
                anthropic_key, int(max_position), int(min_volume),
                button_key="btn_csv",
            )


if __name__ == "__main__":
    main()
