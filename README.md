# Detector de Canibalizaciones SEO — La Fábrica del SEO

App Streamlit para detectar canibalizaciones reales en proyectos SEO usando
**Ahrefs API v3** (`organic-keywords`), priorizar por severidad e impacto, y
sugerir acciones con Claude.

## Qué la diferencia

A diferencia de un análisis basado en *Top keyword* (CSV de Top Pages), esta
versión usa el endpoint `organic-keywords` que devuelve **todas las parejas
keyword × URL** que rankean. Esto detecta canibalizaciones que un análisis de
"Top keyword duplicada" se pierde — por ejemplo, cuando una URL tiene la
keyword como principal pero otra rankea para la misma keyword como secundaria.

## Funcionalidades

### Detección
- Endpoint `site-explorer/organic-keywords` (v3).
- Filtros configurables: posición máxima (default 20), volumen mínimo (default 10).
- Modo `subdomains` por defecto (configurable: domain, prefix, exact).
- Selector de país en la sidebar.

### Severidad y priorización
Cada grupo canibalizado recibe:
- **Severidad** (Alta / Media / Baja) según reglas SEO:
  - **Alta:** ≥2 URLs en top 10 + volumen ≥100, o intención comercial/transaccional con vol ≥50.
  - **Media:** 1 URL top 10 + otra entre 11-20, o volumen 50-100.
  - **Baja:** posiciones distantes, branded, volumen <50.
- **Score de impacto** (= volumen × Σ CTR esperado × nº URLs) para ordenar por urgencia.
- **Tipo de página** detectado por patrón URL (blog, categoría, producto, servicio).
- **Intención** desde flags de Ahrefs (informacional, comercial, transaccional, navegacional, branded).

### Recomendaciones de Claude
El prompt incluye contexto enriquecido: volumen, KD, intención, tipo de página,
posición de cada URL, severidad calculada. Las recomendaciones citan datos
específicos en lugar de sugerencias genéricas.

Acciones posibles: *Consolidar*, *Redirigir 301*, *Desindexar*,
*Diferenciar intención*, *Reescribir y reforzar*, *Mantener y monitorizar*.

### Workflow
1. Filtros visuales por severidad e intención antes de procesar (no quemas
   unidades de Claude analizando lo irrelevante).
2. Excel con dos pestañas:
   - **Resumen** ordenado por severidad → score, con coloreado de severidad
     (rojo / amarillo / verde).
   - **Detalle** con todas las parejas URL × keyword.

## Estructura

```
canibalizaciones-app/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Ejecutar en local

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
export AHREFS_API_KEY="..."
streamlit run app.py
```

## Desplegar en Streamlit Cloud

1. Sube el repo a GitHub.
2. share.streamlit.io → New app → repo + `app.py`.
3. Settings → Secrets:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   AHREFS_API_KEY = "..."
   ```

## Uso típico

### Modo API (recomendado)

1. Sidebar: cliente, país, umbrales (default top 20 + vol ≥10).
2. Pestaña **🔗 API Ahrefs** → dominio + modo + límite + fecha.
3. Pulsa **Traer Organic keywords**.
4. Revisa el resumen priorizado, filtra por severidad/intención si quieres.
5. Pulsa **Procesar con Claude y generar Excel**.

### Modo CSV (fallback)

Sube el CSV de Top Pages del export Ahrefs si no quieres usar la API.
Ten en cuenta que solo ve canibalizaciones donde una keyword es Top
keyword en ambas URLs (menos cobertura que la API).

## Detalles técnicos

- **Endpoint:** `GET https://api.ahrefs.com/v3/site-explorer/organic-keywords`
- **Auth:** `Authorization: Bearer <AHREFS_API_KEY>`
- **Campos solicitados:** `keyword, best_position, best_position_url, volume,
  keyword_difficulty, traffic, cpc, is_branded, is_commercial,
  is_informational, is_navigational, is_transactional`
- **Order:** `traffic:desc` (con el límite, capturas primero las kw con más tráfico).
- **CTR de referencia:** Sistrix / Advanced Web Ranking 2024 (top 10 = 30/16/10/7/5/4/3/2.5/2/1.8 %).
