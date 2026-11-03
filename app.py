import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import eda_core as core

st.set_page_config(page_title="Monitor de Datos de Minerales Críticos y PAM · SGM", layout="wide")

# ───────────── Paleta Institucional (Estilo Monitor EULACRM / SEGEMAR) ─────────────
GREEN, AMBER, RED, BLUE, GREY = "#3EA843", "#F2A900", "#D93829", "#3B82F6", "#646B73"
ECON = {"Pinta (>= minimo)": GREEN, "No pinta": GREY}
TOX = {"Supera NOM-147 (MX)": RED, "Supera ref. europea": AMBER, "Debajo de limites": BLUE}
ESC_ECON = [[0, "#181B20"], [0.5, "#2C3540"], [0.75, GREEN], [1, AMBER]]
ESC_TOX = [[0, "#181B20"], [0.5, "#2C3540"], [0.75, AMBER], [1, RED]]
ESC_TICKS = dict(tickvals=[-2, -1, 0, 1, 2], ticktext=["0.01x", "0.1x", "1x (umbral)", "10x", "100x"])

pio.templates["institucional"] = go.layout.Template(layout=dict(
    font=dict(family="Segoe UI, Roboto, Helvetica, sans-serif", color="#D4D7DD", size=11),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1A1D21",
    colorway=[BLUE, GREEN, AMBER, RED, "#8E6CEF", "#29A3A3", "#E06666", "#8F99A8"],
    xaxis=dict(gridcolor="#2B3038", zerolinecolor="#2B3038", showline=True, linecolor="#333842"),
    yaxis=dict(gridcolor="#2B3038", zerolinecolor="#2B3038", showline=True, linecolor="#333842")))
pio.templates.default = "plotly_dark+institucional"

st.markdown("""<style>
.block-container { padding-top: 0.8rem; padding-bottom: 1.5rem; max-width: 100%; }
header[data-testid="stHeader"] { background: #141619; border-bottom: 1px solid #2A2E35; }
.topbar {
    background: #1B1E23; border: 1px solid #2E333B; border-radius: 3px;
    padding: 8px 16px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;
}
.topbar-title { font-size: 1.05rem; font-weight: 600; color: #FFFFFF; letter-spacing: 0.01em; }
.badge-proj {
    background: #2578B5; color: #FFFFFF; font-size: 0.72rem; font-weight: 600;
    padding: 2px 8px; border-radius: 3px; margin-left: 10px; vertical-align: middle;
}
.topbar-meta { font-size: 0.78rem; color: #8F96A0; }

/* Pestañas estilo botonera institucional */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    background-color: #262A32; color: #B0B7C0; border-radius: 3px 3px 0 0;
    padding: 6px 16px; margin-right: 4px; border: 1px solid #323741; border-bottom: none;
    font-size: 0.82rem; font-weight: 500;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #2578B5; color: #FFFFFF; border-color: #2578B5; font-weight: 600;
}

/* Paneles y tarjetas KPI */
.kpi-card {
    background: #1F2227; border: 1px solid #2F343D; border-radius: 3px;
    padding: 14px 12px; text-align: center; margin-bottom: 8px; height: 100%;
}
.kpi-label { font-size: 0.76rem; color: #A6ADB7; margin-bottom: 6px; font-weight: 500; }
.kpi-val { font-size: 1.95rem; font-weight: 700; line-height: 1.1; margin-bottom: 4px; }
.kpi-sub { font-size: 0.72rem; color: #78808C; }
.panel-hdr {
    font-size: 0.84rem; font-weight: 600; color: #CED4DC;
    border-bottom: 1px solid #2D323A; padding-bottom: 4px; margin-bottom: 8px;
}
</style>""", unsafe_allow_html=True)


def kpi_box(col, titulo, valor, subtitulo, color="#FFFFFF"):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{titulo}</div>'
        f'<div class="kpi-val" style="color:{color}">{valor}</div>'
        f'<div class="kpi-sub">{subtitulo}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="topbar">'
    '<div><span class="topbar-title">Dashboard - Monitor para la Gestión de Datos de Minerales Críticos y Pasivos Mineros</span>'
    '<span class="badge-proj">Proyecto EULACRM · SGM</span></div>'
    '<div class="topbar-meta">Evaluación de Leyes de Corte, Especies Minerales, Óxidos y Normativa Ambiental (NOM-147 / UE)</div>'
    '</div>',
    unsafe_allow_html=True
)

# ───────────── Carga de Archivo ─────────────
with st.sidebar:
    st.markdown('<div class="panel-hdr">Origen de Datos</div>', unsafe_allow_html=True)
    archivo = st.file_uploader("Cargar Base General (Excel)", type=["xlsx"])

if archivo is not None:
    contenido = archivo.getvalue()
elif Path("Base_General.xlsx").exists():
    contenido = Path("Base_General.xlsx").read_bytes()
elif Path("EDA.xlsx").exists():
    contenido = Path("EDA.xlsx").read_bytes()
else:
    st.info("Cargue el archivo Excel de la Base General en el panel lateral para iniciar el análisis.")
    st.stop()


@st.cache_data(show_spinner="Procesando base geoquímica y proyecciones espaciales...")
def cargar(b):
    return core.cargar_base(b)


@st.cache_data(show_spinner="Calculando estadísticos y factores estequiométricos...")
def calcular(_meta, _vals, _cens, _umb, huella):
    return core.estadisticos(_meta, _vals, _cens, _umb)


meta, vals_raw, cens, ld = cargar(contenido)

# ───────────── Barra de Filtros Superior y Lateral ─────────────
with st.sidebar:
    st.markdown('<div class="panel-hdr">Especies Minerales y Óxidos</div>', unsafe_allow_html=True)
    especie_fe = st.selectbox(
        "Hierro (Fe) - Especie mineral",
        list(core.ESPECIES_FE.keys()),
        index=0,
        help="Magnetita (20-30% Fe), Goethita (30-40% Fe), Pirita (30-40% Fe), Hematita (40-50% Fe), Limonita (40-50% Fe)."
    )
    especie_cu = st.selectbox(
        "Cobre (Cu) - Tipo de mena",
        list(core.ESPECIES_CU.keys()),
        index=0,
        help="Óxidos (0.3-0.6% Cu) o Sulfuros (0.5-1.0% Cu)."
    )
    especie_zn = st.selectbox(
        "Zinc (Zn) - Tipo de mena",
        list(core.ESPECIES_ZN.keys()),
        index=0,
        help="Sulfuros (3-5% Zn) o No sulfuros (10-15% Zn)."
    )
    usar_oxidos = st.checkbox(
        "Convertir a óxidos estequiométricos (Al2O3, Cr2O3, TiO2, Li2O, WO3, BeO, CeO2, U3O8, ThO2, Ta2O5)",
        value=True,
        help="Multiplica la concentración elemental por su factor estequiométrico oficial para comparar contra leyes de corte en óxido."
    )
    sep_terreros = st.checkbox(
        "Desglosar terreros individuales por dimensiones",
        value=False,
        help="Evalúa cada cuerpo de terrero con medidas (L x W x H) como una unidad independiente."
    )
    usar_dpl = st.checkbox("Incluir muestras duplicadas (-DPL)", value=False)
    base_mapa = st.selectbox("Cartografía base", ["Oscuro", "Satélite", "Calles"], index=0)

# Barra horizontal de filtros estilo Monitor EULACRM
f1, f2, f3, f4 = st.columns([2, 2, 1.8, 2.2])
estados_disp = sorted(meta.estado.dropna().unique())
tipos_disp = sorted(meta.tipo.dropna().unique())
estados = f1.multiselect("Estado", estados_disp, default=estados_disp)
tipos = f2.multiselect("Tipo de depósito / muestra", tipos_disp, default=tipos_disp)
resolucion_mapa = f3.selectbox("Nivel de agregación espacial", ["Por sitio y tipo", "Por muestra individual"])
stat = "media"
f4.markdown(
    f'<div style="padding-top:24px;font-size:0.76rem;color:#9BA3AE;">'
    f'<b>Cortes activos:</b> Fe ({core.ESPECIES_FE[especie_fe][1]:.0f}%) · '
    f'Cu ({core.ESPECIES_CU[especie_cu][1]}%) · '
    f'Zn ({core.ESPECIES_ZN[especie_zn][1]:.0f}%) · '
    f'Óxidos: {"Sí" if usar_oxidos else "No"}'
    f'</div>',
    unsafe_allow_html=True
)

mask = meta.estado.isin(estados) & meta.tipo.isin(tipos) & (usar_dpl | ~meta.es_dpl)
meta_f = meta[mask].copy()
if sep_terreros:
    meta_f["grupo"] = meta_f.grupo_fino
if meta_f.empty:
    st.warning("La combinación actual de filtros no arroja registros.")
    st.stop()

umb_actual = core.configurar_parametros(especie_fe, especie_zn, especie_cu)
vals = core.aplicar_conversion_oxidos(vals_raw, usar_oxidos)

huella = hash((contenido[:64], len(contenido), tuple(estados), tuple(tipos), usar_dpl, sep_terreros, especie_fe, especie_zn, especie_cu, usar_oxidos))
est = calcular(meta_f, vals, cens, umb_actual, huella)
est_raw = calcular(meta_f, vals_raw, cens, umb_actual, huella + 1)

rmin = core.razones_min(est, stat, umb_actual, usar_oxidos)
rtox = core.razones_tox(est_raw)
est_ev = est[est.elemento.isin(list(umb_actual) + list(core.UMB_TOXICOS))]
res = core.resumen_grupos(meta_f, rmin, rtox)
muestras = core.clasificar_muestras(meta_f, vals, vals_raw, umb_actual)
res["pinta_lbl"] = np.where(res.pinta, "Pinta (>= minimo)", "No pinta")
muestras["pinta_lbl"] = np.where(muestras.pinta, "Pinta (>= minimo)", "No pinta")


# ───────────── Funciones de Graficación ─────────────
def mapa(df, base, **kw):
    d = df.dropna(subset=["lat", "lon"])
    fig = px.scatter_map(d, lat="lat", lon="lon", height=430, **kw)
    span = max(d.lat.max() - d.lat.min(), (d.lon.max() - d.lon.min()) * 0.9, 0.02)
    zoom = float(np.clip(np.log2(422 / span) - 0.3, 3, 14))
    fig.update_layout(
        map_center=dict(lat=d.lat.mean(), lon=d.lon.mean()), map_zoom=zoom,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=0.02, x=0.01, bgcolor="rgba(20,22,25,0.85)")
    )
    if base == "Satélite":
        fig.update_layout(map_style="white-bg", map_layers=[dict(
            below="traces", sourcetype="raster",
            source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"])])
    else:
        fig.update_layout(map_style="carto-darkmatter" if base == "Oscuro" else "open-street-map")
    if "size" not in kw:
        fig.update_traces(marker_size=11)
    return fig


def heat(piv, escala, titulo_cb):
    L = np.log10(piv.where(piv > 0))
    fig = go.Figure(go.Heatmap(
        z=L.values, x=L.columns, y=L.index, zmin=-2, zmax=2, colorscale=escala, xgap=1, ygap=1,
        customdata=piv.values, hovertemplate="%{y}<br>%{x}: %{customdata:.2f}x<extra></extra>",
        colorbar=dict(title=titulo_cb, **ESC_TICKS)))
    fig.update_layout(height=max(360, 18 * len(L) + 100), margin=dict(l=0, r=0, t=10, b=0))
    fig.update_yaxes(autorange="reversed")
    return fig


# ───────────── Navegación por Pestañas (Sin Emojis) ─────────────
t1, t2, t8, t3, t4, t7, t5, t6 = st.tabs([
    "General", "Geología Económica", "Minerales Críticos", "Evaluación Ambiental",
    "Análisis Estadístico", "Zonas Prioritarias", "Control QA/QC", "Datos y Exportación"
])

# ═══ 1. GENERAL (Layout estilo Monitor SEGEMAR / EULACRM) ═══
with t1:
    prim = meta_f[~meta_f.es_dpl]
    pct_completitud = (1.0 - (est_ev.estatus == "Descartado").mean()) * 100

    col_left, col_mid, col_right = st.columns([1.1, 2.3, 2.6])

    with col_left:
        kpi_box(st, "Muestras primarias en base", f"{len(prim)}", f"Representa {res.estado.nunique()} estados y {res.sitio.nunique()} sitios", "#FFFFFF")
        kpi_box(st, "Total de elementos con corte", f"{len(umb_actual)}", f"{len(core.CRITICOS)} críticos en catálogo", "#FFFFFF")
        kpi_box(st, "Completitud analítica (>LD)", f"{pct_completitud:.0f} %", "Variables con censura <= 20 %", GREEN if pct_completitud >= 70 else AMBER)

    with col_mid:
        st.markdown('<div class="panel-hdr">Sitios por estado y clasificación económica / ambiental</div>', unsafe_allow_html=True)
        est_bar = res.assign(
            clase=np.where(res.pinta, "Con potencial (>= minimo)",
                  np.where(res.nivel_tox == "Supera NOM-147 (MX)", "Pasivo crítico (NOM-147)", "Bajo umbral"))
        ).groupby(["estado", "clase"]).size().reset_index(name="sitios")
        fig_bar1 = px.bar(
            est_bar, y="estado", x="sitios", color="clase", orientation="h", height=205,
            color_discrete_map={"Con potencial (>= minimo)": GREEN, "Pasivo crítico (NOM-147)": RED, "Bajo umbral": AMBER},
            labels={"sitios": "Unidades evaluadas", "estado": "", "clase": ""}
        )
        fig_bar1.update_layout(margin=dict(l=0, r=10, t=5, b=5), legend=dict(orientation="h", y=-0.28, x=0))
        st.plotly_chart(fig_bar1, width="stretch")

        st.markdown('<div class="panel-hdr">Índice de excedencia económica máxima por unidad</div>', unsafe_allow_html=True)
        top_g = res.dropna(subset=["razon_econ"]).sort_values("razon_econ", ascending=True).tail(10)
        fig_bar2 = px.bar(
            top_g, y="grupo", x="razon_econ", orientation="h", height=195,
            color_discrete_sequence=[BLUE], labels={"razon_econ": "Razón (Promedio / Mínimo Minable)", "grupo": ""}
        )
        fig_bar2.add_vline(x=1, line_dash="dash", line_color="#FFFFFF")
        fig_bar2.update_layout(margin=dict(l=0, r=10, t=5, b=5))
        st.plotly_chart(fig_bar2, width="stretch")

    with col_right:
        vista_mapa_gen = st.radio(
            "Capa temática del mapa",
            ["Potencial Económico (Leyes de Corte)", "Riesgo Ambiental (NOM-147 / Ref. UE)"],
            horizontal=True
        )
        if resolucion_mapa == "Por sitio y tipo":
            if vista_mapa_gen.startswith("Potencial"):
                hd = {"n": True, "elementos_rentables": True, "lat": False, "lon": False}
                st.plotly_chart(mapa(res, base_mapa, color="pinta_lbl", color_discrete_map=ECON,
                                     hover_name="grupo", hover_data=hd, size="n", size_max=18), width="stretch")
            else:
                ht = {"n": True, "excede_mx": True, "excede_eu": True, "lat": False, "lon": False}
                st.plotly_chart(mapa(res, base_mapa, color="nivel_tox", color_discrete_map=TOX,
                                     hover_name="grupo", hover_data=ht, size="n", size_max=18), width="stretch")
        else:
            if vista_mapa_gen.startswith("Potencial"):
                st.plotly_chart(mapa(muestras, base_mapa, color="pinta_lbl", color_discrete_map=ECON, hover_name="clave",
                                     hover_data={"sitio": True, "elem_econ": True, "razon_econ": ":.2f", "lat": False, "lon": False}), width="stretch")
            else:
                st.plotly_chart(mapa(muestras, base_mapa, color="nivel_tox", color_discrete_map=TOX, hover_name="clave",
                                     hover_data={"sitio": True, "razon_mx": ":.2f", "razon_eu": ":.2f", "lat": False, "lon": False}), width="stretch")

    # Fila inferior estilo Monitor EULACRM (Donut a la izquierda + 5 tarjetas de indicadores)
    b_donut, b_k1, b_k2, b_k3, b_k4, b_k5 = st.columns([1.5, 1, 1, 1, 1, 1])
    with b_donut:
        st.markdown('<div class="panel-hdr">Muestras por tipo de depósito</div>', unsafe_allow_html=True)
        fig_d = px.pie(prim, names="tipo", hole=0.58, height=170, color_discrete_sequence=[BLUE, RED, AMBER, GREEN])
        fig_d.update_traces(textinfo="none")
        fig_d.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)))
        st.plotly_chart(fig_d, width="stretch")

    n_rent = int(res.pinta.sum())
    n_nom = int((res.nivel_tox == "Supera NOM-147 (MX)").sum())
    n_eu = int((res.nivel_tox == "Supera ref. europea").sum())
    n_terr_vol = int(res.vol_m3.notna().sum())

    kpi_box(b_k1, "Unidades que cumplen corte", f"{n_rent}", f"{res.pinta.mean():.0%} de {len(res)} grupos", GREEN)
    kpi_box(b_k2, "Unidades con mineral crítico", f"{int((rmin[rmin.elemento.isin(core.PRIORITARIOS_PROYECTO)].razon >= 1).groupby(rmin.grupo).any().sum())}", "Superan ley mínima", "#FFFFFF")
    kpi_box(b_k3, "Exceden límite NOM-147", f"{n_nom} de {len(res)}", f"+{n_eu} superan ref. europea", AMBER if n_nom == 0 else RED)
    kpi_box(b_k4, "Terreros con cubicación", f"{n_terr_vol}", f"Volumen total: {res.vol_m3.sum():,.0f} m3", AMBER)
    kpi_box(b_k5, "Pares de control (-DPL)", f"{int(meta_f.es_dpl.sum())}", "Control de calidad QA/QC", "#FFFFFF")

# ═══ 2. GEOLOGÍA ECONÓMICA ═══
with t2:
    elems = [e for e in umb_actual if e in rmin.elemento.unique()]
    c_sel1, c_sel2 = st.columns([1, 2])
    el = c_sel1.selectbox("Seleccionar elemento o compuesto", elems, index=elems.index("Pb") if "Pb" in elems else 0, key="el_min")
    d = rmin[(rmin.elemento == el) & rmin.razon.notna()].sort_values("razon")
    u, mn = umb_actual[el][1], umb_actual[el][0]
    etiq = core.etiqueta_elemento(el, usar_oxidos)
    c_sel2.markdown(
        f'<div style="padding-top:28px;font-size:0.85rem;color:#B0B7C0;">'
        f'<b>Evaluación actual:</b> {etiq} &nbsp;|&nbsp; <b>Umbral mínimo minable:</b> {mn} {u} '
        f'{"(Convertido a óxido por factor " + str(core.FACTORES_OXIDO[el][1]) + ")" if (usar_oxidos and el in core.FACTORES_OXIDO) else ""}'
        f'</div>',
        unsafe_allow_html=True
    )

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f'<div class="panel-hdr">{etiq}: Promedio del sitio entre mínimo minable ({mn} {u})</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_bar(y=d.grupo, x=d.razon, orientation="h", name="Promedio", marker_color=np.where(d.razon >= 1, GREEN, "#3A424E"))
        fig.add_vline(x=1, line_dash="dash", line_color="#FFFFFF")
        fig.update_xaxes(type="log", title="Razón respecto al mínimo minable (escala log)")
        fig.update_layout(height=max(400, 20 * len(d) + 100), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with m2:
        st.markdown(f'<div class="panel-hdr">Distribución espacial de {etiq}</div>', unsafe_allow_html=True)
        dm = d.merge(res[["grupo", "lat", "lon"]], on="grupo")
        dm["log_razon"] = np.log10(dm.razon)
        fig = mapa(dm, base_mapa, color="log_razon", color_continuous_scale=ESC_ECON, range_color=(-2, 2),
                   size="n", size_max=20, hover_name="grupo",
                   hover_data={"valor": ":.3f", "razon": ":.2f", "pct_sobre_min": ":.0%", "n": True,
                               "log_razon": False, "lat": False, "lon": False})
        fig.update_coloraxes(colorbar=dict(title="", **ESC_TICKS))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="panel-hdr">Matriz de calor: Sitios por Elementos / Óxidos (Razón sobre umbral mínimo)</div>', unsafe_allow_html=True)
    st.plotly_chart(heat(rmin.pivot_table(index="grupo", columns="etiqueta_elem", values="razon"), ESC_ECON, ""), width="stretch")

    st.markdown('<div class="panel-hdr">Contraste de elementos declarados contra leyes confirmadas</div>', unsafe_allow_html=True)
    cmp_ = res[["grupo", "n", "declarados", "elementos_rentables", "no_declarados", "sin_confirmar"]].rename(columns={
        "declarados": "Elementos de interés (declarados)", "elementos_rentables": "Confirmados >= mínimo",
        "no_declarados": "Hallazgos adicionales (no declarados)", "sin_confirmar": "Declarados bajo umbral"})
    st.dataframe(cmp_, width="stretch", hide_index=True)

# ═══ 3. MINERALES CRÍTICOS (Ga, Ge, In, U, Th, Ce, Nd, PGE, REE, Co, V, Ti, Cr) ═══
with t8:
    st.markdown('<div class="panel-hdr">Evaluación de Materias Primas Críticas, Subproductos (Ga, Ge, In), Óxidos y Platinoides (Pt+Pd+Au)</div>', unsafe_allow_html=True)
    st.caption("conversiones estequiométricas activas para: Al2O3, Cr2O3, TiO2, Li2O, WO3, BeO, CeO2, U3O8, ThO2 y Ta2O5. El Fósforo (700 ppm P) y el Vanadio (0.1% V) se evalúan en forma elemental directa conforme al documento técnico.")

    sin_min = [e for e in core.CRITICOS if e not in umb_actual and e not in core.REE_IND and e in vals.columns]
    if sin_min:
        with st.expander("Configurar umbrales personalizados para elementos adicionales sin ley de corte predefinida"):
            ed = st.data_editor(
                pd.DataFrame({"Elemento": sin_min, "Mínimo": 0.0, "Unidad": "ppm"}),
                hide_index=True, key="min_crit", disabled=["Elemento"], width="stretch",
                column_config={"Unidad": st.column_config.SelectboxColumn(options=["ppm", "%", "g/t"]),
                               "Mínimo": st.column_config.NumberColumn(min_value=0.0)}
            )
        extra_dict = {e: (v, u) for e, v, u in zip(ed["Elemento"], ed["Mínimo"], ed["Unidad"])}
    else:
        extra_dict = {}

    minimos = core.minimos_ppm(umb_actual, extra_dict)
    rc = core.razones_criticos(est, minimos)
    idx = meta_f.index

    filas = []
    for e, (nom, uec, ues, us) in core.CRITICOS.items():
        if e not in vals.columns:
            continue
        x = vals.loc[idx, e]
        etiq_ox = core.etiqueta_elemento(e, usar_oxidos)
        filas.append({
            "Elemento / Compuesto": etiq_ox, "Referencia Documental": nom,
            "UE Crítica": "Sí" if uec else "No", "UE Estratégica": "Sí" if ues else "No",
            "USGS": "Sí" if us else "No", "Prioritario Proyecto": "Sí" if e in core.PRIORITARIOS_PROYECTO else "No",
            "% < LD": cens.loc[idx, e].sum() / max(x.notna().sum(), 1) * 100,
            "Promedio (ppm)": x.mean(), "Máximo (ppm)": x.max(),
            "Umbral Corte (ppm)": minimos.get(e, np.nan),
            "Sitios >= Umbral": rc[(rc.elemento == e) & (rc.razon >= 1)].grupo.nunique() if e in minimos else np.nan
        })
    tc = pd.DataFrame(filas)
    vista = st.radio("Filtro de catálogo", ["Todos", "Prioritarios del Proyecto (Ga, Ge, In, U, Th, Ce, Nd, PGE, REE, Co, Cr, Ti, V)", "Estratégicos UE", "Con ley de corte activa"], horizontal=True)
    if vista.startswith("Prioritarios"):
        tc = tc[tc["Prioritario Proyecto"] == "Sí"]
    elif vista == "Estratégicos UE":
        tc = tc[tc["UE Estratégica"] == "Sí"]
    elif vista == "Con ley de corte activa":
        tc = tc[tc["Umbral Corte (ppm)"].notna()]

    st.dataframe(tc, width="stretch", hide_index=True, column_config={
        "% < LD": st.column_config.NumberColumn(format="%.1f %%"),
        "Promedio (ppm)": st.column_config.NumberColumn(format="%.2f"),
        "Máximo (ppm)": st.column_config.NumberColumn(format="%.2f"),
        "Umbral Corte (ppm)": st.column_config.NumberColumn(format="%.4g")
    })
    st.caption("Parámetros del documento de referencia ausentes en las columnas analíticas actuales del Excel: " + ", ".join(core.CRITICOS_SIN_DATO) + ".")

    st.markdown('<div class="panel-hdr">Matriz comparativa de elementos críticos por sitio</div>', unsafe_allow_html=True)
    modo = st.radio("Métrica de visualización", ["Razón sobre ley de corte (Promedio / Mínimo)", "Percentil relativo dentro de la base (0 - 100)"], horizontal=True)
    disp = [e for e in core.CRITICOS if e in vals.columns and e not in core.REE_IND]
    defecto = [e for e in ["Ga", "Ge", "In", "U", "Th", "Ce", "Nd", "PGE", "REE", "Co", "Cr", "Ti", "V", "Sb", "W", "Be", "Mn", "Ni", "Cu", "As", "Bi", "Sc"] if e in disp]
    sel_c = st.multiselect("Seleccionar elementos críticos", disp, default=defecto)
    sub = rc[rc.elemento.isin(sel_c)].copy()
    sub["etiqueta"] = sub.elemento.map(lambda e: core.etiqueta_elemento(e, usar_oxidos))
    if modo.startswith("Razón"):
        piv = sub.pivot_table(index="grupo", columns="etiqueta", values="razon").dropna(axis=1, how="all")
        if not piv.empty:
            st.plotly_chart(heat(piv, ESC_ECON, ""), width="stretch")
    else:
        pv = sub.pivot_table(index="grupo", columns="etiqueta", values="percentil")
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=pv.columns, y=pv.index, zmin=0, zmax=100, xgap=1, ygap=1,
            colorscale=[[0, "#181B20"], [0.6, "#2C3540"], [0.85, GREEN], [1, AMBER]],
            colorbar=dict(title="Percentil"), hovertemplate="%{y}<br>%{x}: percentil %{z:.0f}<extra></extra>"
        ))
        fig.update_layout(height=max(360, 18 * len(pv) + 100), margin=dict(l=0, r=0, t=10, b=0))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="panel-hdr">Asociación Metalúrgica: Metales principales (Au, Ag, Cu, Pb, Zn) y subproductos críticos</div>', unsafe_allow_html=True)
    paga = rmin[(rmin.razon >= 1) & rmin.elemento.isin(core.HOSTS)].groupby("grupo").etiqueta_elem.agg(lambda s: ", ".join(sorted(s)))
    crit_min = rc[(rc.razon >= 1) & ~rc.elemento.isin(core.HOSTS)]
    sin_m = rc[rc.min_ppm.isna() & (rc.percentil >= 90) & ~rc.elemento.isin(core.REE_IND) & ~rc.elemento.isin(core.HOSTS)]
    a_ = crit_min.groupby("grupo").elemento.agg(lambda s: ", ".join(sorted(core.etiqueta_elemento(x, usar_oxidos) for x in s)))
    b_ = sin_m.groupby("grupo").elemento.agg(lambda s: ", ".join(sorted(core.etiqueta_elemento(x, usar_oxidos) for x in s)))
    co = res[["grupo", "estado", "n"]].set_index("grupo").join(paga.rename("Metales principales rentables")).join(a_.rename("Críticos >= ley de corte")).join(b_.rename("Críticos en P90 (sin corte)"))
    co = co.reset_index().rename(columns={"grupo": "Sitio · Tipo", "estado": "Estado"})
    st.dataframe(co, width="stretch", hide_index=True)

# ═══ 4. EVALUACIÓN AMBIENTAL ═══
with t3:
    norma = st.radio("Marco normativo de referencia", ["NOM-147-SEMARNAT (México - Uso Industrial)", "Valores de Intervención (Referencia Europea)"], horizontal=True)
    col = "razon_mx" if norma.startswith("NOM") else "razon_eu"
    a1, a2 = st.columns([3, 2])
    with a1:
        st.markdown(f'<div class="panel-hdr">Concentración media entre límite permisible ({norma})</div>', unsafe_allow_html=True)
        st.plotly_chart(heat(rtox.pivot_table(index="grupo", columns="elemento", values=col), ESC_TOX, ""), width="stretch")
    with a2:
        st.markdown('<div class="panel-hdr">Principales excedencias ambientales por unidad</div>', unsafe_allow_html=True)
        top = rtox.dropna(subset=[col]).sort_values(col, ascending=False).head(15)
        top["etiqueta"] = top.elemento + " · " + top.grupo
        fig = px.bar(top.iloc[::-1], x=col, y="etiqueta", orientation="h", color="elemento", height=480,
                     labels={col: "Veces sobre el límite permisible", "etiqueta": ""})
        fig.add_vline(x=1, line_dash="dash", line_color="#FFFFFF")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

# ═══ 5. ANÁLISIS ESTADÍSTICO ═══
with t4:
    todas = [e for e in vals.columns if e != "REE"] + ["REE"]
    e1, e2 = st.columns([1, 1])
    el = e1.selectbox("Elemento", todas, index=todas.index("Pb") if "Pb" in todas else 0, key="el_dist")
    col_by = e2.selectbox("Agrupar color por", ["tipo", "estado", "sitio"])
    dd = meta_f.assign(valor=vals.loc[meta_f.index, el])
    dd = dd[dd.valor > 0]
    etiq_el = core.etiqueta_elemento(el, usar_oxidos)
    st.markdown(f'<div class="panel-hdr">Distribución de frecuencias de {etiq_el} (mg/kg equivalente, escala logarítmica)</div>', unsafe_allow_html=True)
    fig = px.histogram(dd, x="valor", color=col_by, log_x=True, nbins=45, marginal="box", barmode="overlay",
                       opacity=0.75, height=400, labels={"valor": f"{etiq_el} (mg/kg)"})
    if el in umb_actual:
        fig.add_vline(x=core._min_ppm(el, umb_actual), line_dash="dash", line_color=GREEN, annotation_text="Ley de corte")
    if el in core.UMB_TOXICOS:
        for v, nm, colr in zip(core.UMB_TOXICOS[el], ["NOM-147", "Ref. UE"], [RED, AMBER]):
            if not np.isnan(v):
                fig.add_vline(x=v, line_dash="dot", line_color=colr, annotation_text=nm)
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="panel-hdr">Matriz de correlación de Spearman (log10)</div>', unsafe_allow_html=True)
    defecto = [e for e in ["Pb", "Zn", "Ag", "Cu", "As", "Cd", "Sb", "Fe", "Mn", "Ga", "Ge", "In", "U", "Al"] if e in vals]
    sel = st.multiselect("Variables para correlación", todas, default=defecto)
    if len(sel) >= 3:
        X = vals.loc[meta_f.index, sel]
        C = np.log10(X.where(X > 0)).rank().corr()
        fig = px.imshow(C, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", height=520, aspect="auto")
        st.plotly_chart(fig, width="stretch")

# ═══ 6. ZONAS PRIORITARIAS ═══
with t7:
    st.markdown('<div class="panel-hdr">Priorización de Zonas por Potencial Económico y Volumen</div>', unsafe_allow_html=True)
    z1, z2, z3 = st.columns(3)
    n_min = z1.slider("Muestras mínimas (n) por unidad", 1, 10, 3)
    sin_p = z2.checkbox("Excluir Fósforo (P) del ranking líder", value=True)
    vol_min = z3.number_input("Volumen mínimo requerido en terreros (m3)", 0.0, 1e6, 0.0, 50.0)
    zonas = core.proponer_zonas(rmin, res, n_min, ("P",) if sin_p else (), vol_min)
    if not zonas.empty:
        fig = px.bar(
            zonas, x="razon", y="grupo", color="categoria", facet_col="estado", facet_col_wrap=4, orientation="h",
            log_x=True, height=max(420, 28 * len(zonas) // 2 + 140),
            color_discrete_map={"Zona propuesta": GREEN, "Potencial adicional": BLUE, "Potencial con n bajo (verificar)": GREY, "Descartada por volumen": RED},
            labels={"razon": "Razón sobre corte", "grupo": ""}
        )
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(zonas[["estado", "categoria", "rank_estado", "grupo", "n", "elemento", "valor", "unidad", "razon",
                            "pct_sobre_min", "vol_m3", "nivel_tox", "declarados"]].rename(columns={
            "elemento": "Elemento líder", "razon": "Razón / Corte", "pct_sobre_min": "% Muestras >= Corte", "vol_m3": "Volumen aprox. (m3)", "nivel_tox": "Estatus Ambiental"}),
            width="stretch", hide_index=True)
    else:
        st.info("Ningún grupo cumple los criterios de selección actuales.")

# ═══ 7. CONTROL QA/QC ═══
with t5:
    st.markdown('<div class="panel-hdr">Porcentaje de Censura Analítica (% < Límite de Detección)</div>', unsafe_allow_html=True)
    idx = meta_f.index
    cols_drop = [c for c in ("REE", "PGE") if c in cens.columns]
    pc_g = (cens.loc[idx].sum() / vals.loc[idx].notna().sum() * 100).drop(cols_drop).sort_values(ascending=False)
    fig = px.bar(
        pc_g,
        color=np.where(pc_g > 20, "Descartar (>20 %)", np.where(pc_g > 13, "Advertencia (13-20 %)", "Aceptable (<=13 %)")),
        color_discrete_map={"Aceptable (<=13 %)": GREEN, "Advertencia (13-20 %)": AMBER, "Descartar (>20 %)": RED},
        height=350, labels={"value": "% < LD", "index": "", "color": "Criterio"}
    )
    fig.add_hline(y=13, line_dash="dot", line_color=AMBER)
    fig.add_hline(y=20, line_dash="dot", line_color=RED)
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="panel-hdr">Repetibilidad en Duplicados de Campo (-DPL): Diferencia Relativa Porcentual (RPD)</div>', unsafe_allow_html=True)
    pares = core.pares_dpl(meta, vals_raw, cens)
    if pares.empty:
        st.info("No se registran pares primaria-duplicado en la selección.")
    else:
        orden = pares.groupby("elemento").rpd.median().sort_values().index
        fig = px.box(pares, x="elemento", y="rpd", category_orders={"elemento": list(orden)}, height=380,
                     labels={"rpd": "RPD (%)", "elemento": ""})
        fig.add_hline(y=20, line_dash="dash", line_color=RED, annotation_text="Límite 20 %")
        st.plotly_chart(fig, width="stretch")

# ═══ 8. DATOS Y EXPORTACIÓN ═══
with t6:
    st.markdown('<div class="panel-hdr">Resumen Ejecutivo por Sitio y Tipo de Depósito</div>', unsafe_allow_html=True)
    cols = ["grupo", "estado", "municipio", "n", "pinta_lbl", "elementos_rentables", "nivel_tox", "excede_mx", "excede_eu"]
    tabla = res[cols].rename(columns={
        "pinta_lbl": "Potencial Económico", "elementos_rentables": "Elementos >= Corte",
        "nivel_tox": "Estatus Ambiental", "excede_mx": "Excede NOM-147 (mg/kg)", "excede_eu": "Excede Ref. UE (mg/kg)"
    })
    st.dataframe(tabla, width="stretch", hide_index=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        tabla.to_excel(xw, sheet_name="Resumen_Ejecutivo", index=False)
        rmin.drop(columns=["gm_u", "media_u", "p1_u", "p2_u", "maximo_u"], errors="ignore").to_excel(xw, sheet_name="Evaluacion_Economica", index=False)
        rtox.to_excel(xw, sheet_name="Evaluacion_Ambiental", index=False)
    d1, d2 = st.columns(2)
    d1.download_button("Descargar Resumen (CSV)", tabla.to_csv(index=False).encode("utf-8-sig"), "resumen_eda_sgm.csv", "text/csv")
    d2.download_button("Descargar Reporte Completo (Excel)", buf.getvalue(), "reporte_geoquimico_eulacrm.xlsx")

    with st.expander("Matriz de concentraciones medias convertidas (Verde >= Ley de Corte | Rojo < Ley de Corte)", expanded=True):
        mat = rmin.assign(col=rmin.etiqueta_elem + " (" + rmin.unidad + ")").pivot_table(index="grupo", columns="col", values="valor")
        mapa_min_col = {f"{r.etiqueta_elem} ({r.unidad})": r.minimo for r in rmin[["etiqueta_elem", "unidad", "minimo"]].drop_duplicates().itertuples()}
        sty = mat.style
        for cn in mat.columns:
            u_min = mapa_min_col.get(cn, np.nan)
            sty = sty.map(
                lambda v, u=u_min: "background-color:#262A32;color:#78808C" if pd.isna(v) else
                ("background-color:#194D28;color:#D4F7DC;font-weight:bold" if v >= u else "background-color:#4A1C1C;color:#FAD2D2"),
                subset=[cn]
            )
        st.dataframe(sty.format("{:.3g}"), width="stretch")
