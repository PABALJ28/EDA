"""Núcleo de cálculo del EDA geoquímico PAM-SGM."""
import io
import numpy as np
import pandas as pd
from pyproj import Transformer

# ───────────── Tablas de referencia y factores estequiométricos (Word de referencia) ─────────────
# Únicamente los elementos cuyo umbral mínimo minable en el documento de referencia está expresado como óxido:
# Elemento -> (Fórmula del óxido, Factor gravimétrico estequiométrico)
FACTORES_OXIDO = {
    'Al': ('Al2O3', 1.8895),   # Aluminum: 20-30% Al2O3
    'Cr': ('Cr2O3', 1.4615),   # Chromium: 20-25% Cr2O3
    'Ti': ('TiO2',  1.6685),   # Titanium: 10-15% TiO2
    'Li': ('Li2O',  2.1527),   # Lithium: 0.6-1.2% Li2O
    'W':  ('WO3',   1.2611),   # Tungsten: 0.3-0.5% WO3
    'Be': ('BeO',   2.7753),   # Beryllium: 0.1-0.3% BeO
    'Ce': ('CeO2',  1.2284),   # Cerium: 0.1-0.2% CeO2
    'U':  ('U3O8',  1.1792),   # Uranium: 0.05-0.1% U3O8
    'Th': ('ThO2',  1.1379),   # Thorium: 0.05-0.1% ThO2
    'Ta': ('Ta2O5', 1.2210),   # Tantalum: 0.02-0.05% Ta2O5
}

# Especies minerales según el documento de referencia:
# Especie -> (Rango documento, Ley de corte mínima % Fe)
ESPECIES_FE = {
    'Magnetita (20-30% Fe)': ('20-30% Fe', 20.0),
    'Goethita (30-40% Fe)':  ('30-40% Fe', 30.0),
    'Pirita (30-40% Fe)':    ('30-40% Fe', 30.0),
    'Hematita (40-50% Fe)':  ('40-50% Fe', 40.0),
    'Limonita (40-50% Fe)':  ('40-50% Fe', 40.0),
}

ESPECIES_ZN = {
    'Sulfuros (3-5% Zn)':       ('3-5% Zn',   3.0),
    'No sulfuros (10-15% Zn)':  ('10-15% Zn', 10.0),
}

ESPECIES_CU = {
    'Óxidos (0.3-0.6% Cu)':    ('0.3-0.6% Cu', 0.3),
    'Sulfuros (0.5-1.0% Cu)':  ('0.5-1.0% Cu', 0.5),
}

# Elemento -> (mínimo rentable límite inferior del documento, unidad objetivo)
UMB_MINABLES_BASE = {
    'Fe':  (20.0,  '%'),      # Según especie seleccionada (20%, 30% o 40% Fe)
    'Cr':  (20.0,  '%'),      # 20-25% Cr2O3
    'Mn':  (20.0,  '%'),      # 20-30% Mn
    'Al':  (20.0,  '%'),      # 20-30% Al2O3
    'Ti':  (10.0,  '%'),      # 10-15% TiO2
    'Zn':  (3.0,   '%'),      # 3-5% Zn (sulfuros) o 10-15% Zn (no sulfuros)
    'Pb':  (3.0,   '%'),      # 3-5% Pb
    'Ni':  (1.0,   '%'),      # 1-2% Ni
    'Li':  (0.6,   '%'),      # 0.6-1.2% Li2O
    'Cu':  (0.3,   '%'),      # 0.3-0.6% Cu (óxidos) o 0.5-1.0% Cu (sulfuros)
    'Sb':  (0.5,   '%'),      # 0.5-1.0% Sb
    'W':   (0.3,   '%'),      # 0.3-0.5% WO3
    'Be':  (0.1,   '%'),      # 0.1-0.3% BeO
    'Nb':  (0.1,   '%'),      # 0.1-0.2% Nb
    'Co':  (0.1,   '%'),      # 0.1-0.2% Co
    'Ce':  (0.1,   '%'),      # 0.1-0.2% CeO2
    'V':   (0.1,   '%'),      # 0.1-0.2% V (elemental directo, sin convertir a óxido)
    'REE': (0.1,   '%'),      # 0.1-0.2% total REE
    'P':   (700.0, 'ppm'),    # 700 ppm P (elemental directo, sin convertir a óxido)
    'U':   (0.05,  '%'),      # 0.05-0.1% U3O8
    'Th':  (0.05,  '%'),      # 0.05-0.1% ThO2
    'Nd':  (0.03,  '%'),      # 0.03-0.05% Nd
    'Ta':  (0.02,  '%'),      # 0.02-0.05% Ta2O5
    'Mo':  (0.02,  '%'),      # 0.02-0.05% Mo
    'Sc':  (0.02,  '%'),      # 0.02-0.05% Sc
    'Ag':  (50.0,  'g/t'),    # 50-100 g/t Ag
    'PGE': (2.0,   'g/t'),    # 2-10 g/t Pt+Pd+Au
    'Au':  (1.0,   'g/t'),    # 1-2 g/t Au
    'Ga':  (30.0,  'ppm'),    # Subproducto crítico (literatura técnica: 30-50 ppm)
    'Ge':  (30.0,  'ppm'),    # Subproducto crítico (literatura técnica: 30-50 ppm)
    'In':  (15.0,  'ppm'),    # Subproducto crítico (literatura técnica: 15-20 ppm)
}

UMB_MINABLES = UMB_MINABLES_BASE.copy()

# Elemento -> (NOM-147 Industrial MX, Referencia de Intervención Europea) en mg/kg
UMB_TOXICOS = {
    'As': (260.0,  76.0),
    'Pb': (800.0,  530.0),
    'Cd': (380.0,  13.0),
    'Cr': (280.0,  180.0),
    'Be': (150.0,  np.nan),
    'V':  (1000.0, 250.0),
    'Ni': (1600.0, 100.0),
    'Cu': (np.nan, 190.0),
    'Zn': (np.nan, 720.0),
}

TIERRAS_RARAS = ['Ce', 'Eu', 'La', 'Nd', 'Pr', 'Sm', 'Dy', 'Er', 'Gd', 'Ho',
                 'Lu', 'Sc', 'Tb', 'Tm', 'Y', 'Yb']
# En el documento de referencia: Platinum group elements (PGE) = 2-10 g/t Pt+Pd+Au
COMPONENTES_PGE = ['Pt', 'Pd', 'Au']

LIM_CENS_OK, LIM_CENS_MAX = 0.13, 0.20
UNIDADES = {'mg/kg', '%', 'µg/kg', 'ug/kg', 'g/t', 'ppm', 'ppb'}
FACTOR_PPM = {'%': 1e4, 'µg/kg': 1e-3, 'ug/kg': 1e-3, 'ppb': 1e-3}


def configurar_parametros(
    especie_fe='Magnetita (20-30% Fe)',
    especie_zn='Sulfuros (3-5% Zn)',
    especie_cu='Óxidos (0.3-0.6% Cu)'
):
    """Retorna diccionario UMB_MINABLES ajustado por especie mineral de Fe, Zn y Cu."""
    umb = UMB_MINABLES_BASE.copy()
    if especie_fe in ESPECIES_FE:
        umb['Fe'] = (ESPECIES_FE[especie_fe][1], '%')
    if especie_zn in ESPECIES_ZN:
        umb['Zn'] = (ESPECIES_ZN[especie_zn][1], '%')
    if especie_cu in ESPECIES_CU:
        umb['Cu'] = (ESPECIES_CU[especie_cu][1], '%')
    return umb


def _num(serie):
    """Texto -> (valor con <LD sustituido por LD/2, flag censurado, LD)."""
    s = serie.astype("string").str.strip()
    cens = s.str.startswith("<").fillna(False).astype(bool)
    num = pd.to_numeric(s.str.lstrip("<").str.strip(), errors="coerce")
    num = pd.Series(num.to_numpy(dtype=float, na_value=np.nan), index=serie.index)
    cens = cens & num.notna()
    return num.where(~cens, num / 2.0), cens, num.where(cens)


def _col_opt(df, claves):
    """Busca una columna en ambos niveles del MultiIndex usando una lista de posibles nombres."""
    if isinstance(claves, str):
        claves = [claves]
    for k in claves:
        kl = k.lower()
        for c in df.columns:
            if kl in str(c[0]).lower() or kl in str(c[1]).lower():
                return c
    return None


def _leer_hoja(xls, hoja):
    df = pd.read_excel(xls, sheet_name=hoja, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples([(str(a).strip(), str(b).strip()) for a, b in df.columns])
    return df


def cargar_base(contenido: bytes):
    """Devuelve (meta, vals_ppm, censurado, LD_ppm)."""
    xls = pd.ExcelFile(io.BytesIO(contenido))
    hojas = [h for h in xls.sheet_names if "LREE" not in h.upper() and "AVALOS" not in h.upper()]
    if not hojas:
        hojas = [h for h in xls.sheet_names if "LREE" not in h.upper()]
    if not hojas:
        hojas = list(xls.sheet_names[:1])
    for nombre_base in ("Base general", "Base General", "Base_General"):
        if nombre_base in hojas:
            hojas = [nombre_base]
            break
    raw = pd.concat([_leer_hoja(xls, h) for h in hojas], ignore_index=True)

    c_x = next((c for c in raw.columns if c[1].upper() in ("X", "ESTE", "X_UTM", "UTM_X") or c[0].upper() in ("X", "ESTE", "X_UTM")), None)
    c_y = next((c for c in raw.columns if c[1].upper() in ("Y", "NORTE", "Y_UTM", "UTM_Y") or c[0].upper() in ("Y", "NORTE", "Y_UTM")), None)

    def txt(claves, default="Sin dato"):
        col = _col_opt(raw, claves)
        if col is None:
            return pd.Series([default] * len(raw), index=raw.index, dtype="string")
        s = raw[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.astype("string").str.strip()

    def num_col(claves):
        col = _col_opt(raw, claves)
        if col is None:
            return pd.Series(np.nan, index=raw.index, dtype=float)
        s = raw[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    s_clave = txt(["Clave", "ID", "Muestra", "Codigo", "Código"], default="")
    s_clave = s_clave.where(s_clave.notna() & (s_clave != ""), pd.Series([f"M-{i+1}" for i in range(len(raw))], index=raw.index, dtype="string"))

    meta = pd.DataFrame({
        "clave": s_clave,
        "zona": num_col(["Zona", "Zona UTM", "Zone"]).ffill(),
        "x": (pd.to_numeric(raw[c_x], errors="coerce").ffill() if c_x is not None else pd.Series(np.nan, index=raw.index)),
        "y": (pd.to_numeric(raw[c_y], errors="coerce").ffill() if c_y is not None else pd.Series(np.nan, index=raw.index)),
        "estado": txt(["Estado", "Entidad"], default="General").ffill().fillna("General"),
        "municipio": txt(["Municipio", "Localidad"], default="Sin especificar").ffill().fillna("Sin especificar"),
        "sitio": txt(["Sitio", "Proyecto", "Mina", "Localidad", "Area", "Área"], default="Sitio General").ffill().fillna("Sitio General"),
        "interes": txt(["Elementos de", "Interés", "Interes", "Asociación", "Asociacion", "Target"], default="No especificado").ffill().fillna("No especificado"),
        "tipo": txt(["Tipo de muestra", "Tipo", "Depósito", "Deposito", "Material", "Matriz"], default="Muestra").ffill().fillna("Muestra"),
    })
    meta["es_dpl"] = meta.clave.str.contains("DPL", case=False).fillna(False).astype(bool)
    meta["clave_base"] = meta.clave.str.replace(r"(?i)[-_ ]*DPL$", "", regex=True)
    meta["grupo"] = meta.sitio + " · " + meta.tipo
    med = [c for c in raw.columns if c[0].lower().startswith("medida") or c[0].lower().startswith("dimensi")]
    for nom in ("L", "W", "H"):
        meta[nom] = np.nan
    for nom, cc in zip(["L", "W", "H"], med[:3]):
        meta[nom] = pd.to_numeric(raw[cc], errors="coerce")
    es_terr = meta.tipo.str.contains("Terrero", case=False, na=False).to_numpy()
    meta["terrero"] = (meta.L.notna() & es_terr).groupby(meta.sitio).cumsum().where(es_terr, 0).astype(int)
    meta["vol_m3"] = (meta.L * meta.W * meta.H).where(es_terr)
    n_terr = meta.groupby("sitio").terrero.transform("max")
    meta["grupo_fino"] = meta.grupo.where(~(es_terr & (n_terr > 1)), meta.grupo + " #" + meta.terrero.astype(str))

    lat, lon = np.full(len(meta), np.nan), np.full(len(meta), np.nan)
    for z in meta.zona.dropna().unique():
        try:
            z_int = int(z)
            if 1 <= z_int <= 60:
                m = ((meta.zona == z) & meta.x.notna() & meta.y.notna()).to_numpy()
                t = Transformer.from_crs(f"EPSG:{32600 + z_int}", "EPSG:4326", always_xy=True)
                lon[m], lat[m] = t.transform(meta.x[m].to_numpy(), meta.y[m].to_numpy())
        except Exception:
            pass
    meta["lat"], meta["lon"] = lat, lon

    vals, cens, ld, au = {}, {}, {}, []
    for c in raw.columns:
        e, u = c
        if u.split(".")[0] not in UNIDADES or u.endswith(".1"):
            continue
        f = FACTOR_PPM.get(u, 1.0)
        v, k, l = _num(raw[c])
        if e == "Au":
            au.append((v * f, k, l * f))
            continue
        vals[e], cens[e], ld[e] = v * f, k, l * f
    if au:
        v, k, l = au[0]
        for v2, k2, l2 in au[1:]:
            k, l = k.where(v.notna(), k2), l.where(v.notna(), l2)
            v = v.combine_first(v2)
        vals["Au"], cens["Au"], ld["Au"] = v, k, l
    vals, cens, ld = pd.DataFrame(vals), pd.DataFrame(cens), pd.DataFrame(ld)

    # Suma de Tierras Raras (Σ ETR)
    ree = [e for e in TIERRAS_RARAS if e in vals]
    vals["REE"] = vals[ree].fillna(0).sum(axis=1)
    cens["REE"], ld["REE"] = False, np.nan

    # Suma de Platinoides según documento de referencia: PGE = Pt + Pd + Au (2-10 g/t)
    cols_pge = [e for e in COMPONENTES_PGE if e in vals]
    if cols_pge:
        vals["PGE"] = vals[cols_pge].fillna(0).sum(axis=1)
        if "Au" in cens and len(cols_pge) == 1:
            cens["PGE"] = cens["Au"]
        else:
            cens["PGE"] = cens[cols_pge].all(axis=1)
        ld["PGE"] = np.nan

    return meta, vals, cens, ld


def aplicar_conversion_oxidos(vals, usar_oxidos=True):
    """Multiplica las columnas de elementos evaluados como óxido por su factor gravimétrico."""
    if not usar_oxidos:
        return vals.copy()
    v_out = vals.copy()
    for elem, (_, factor) in FACTORES_OXIDO.items():
        if elem in v_out.columns:
            v_out[elem] = v_out[elem] * factor
    return v_out


def _min_ppm(e, umb_dict=None):
    d = umb_dict if umb_dict is not None else UMB_MINABLES
    v, u = d[e]
    return v * 1e4 if u == "%" else v


def estadisticos(meta, vals, cens, umb_dict=None):
    """Una fila por (grupo, elemento): censura, media, media geométrica, +1σ, +2σ (log10)."""
    d_umb = umb_dict if umb_dict is not None else UMB_MINABLES
    elems = list(vals.columns)
    filas = []
    for g, idx in meta.groupby("grupo").groups.items():
        sub, csub, m0 = vals.loc[idx, elems], cens.loc[idx, elems], meta.loc[idx[0]]
        for e in elems:
            x = sub[e].dropna()
            if x.empty:
                continue
            pct = float(csub.loc[x.index, e].mean())
            lx = np.log10(x[x > 0])
            mu = lx.mean() if len(lx) else np.nan
            sd = lx.std(ddof=1) if len(lx) > 1 else 0.0
            filas.append(dict(
                grupo=g, sitio=m0.sitio, tipo=m0.tipo, estado=m0.estado, elemento=e,
                n=len(idx), n_val=len(x), pct_cens=pct,
                estatus="OK" if pct <= LIM_CENS_OK else ("Advertencia" if pct <= LIM_CENS_MAX else "Descartado"),
                media=x.mean(), gm=10 ** mu, p1=10 ** (mu + sd), p2=10 ** (mu + 2 * sd), maximo=x.max(),
                pct_sobre_min=(x >= _min_ppm(e, d_umb)).mean() if e in d_umb else np.nan))
    return pd.DataFrame(filas)


def etiqueta_elemento(e, usar_oxidos=True):
    if e == "PGE":
        return "PGE (Pt+Pd+Au)"
    if usar_oxidos and e in FACTORES_OXIDO:
        return FACTORES_OXIDO[e][0]
    return e


def razones_min(est, stat="media", umb_dict=None, usar_oxidos=True):
    d_umb = umb_dict if umb_dict is not None else UMB_MINABLES
    d = est[est.elemento.isin(d_umb)].copy()
    d["unidad"] = d.elemento.map(lambda e: d_umb[e][1])
    d["minimo"] = d.elemento.map(lambda e: d_umb[e][0])
    d["etiqueta_elem"] = d.elemento.map(lambda e: etiqueta_elemento(e, usar_oxidos))
    f = np.where(d.unidad == "%", 1e-4, 1.0)
    for c in ("media", "gm", "p1", "p2", "maximo"):
        d[c + "_u"] = d[c] * f
    d["valor"] = d[stat + "_u"]
    d["razon"] = d.valor / d.minimo
    d["razon_p1"], d["razon_p2"] = d.p1_u / d.minimo, d.p2_u / d.minimo
    d.loc[d.estatus == "Descartado", ["valor", "razon", "razon_p1", "razon_p2"]] = np.nan
    return d


def razones_tox(est):
    d = est[est.elemento.isin(UMB_TOXICOS)].copy()
    d["lim_mx"] = d.elemento.map(lambda e: UMB_TOXICOS[e][0])
    d["lim_eu"] = d.elemento.map(lambda e: UMB_TOXICOS[e][1])
    d["razon_mx"], d["razon_eu"] = d.media / d.lim_mx, d.media / d.lim_eu
    d.loc[d.estatus == "Descartado", ["razon_mx", "razon_eu"]] = np.nan
    return d


def resumen_grupos(meta, rmin, rtox):
    filas = []
    for g, idx in meta.groupby("grupo").groups.items():
        m = meta.loc[idx]
        a, t = rmin[rmin.grupo == g], rtox[rtox.grupo == g]
        rent = a[a.razon >= 1].sort_values("razon", ascending=False)
        mx = t[t.razon_mx > 1]
        eu = t[(t.razon_eu > 1) & ~(t.razon_mx > 1)]
        decl = {s.strip() for s in str(m.interes.iloc[0]).split(",")}
        conf = set(rent.elemento)
        filas.append(dict(
            grupo=g, sitio=m.sitio.iloc[0], tipo=m.tipo.iloc[0], estado=m.estado.iloc[0],
            municipio=m.municipio.iloc[0], n=len(m), lat=m.lat.mean(), lon=m.lon.mean(),
            declarados=m.interes.iloc[0],
            vol_m3=m.vol_m3.sum(min_count=1),
            pinta=len(rent) > 0,
            elementos_rentables=", ".join(f"{r.etiqueta_elem} ({r.valor:.2f} {r.unidad})" for r in rent.itertuples()),
            no_declarados=", ".join(sorted(conf - decl)),
            sin_confirmar=", ".join(sorted((decl & set(rmin.elemento.unique())) - conf)),
            razon_econ=a.razon.max() if a.razon.notna().any() else np.nan,
            elem_econ=a.loc[a.razon.idxmax(), "etiqueta_elem"] if a.razon.notna().any() else "",
            nivel_tox="Supera NOM-147 (MX)" if len(mx) else ("Supera ref. europea" if len(eu) else "Debajo de límites"),
            excede_mx=", ".join(f"{r.elemento} ({r.media:.0f})" for r in mx.itertuples()),
            excede_eu=", ".join(f"{r.elemento} ({r.media:.0f})" for r in eu.itertuples()),
            razon_tox=t.razon_mx.max() if t.razon_mx.notna().any() else np.nan,
            elem_tox=t.loc[t.razon_mx.idxmax(), "elemento"] if t.razon_mx.notna().any() else ""))
    return pd.DataFrame(filas)


def clasificar_muestras(meta, vals_eval, vals_raw, umb_dict=None):
    """Clasificación muestra por muestra."""
    d_umb = umb_dict if umb_dict is not None else UMB_MINABLES
    out = meta.copy()
    Rm = pd.concat({e: vals_eval[e] * (1e-4 if d_umb[e][1] == "%" else 1.0) / d_umb[e][0]
                    for e in d_umb if e in vals_eval}, axis=1)
    out["razon_econ"] = Rm.max(axis=1)
    out["elem_econ"] = Rm.fillna(-np.inf).idxmax(axis=1).where(out.razon_econ.notna(), "")
    Rx = pd.concat({e: vals_raw[e] / UMB_TOXICOS[e][0] for e in UMB_TOXICOS if e in vals_raw}, axis=1)
    Re = pd.concat({e: vals_raw[e] / UMB_TOXICOS[e][1] for e in UMB_TOXICOS if e in vals_raw}, axis=1)
    out["razon_mx"], out["razon_eu"] = Rx.max(axis=1), Re.max(axis=1)
    out["pinta"] = out.razon_econ >= 1
    out["nivel_tox"] = np.where(out.razon_mx > 1, "Supera NOM-147 (MX)",
                        np.where(out.razon_eu > 1, "Supera ref. europea", "Debajo de límites"))
    return out


# ───────────── Materias primas críticas ─────────────
_T, _F = True, False
CRITICOS = {
    'Al':  ('Aluminio (20-30% Al2O3)', _T, _T, _T),
    'As':  ('Arsénico', _T, _F, _T),
    'Ba':  ('Barita (4.2-4.5 g/cm3)', _T, _F, _T),
    'Be':  ('Berilio (0.1-0.3% BeO)', _T, _F, _T),
    'Bi':  ('Bismuto', _T, _F, _T),
    'Co':  ('Cobalto (0.1-0.2% Co)', _T, _T, _T),
    'Cr':  ('Cromo (20-25% Cr2O3)', _F, _F, _T),
    'Cu':  ('Cobre (0.3% óxidos / 0.5% sulfuros)', _T, _T, _T),
    'Ga':  ('Galio (subproducto 30 ppm)', _T, _T, _T),
    'Ge':  ('Germanio (subproducto 30 ppm)', _T, _T, _T),
    'In':  ('Indio (subproducto 15 ppm)', _F, _F, _T),
    'Li':  ('Litio (0.6-1.2% Li2O)', _T, _T, _T),
    'Mg':  ('Magnesio', _T, _T, _T),
    'Mn':  ('Manganeso (20-30% Mn)', _T, _T, _T),
    'Nb':  ('Niobio (0.1-0.2% Nb)', _T, _F, _T),
    'Ni':  ('Níquel (1-2% Ni)', _T, _T, _T),
    'P':   ('Fósforo (700 ppm P)', _T, _F, _T),
    'Pb':  ('Plomo (3-5% Pb)', _F, _F, _T),
    'PGE': ('Platinoides (2-10 g/t Pt+Pd+Au)', _T, _T, _T),
    'Sb':  ('Antimonio (0.5-1.0% Sb)', _T, _F, _T),
    'Sc':  ('Escandio (0.02-0.05% Sc)', _T, _F, _T),
    'Sn':  ('Estaño', _F, _F, _T),
    'Sr':  ('Estroncio', _T, _F, _F),
    'Ta':  ('Tántalo (0.02-0.05% Ta2O5)', _T, _F, _T),
    'Te':  ('Telurio', _F, _F, _T),
    'Th':  ('Torio (0.05-0.1% ThO2)', _F, _F, _F),
    'Ti':  ('Titanio (10-15% TiO2)', _T, _T, _T),
    'U':   ('Uranio (0.05-0.1% U3O8)', _F, _F, _T),
    'V':   ('Vanadio (0.1-0.2% V)', _T, _F, _T),
    'W':   ('Tungsteno (0.3-0.5% WO3)', _T, _T, _T),
    'Zn':  ('Zinc (3% sulfuros / 10% no sulfuros)', _F, _F, _T),
    'Ag':  ('Plata (50-100 g/t Ag)', _F, _F, _T),
    'REE': ('Tierras raras (0.1-0.2% total REE)', _T, _T, _T),
    'Ce':  ('Cerio (0.1-0.2% CeO2)', _T, _T, _T),
    'La':  ('Lantano (LREE)', _T, _F, _T),
    'Nd':  ('Neodimio (0.03-0.05% Nd)', _T, _T, _T),
    'Pr':  ('Praseodimio (LREE)', _T, _T, _T),
    'Sm':  ('Samario (LREE)', _T, _T, _T),
    'Dy':  ('Disprosio (HREE)', _T, _T, _T),
    'Tb':  ('Terbio (HREE)', _T, _T, _T),
    'Gd':  ('Gadolinio (HREE)', _T, _T, _T),
    'Er':  ('Erbio (HREE)', _T, _F, _T),
    'Eu':  ('Europio (HREE)', _T, _F, _T),
    'Ho':  ('Holmio (HREE)', _T, _F, _T),
    'Lu':  ('Lutecio (HREE)', _T, _F, _T),
    'Tm':  ('Tulio (HREE)', _T, _F, _T),
    'Y':   ('Itrio (HREE)', _T, _F, _T),
    'Yb':  ('Iterbio (HREE)', _T, _F, _T),
}
PRIORITARIOS_PROYECTO = {'Co', 'Cr', 'Ga', 'Ge', 'In', 'Ti', 'V', 'Ce', 'Nd', 'As', 'REE', 'U', 'Th', 'PGE', 'Li', 'Nb', 'Ta'}
HOSTS = ['Au', 'Ag', 'Cu', 'Pb', 'Zn']
REE_IND = [e for e in TIERRAS_RARAS if e not in ('Sc', 'Ce', 'Nd')]
CRITICOS_SIN_DATO = ['Li (0.6% Li2O)', 'Nb (0.1% Nb)', 'Ta (0.02% Ta2O5)', 'Pt y Pd (en PGE)', 'Barita (4.2-4.5 g/cm3, densidad)']


def minimos_ppm(umb_dict=None, extra=None):
    """Mínimos minables en ppm."""
    d_umb = umb_dict if umb_dict is not None else UMB_MINABLES
    out = {e: _min_ppm(e, d_umb) for e in d_umb}
    for e, (v, u) in (extra or {}).items():
        if v and v > 0:
            out[e] = v * 1e4 if u == "%" else v
    return out


def razones_criticos(est, minimos):
    """Por grupo y elemento crítico: promedio ÷ mínimo y percentil dentro de la base."""
    d = est[est.elemento.isin(CRITICOS)].copy()
    d["min_ppm"] = d.elemento.map(minimos)
    d["razon"] = d.media / d.min_ppm
    d.loc[d.estatus == "Descartado", "razon"] = np.nan
    ok = d.estatus != "Descartado"
    d["percentil"] = d.media.where(ok).groupby(d.elemento).rank(pct=True) * 100
    return d


# ───────────── QA/QC ─────────────
def pares_dpl(meta, vals, cens):
    """RPD (%) entre muestra primaria y su duplicado de campo (-DPL), solo valores no censurados."""
    prim = meta.loc[~meta.es_dpl, ["clave", "sitio"]].reset_index().rename(columns={"index": "i_p"})
    dpl = meta.loc[meta.es_dpl, ["clave_base", "sitio"]].reset_index().rename(
        columns={"index": "i_d", "clave_base": "clave"})
    par = dpl.merge(prim, on=["clave", "sitio"], how="inner")
    filas = []
    for e in vals.columns:
        if e in ("REE", "PGE"):
            continue
        a, b = vals.loc[par.i_p, e].to_numpy(), vals.loc[par.i_d, e].to_numpy()
        ok = (~np.isnan(a)) & (~np.isnan(b)) & (~cens.loc[par.i_p, e].to_numpy()) & (~cens.loc[par.i_d, e].to_numpy()) & (a + b > 0)
        rpd = np.abs(a - b) / ((a + b) / 2) * 100
        filas += [dict(clave=c, sitio=s, elemento=e, primaria=x, duplicado=y, rpd=r)
                  for c, s, x, y, r in zip(par.clave[ok], par.sitio[ok], a[ok], b[ok], rpd[ok])]
    return pd.DataFrame(filas)


def diagnostico(meta):
    m = meta[~meta.es_dpl].copy()
    med = m.groupby("sitio")[["lat", "lon"]].transform("median")
    m["dist_km"] = np.hypot((m.lat - med.lat) * 111.0, (m.lon - med.lon) * 111.0 * np.cos(np.radians(med.lat)))
    atip = m.loc[m.dist_km > 5, ["clave", "sitio", "estado", "x", "y", "dist_km"]].round(1)
    dup = m.loc[m.clave.duplicated(keep=False), ["clave", "sitio", "estado"]].sort_values("clave")
    chicos = m.groupby("grupo").size().loc[lambda s: s < 3].rename("n").reset_index()
    return atip, dup, chicos


def proponer_zonas(rmin, res, n_min=3, excluir=("P",), vol_min=0.0):
    """Ordena por estado los grupos con potencial."""
    d = rmin[~rmin.elemento.isin(excluir) & (rmin.razon >= 1)]
    if d.empty:
        return pd.DataFrame(columns=["estado", "categoria", "rank_estado", "grupo", "n", "elemento", "valor", "unidad", "razon", "pct_sobre_min", "vol_m3", "nivel_tox", "declarados", "descartada", "confiable"])
    mejor = (d.sort_values("razon", ascending=False).groupby("grupo").first().reset_index()
             [["grupo", "elemento", "razon", "valor", "unidad", "pct_sobre_min"]])
    z = res.drop(columns=["elem_econ", "razon_econ"]).merge(mejor, on="grupo")
    z["vol_ok"] = np.where(z.vol_m3.isna(), "sin dato", np.where(z.vol_m3 >= vol_min, "sí", "no"))
    z["descartada"] = z.vol_ok == "no"
    z["confiable"] = z.n >= n_min
    z = z.sort_values(["estado", "descartada", "confiable", "razon"], ascending=[True, True, False, False])
    z["rank_estado"] = z.groupby("estado").cumcount() + 1
    z["categoria"] = np.where(z.descartada, "Descartada por volumen",
                      np.where((z.rank_estado == 1) & z.confiable, "Zona propuesta",
                      np.where(z.confiable, "Potencial adicional", "Potencial con n bajo (verificar)")))
    return z
