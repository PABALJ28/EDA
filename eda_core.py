"""Núcleo de cálculo universal del EDA geoquímico PAM-SGM."""
import io
import re
import numpy as np
import pandas as pd
from pyproj import Transformer

# ───────────── Tablas de referencia y factores estequiométricos (Word de referencia) ─────────────
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

# Diccionario inverso para bases de datos que ya reportan óxidos desde laboratorio -> convierte a elemento base en ppm
OXIDOS_LAB_A_ELEMENTO = {
    'Al2O3': ('Al', 1.0 / 1.8895),
    'Cr2O3': ('Cr', 1.0 / 1.4615),
    'TiO2':  ('Ti', 1.0 / 1.6685),
    'Li2O':  ('Li', 1.0 / 2.1527),
    'WO3':   ('W',  1.0 / 1.2611),
    'BeO':   ('Be', 1.0 / 2.7753),
    'CeO2':  ('Ce', 1.0 / 1.2284),
    'U3O8':  ('U',  1.0 / 1.1792),
    'ThO2':  ('Th', 1.0 / 1.1379),
    'Ta2O5': ('Ta', 1.0 / 1.2210),
    'Fe2O3': ('Fe', 1.0 / 1.4297),
    'FeO':   ('Fe', 1.0 / 1.2865),
    'MnO':   ('Mn', 1.0 / 1.2912),
    'MgO':   ('Mg', 1.0 / 1.6583),
    'CaO':   ('Ca', 1.0 / 1.3992),
    'Na2O':  ('Na', 1.0 / 1.3480),
    'K2O':   ('K',  1.0 / 1.2046),
    'P2O5':  ('P',  1.0 / 2.2914),
    'V2O5':  ('V',  1.0 / 1.7852),
    'SiO2':  ('Si', 1.0 / 2.1393),
    'BaO':   ('Ba', 1.0 / 1.1165),
    'SrO':   ('Sr', 1.0 / 1.1826),
    'ZrO2':  ('Zr', 1.0 / 1.3508),
    'Nb2O5': ('Nb', 1.0 / 1.4305),
    'SnO2':  ('Sn', 1.0 / 1.2696),
    'Sb2O3': ('Sb', 1.0 / 1.1971),
    'Co3O4': ('Co', 1.0 / 1.3620),
    'NiO':   ('Ni', 1.0 / 1.2725),
    'CuO':   ('Cu', 1.0 / 1.2518),
    'ZnO':   ('Zn', 1.0 / 1.2448),
    'PbO':   ('Pb', 1.0 / 1.0772),
}

ELEMENTOS_TABLA_PERIODICA = {
    'Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs',
    'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
    'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U'
}

NOMBRES_ELEMENTOS = {
    'oro': 'Au', 'gold': 'Au', 'plata': 'Ag', 'silver': 'Ag', 'cobre': 'Cu', 'copper': 'Cu',
    'plomo': 'Pb', 'lead': 'Pb', 'zinc': 'Zn', 'cinc': 'Zn', 'hierro': 'Fe', 'iron': 'Fe',
    'aluminio': 'Al', 'aluminum': 'Al', 'aluminium': 'Al', 'arsenico': 'As', 'arsénico': 'As', 'arsenic': 'As',
    'antimonio': 'Sb', 'antimony': 'Sb', 'bario': 'Ba', 'barium': 'Ba', 'berilio': 'Be', 'beryllium': 'Be',
    'bismuto': 'Bi', 'bismuth': 'Bi', 'cadmio': 'Cd', 'cadmium': 'Cd', 'calcio': 'Ca', 'calcium': 'Ca',
    'cerio': 'Ce', 'cerium': 'Ce', 'cobalto': 'Co', 'cobalt': 'Co', 'cromo': 'Cr', 'chromium': 'Cr',
    'escandio': 'Sc', 'scandium': 'Sc', 'estano': 'Sn', 'estaño': 'Sn', 'tin': 'Sn',
    'estroncio': 'Sr', 'strontium': 'Sr', 'fosforo': 'P', 'fósforo': 'P', 'phosphorus': 'P',
    'galio': 'Ga', 'gallium': 'Ga', 'germanio': 'Ge', 'germanium': 'Ge', 'indio': 'In', 'indium': 'In',
    'litio': 'Li', 'lithium': 'Li', 'magnesio': 'Mg', 'magnesium': 'Mg', 'manganeso': 'Mn', 'manganese': 'Mn',
    'mercurio': 'Hg', 'mercury': 'Hg', 'molibdeno': 'Mo', 'molybdenum': 'Mo', 'neodimio': 'Nd', 'neodymium': 'Nd',
    'niobio': 'Nb', 'niobium': 'Nb', 'niquel': 'Ni', 'níquel': 'Ni', 'nickel': 'Ni',
    'platino': 'Pt', 'platinum': 'Pt', 'paladio': 'Pd', 'palladium': 'Pd', 'potasio': 'K', 'potassium': 'K',
    'selenio': 'Se', 'selenium': 'Se', 'silicio': 'Si', 'silicon': 'Si', 'sodio': 'Na', 'sodium': 'Na',
    'talio': 'Tl', 'thallium': 'Tl', 'tantalo': 'Ta', 'tántalo': 'Ta', 'tantalum': 'Ta',
    'telurio': 'Te', 'tellurium': 'Te', 'titanio': 'Ti', 'titanium': 'Ti', 'torio': 'Th', 'thorium': 'Th',
    'tungsteno': 'W', 'wolframio': 'W', 'tungsten': 'W', 'uranio': 'U', 'uranium': 'U',
    'vanadio': 'V', 'vanadium': 'V', 'itrio': 'Y', 'yttrium': 'Y', 'lantano': 'La', 'lanthanum': 'La',
    'praseodimio': 'Pr', 'samario': 'Sm', 'europio': 'Eu', 'gadolinio': 'Gd', 'terbio': 'Tb',
    'disprosio': 'Dy', 'holmio': 'Ho', 'erbio': 'Er', 'tulio': 'Tm', 'iterbio': 'Yb', 'lutecio': 'Lu',
    'circonio': 'Zr', 'zirconio': 'Zr', 'zirconium': 'Zr', 'hafnio': 'Hf', 'rubidio': 'Rb', 'cesio': 'Cs',
}

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

UMB_MINABLES_BASE = {
    'Fe':  (20.0,  '%'),
    'Cr':  (20.0,  '%'),
    'Mn':  (20.0,  '%'),
    'Al':  (20.0,  '%'),
    'Ti':  (10.0,  '%'),
    'Zn':  (3.0,   '%'),
    'Pb':  (3.0,   '%'),
    'Ni':  (1.0,   '%'),
    'Li':  (0.6,   '%'),
    'Cu':  (0.3,   '%'),
    'Sb':  (0.5,   '%'),
    'W':   (0.3,   '%'),
    'Be':  (0.1,   '%'),
    'Nb':  (0.1,   '%'),
    'Co':  (0.1,   '%'),
    'Ce':  (0.1,   '%'),
    'V':   (0.1,   '%'),
    'REE': (0.1,   '%'),
    'P':   (700.0, 'ppm'),
    'U':   (0.05,  '%'),
    'Th':  (0.05,  '%'),
    'Nd':  (0.03,  '%'),
    'Ta':  (0.02,  '%'),
    'Mo':  (0.02,  '%'),
    'Sc':  (0.02,  '%'),
    'Ag':  (50.0,  'g/t'),
    'PGE': (2.0,   'g/t'),
    'Au':  (1.0,   'g/t'),
    'Ga':  (30.0,  'ppm'),
    'Ge':  (30.0,  'ppm'),
    'In':  (15.0,  'ppm'),
}

UMB_MINABLES = UMB_MINABLES_BASE.copy()

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
COMPONENTES_PGE = ['Pt', 'Pd', 'Au']

LIM_CENS_OK, LIM_CENS_MAX = 0.13, 0.20
UNIDADES = {'mg/kg', '%', 'wt%', 'µg/kg', 'ug/kg', 'g/t', 'gpt', 'ppm', 'ppb'}
FACTOR_PPM = {'%': 1e4, 'wt%': 1e4, 'µg/kg': 1e-3, 'ug/kg': 1e-3, 'ppb': 1e-3, 'ppm': 1.0, 'mg/kg': 1.0, 'g/t': 1.0, 'gpt': 1.0}
PATRON_UNIDAD = re.compile(r'(?i)(?:^|[\s_\-\(\[/])(wt\s*%|%|ppm|mg/kg|g/t|gpt|ppb|ug/kg|µg/kg|mcg/kg|pct)(?:$|[\s_\-\)\]\.])')


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
    """Convierte serie con texto (<0.01, ND, BLD, <LD, comas decimales) -> (valor con <LD sustituido por LD/2, flag censurado, LD)."""
    if isinstance(serie, pd.DataFrame):
        serie = serie.iloc[:, 0]
    s = serie.astype("string").str.strip()
    s = s.str.replace(r'^([<>]?\s*\d+),(\d+)$', r'\1.\2', regex=True)
    es_nd_texto = s.str.upper().isin(["ND", "N.D.", "BLD", "<LD", "<DL", "BELOW DETECTION", "INF", "N/D"]).fillna(False)
    cens_menor = s.str.startswith("<").fillna(False)
    cens = (cens_menor | es_nd_texto).astype(bool)

    limpio = s.str.lstrip("<").str.replace(",", "", regex=False).str.strip()
    num = pd.to_numeric(limpio, errors="coerce")
    num = pd.Series(num.to_numpy(dtype=float, na_value=np.nan), index=serie.index)

    if es_nd_texto.any():
        pos = num[num > 0]
        ld_est = float(pos.min()) if not pos.empty else np.nan
        num = num.where(~es_nd_texto, ld_est)

    cens = cens & num.notna()
    return num.where(~cens, num / 2.0), cens, num.where(cens)


def _detectar_unidad(texto):
    t = str(texto).strip()
    if not t or t.lower().startswith("unnamed"):
        return None
    tl = t.lower().split('.')[0].strip()
    if tl in ('%', 'wt%', 'wt %', 'pct', 'porcentaje'):
        return '%'
    if tl in ('ppm', 'mg/kg', 'g/t', 'gpt', 'ug/g', 'µg/g'):
        return 'ppm'
    if tl in ('ppb', 'ug/kg', 'µg/kg', 'mcg/kg', 'ng/g'):
        return 'ppb'
    m = PATRON_UNIDAD.search(t)
    if m:
        u = m.group(1).lower().replace(' ', '')
        if u in ('%', 'wt%', 'pct'):
            return '%'
        if u in ('ppb', 'ug/kg', 'µg/kg', 'mcg/kg'):
            return 'ppb'
        return 'ppm'
    return None


def _parsear_columna_geoquimica(c0, c1=""):
    """
    Analiza cualquier encabezado de 1 o 2 niveles y devuelve:
    (elemento_base, unidad_detectada, factor_oxido_a_elemento) o None si es metadato.
    """
    s0 = str(c0).strip() if c0 is not None else ""
    s1 = str(c1).strip() if c1 is not None else ""
    if (not s0 or s0.lower().startswith("unnamed")) and (not s1 or s1.lower().startswith("unnamed")):
        return None

    if re.search(r'\.\d+$', s1) and _detectar_unidad(s1.split('.')[0]) and "au" not in s0.lower():
        return None

    u = _detectar_unidad(s1) or _detectar_unidad(s0)

    palabras_meta = {
        'clave', 'id', 'muestra', 'sample', 'codigo', 'código', 'estado', 'entidad', 'state',
        'municipio', 'localidad', 'sitio', 'proyecto', 'mina', 'area', 'área', 'tipo',
        'deposito', 'depósito', 'material', 'matriz', 'zona', 'zone', 'utm', 'este', 'norte',
        'easting', 'northing', 'latitud', 'longitud', 'latitude', 'longitude', 'lat', 'lon',
        'medidas', 'dimensiones', 'largo', 'ancho', 'alto', 'volumen', 'peso', 'weight',
        'fecha', 'date', 'observaciones', 'comentarios', 'notas', 'interes', 'interés', 'elementos'
    }

    for cand in (s0, s1):
        if not cand or cand.lower().startswith("unnamed"):
            continue
        limpio = re.sub(r'\.\d+$', '', cand).strip()
        if limpio.lower() in palabras_meta:
            return None
        sin_uni = re.sub(r'(?i)[\(\[\s_\-/]+(wt\s*%|%|ppm|mg/kg|g/t|gpt|ppb|ug/kg|µg/kg|mcg/kg|pct|porcentaje)[\)\]\s]*$', '', limpio).strip()
        tokens = [sin_uni] + [t for t in re.split(r'[\s_\-\(\)/]+', sin_uni) if t]

        for tok in tokens:
            if tok.lower() in palabras_meta:
                break
            for ox_key, (el_base, f_ox_a_el) in OXIDOS_LAB_A_ELEMENTO.items():
                if tok.lower() == ox_key.lower():
                    return (el_base, u or '%', f_ox_a_el)
            if len(tok) <= 2 and tok.isalpha():
                tok_cap = tok[0].upper() + tok[1:].lower()
                if tok_cap in ELEMENTOS_TABLA_PERIODICA:
                    contexto = (s0 + " " + s1).lower()
                    if any(pm in contexto for pm in ('coordenada', 'utm', 'medida', 'dimensi', 'zona', 'fecha', 'clave', 'sitio', 'tipo')):
                        continue
                    if tok_cap == 'Y' and u is None and len(tokens) == 1 and s1.upper() == 'Y':
                        continue
                    return (tok_cap, u, 1.0)
            if tok.lower() in NOMBRES_ELEMENTOS:
                return (NOMBRES_ELEMENTOS[tok.lower()], u, 1.0)
    return None


def _es_fila_subencabezado(fila_vals):
    """Detecta si una fila corresponde a un segundo nivel de encabezado (unidades mg/kg, %, g/t, X, Y, L, W, H)."""
    textos = [str(x).strip() for x in fila_vals if pd.notna(x) and str(x).strip() != "" and not str(x).lower().startswith("unnamed")]
    if not textos:
        return False
    sub_conocidos = {'mg/kg', '%', 'wt%', 'µg/kg', 'ug/kg', 'g/t', 'gpt', 'ppm', 'ppb', 'x', 'y', 'z', 'l', 'w', 'h', 'este', 'norte', 'lat', 'lon'}
    aciertos = sum(1 for t in textos if t.lower().split('.')[0] in sub_conocidos or _detectar_unidad(t) is not None)
    nums_datos = sum(1 for t in textos if re.match(r'^[<>]?\s*-?\d+([\.,]\d+)?$', t))
    return (aciertos >= 2) and (aciertos >= nums_datos)


def _puntaje_fila_encabezado(fila_vals):
    """Calcula cuántos elementos químicos o columnas de metadatos clave contiene una fila para hallar dónde inicia la tabla."""
    textos = [str(x).strip() for x in fila_vals if pd.notna(x) and str(x).strip() != ""]
    if len(textos) < 2:
        return 0
    pts = 0
    for t in textos:
        if _parsear_columna_geoquimica(t, "") is not None:
            pts += 2
        elif any(k in t.lower() for k in ('clave', 'muestra', 'sample', 'id', 'sitio', 'estado', 'tipo', 'coordenada', 'utm', 'lat', 'lon')):
            pts += 2
    return pts


def _normalizar_dataframe_universal(df_sin_header):
    """Localiza automáticamente la fila de encabezado y normaliza tablas de 1 o 2 niveles de encabezado."""
    n_insp = min(15, len(df_sin_header))
    mejor_fila, mejor_pts = 0, -1
    for i in range(n_insp):
        pts = _puntaje_fila_encabezado(df_sin_header.iloc[i].values)
        if pts > mejor_pts:
            mejor_pts = pts
            mejor_fila = i

    h0 = df_sin_header.iloc[mejor_fila].values
    tiene_2_niveles = False
    if mejor_fila + 1 < len(df_sin_header):
        h1_cand = df_sin_header.iloc[mejor_fila + 1].values
        if _es_fila_subencabezado(h1_cand):
            tiene_2_niveles = True

    if tiene_2_niveles:
        raw_h0 = pd.Series(h0).ffill().fillna("")
        raw_h1 = pd.Series(h1_cand).fillna("")
        datos = df_sin_header.iloc[mejor_fila + 2:].copy().reset_index(drop=True)
        tuplas = [(str(a).strip(), str(b).strip()) for a, b in zip(raw_h0, raw_h1)]
    else:
        raw_h0 = pd.Series(h0).fillna("")
        datos = df_sin_header.iloc[mejor_fila + 1:].copy().reset_index(drop=True)
        tuplas = [(str(a).strip(), "") for a in raw_h0]

    vistos = {}
    tuplas_unicas = []
    for a, b in tuplas:
        clave_t = (a, b)
        if clave_t in vistos and a != "Au":
            vistos[clave_t] += 1
            b_nuevo = f"{b}.{vistos[clave_t]}" if b else f"dup.{vistos[clave_t]}"
            tuplas_unicas.append((a, b_nuevo))
        else:
            vistos[clave_t] = 0
            tuplas_unicas.append(clave_t)

    datos.columns = pd.MultiIndex.from_tuples(tuplas_unicas)
    datos = datos.dropna(how="all").reset_index(drop=True)
    return datos


def _leer_archivo_universal(contenido: bytes):
    """Soporta archivos Excel (.xlsx, .xls) y archivos de texto (.csv, .txt, .tsv) con cualquier estructura de hojas."""
    try:
        xls = pd.ExcelFile(io.BytesIO(contenido))
        hojas = [h for h in xls.sheet_names if "LREE" not in h.upper() and "AVALOS" not in h.upper()]
        if not hojas:
            hojas = list(xls.sheet_names)
        for pref in ("Base general", "Base General", "Base_General", "Datos", "Geoquimica", "Geoquímica", "Resultados"):
            for h in hojas:
                if h.strip().lower() == pref.lower():
                    hojas = [h]
                    break
            if len(hojas) == 1:
                break

        dfs = []
        for h in hojas:
            df_raw = pd.read_excel(xls, sheet_name=h, header=None)
            if not df_raw.empty and len(df_raw) >= 2:
                df_norm = _normalizar_dataframe_universal(df_raw)
                tiene_geo = any(_parsear_columna_geoquimica(c[0], c[1]) is not None for c in df_norm.columns)
                if tiene_geo:
                    dfs.append(df_norm)
        if not dfs:
            df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)
            dfs = [_normalizar_dataframe_universal(df_raw)]
        return pd.concat(dfs, ignore_index=True)
    except Exception:
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                texto = contenido.decode(enc)
                df_raw = pd.read_csv(io.StringIO(texto), sep=None, engine="python", header=None)
                return _normalizar_dataframe_universal(df_raw)
            except Exception:
                continue
        raise ValueError("No se pudo leer el archivo. Asegúrese de subir un archivo Excel (.xlsx) o CSV válido.")


def _col_opt(df, claves, exacto=False):
    """Busca una columna en ambos niveles del MultiIndex usando una lista de posibles nombres."""
    if isinstance(claves, str):
        claves = [claves]
    for k in claves:
        kl = k.lower()
        for c in df.columns:
            if _parsear_columna_geoquimica(c[0], c[1]) is not None and kl not in ('x', 'y', 'w', 'h'):
                continue
            s0, s1 = str(c[0]).lower().strip(), str(c[1]).lower().strip()
            if exacto:
                if s0 == kl or s1 == kl:
                    return c
            else:
                if kl in s0 or kl in s1:
                    return c
    return None


def cargar_base(contenido: bytes):
    """
    Cargador universal de bases de datos geoquímicas:
    - Detecta encabezados de 1 o 2 filas en cualquier hoja de Excel o CSV.
    - Reconoce elementos puros (ppm, mg/kg, g/t, ppb, %) y óxidos de laboratorio (Al2O3, Fe2O3, TiO2, etc.).
    - Soporta coordenadas UTM (con o sin columna Zona) o Geográficas (Latitud / Longitud), o bases sin coordenadas.
    - Funciona aunque no existan columnas de Estado, Municipio, Sitio, Tipo de muestra o Elementos de interés.
    """
    raw = _leer_archivo_universal(contenido)

    def txt(claves, default="Sin dato"):
        col = _col_opt(raw, claves)
        if col is None:
            return pd.Series([default] * len(raw), index=raw.index, dtype="string")
        s = raw[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.astype("string").str.strip()

    def num_col(claves, exacto=False):
        col = _col_opt(raw, claves, exacto=exacto)
        if col is None:
            return pd.Series(np.nan, index=raw.index, dtype=float)
        s = raw[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    s_clave = txt(["Clave", "ID_Muestra", "Muestra", "Sample_ID", "Sample", "ID", "Codigo", "Código", "Folio"], default="")
    s_clave = s_clave.where(s_clave.notna() & (s_clave != "") & (s_clave != "nan"),
                            pd.Series([f"M-{i+1:03d}" for i in range(len(raw))], index=raw.index, dtype="string"))

    # Coordenadas UTM o Geográficas
    x_coord = num_col(["X", "Este", "Easting", "X_UTM", "UTM_X", "Coord_X", "Longitud", "Longitude", "Lon", "Long"], exacto=True)
    if x_coord.isna().all():
        x_coord = num_col(["Este", "Easting", "X_UTM", "UTM_X", "Coord_X", "Longitud", "Longitude", "Lon"])
    y_coord = num_col(["Y", "Norte", "Northing", "Y_UTM", "UTM_Y", "Coord_Y", "Latitud", "Latitude", "Lat"], exacto=True)
    if y_coord.isna().all():
        y_coord = num_col(["Norte", "Northing", "Y_UTM", "UTM_Y", "Coord_Y", "Latitud", "Latitude", "Lat"])

    zona_col = num_col(["Zona", "Zona UTM", "Zone", "UTM_Zone"]).ffill()

    s_estado = txt(["Estado", "Entidad", "State", "Provincia", "Departamento", "Region", "Región"], default="General").ffill().fillna("General")
    s_muni = txt(["Municipio", "Distrito", "Condado", "County", "Localidad"], default="Sin especificar").ffill().fillna("Sin especificar")
    s_sitio = txt(["Sitio", "Proyecto", "Project", "Mina", "Mine", "Yacimiento", "Localidad", "Area", "Área", "Zona de estudio", "Lugar"], default="")
    if (s_sitio == "").all() or s_sitio.isna().all():
        if not (s_muni == "Sin especificar").all():
            s_sitio = s_muni.copy()
        elif not (s_estado == "General").all():
            s_sitio = s_estado.copy()
        else:
            s_sitio = pd.Series(["Sitio Único"] * len(raw), index=raw.index, dtype="string")
    else:
        s_sitio = s_sitio.replace("", np.nan).ffill().fillna("Sitio General")

    s_tipo = txt(["Tipo de muestra", "Tipo_Muestra", "Sample_Type", "Tipo", "Depósito", "Deposito", "Material", "Matriz", "Litología", "Litologia", "Roca"], default="Muestra").ffill().fillna("Muestra")
    s_interes = txt(["Elementos de", "Interés", "Interes", "Asociación", "Asociacion", "Target", "Commodity", "Commodities"], default="No especificado").ffill().fillna("No especificado")

    meta = pd.DataFrame({
        "clave": s_clave,
        "zona": zona_col,
        "x": x_coord.ffill(),
        "y": y_coord.ffill(),
        "estado": s_estado,
        "municipio": s_muni,
        "sitio": s_sitio,
        "interes": s_interes,
        "tipo": s_tipo,
    })
    meta["es_dpl"] = meta.clave.str.contains("DPL|DUP", case=False, regex=True).fillna(False).astype(bool)
    meta["clave_base"] = meta.clave.str.replace(r"(?i)[-_ ]*(DPL|DUP\d*)$", "", regex=True)
    meta["grupo"] = meta.sitio + " · " + meta.tipo

    # Medidas de terreros (opcional)
    med = [c for c in raw.columns if c[0].lower().startswith("medida") or c[0].lower().startswith("dimensi")]
    for nom in ("L", "W", "H"):
        meta[nom] = np.nan
    for nom, cc in zip(["L", "W", "H"], med[:3]):
        meta[nom] = pd.to_numeric(raw[cc], errors="coerce")
    es_terr = meta.tipo.str.contains("Terrero|Jal|Deposito|Depósito|Desmonte|Escoria", case=False, na=False).to_numpy()
    meta["terrero"] = (meta.L.notna() & es_terr).groupby(meta.sitio).cumsum().where(es_terr, 0).astype(int)
    meta["vol_m3"] = (meta.L * meta.W * meta.H).where(meta.L.notna() & meta.W.notna() & meta.H.notna())
    n_terr = meta.groupby("sitio").terrero.transform("max")
    meta["grupo_fino"] = meta.grupo.where(~(es_terr & (n_terr > 1)), meta.grupo + " #" + meta.terrero.astype(str))

    # Conversión inteligente de coordenadas (soporta Grados Decimales Lat/Lon directos y UTM con o sin columna Zona)
    lat, lon = np.full(len(meta), np.nan), np.full(len(meta), np.nan)
    xv, yv = meta.x.to_numpy(dtype=float), meta.y.to_numpy(dtype=float)
    es_geo = (np.abs(xv) <= 180.0) & (np.abs(yv) <= 90.0) & (~np.isnan(xv)) & (~np.isnan(yv))
    lon[es_geo] = xv[es_geo]
    lat[es_geo] = yv[es_geo]
    es_geo_inv = (xv >= -90.0) & (xv <= 90.0) & (yv >= -180.0) & (yv < -30.0)
    lat[es_geo_inv] = xv[es_geo_inv]
    lon[es_geo_inv] = yv[es_geo_inv]

    es_utm = (~es_geo) & (~es_geo_inv) & (xv > 100000) & (yv > 100000)
    if es_utm.any():
        zonas_uso = meta.zona.fillna(13).copy()  # Zona 13N por defecto si la base no trae columna Zona
        for z in zonas_uso[es_utm].dropna().unique():
            try:
                z_int = int(z)
                if 1 <= z_int <= 60:
                    m = es_utm & (zonas_uso == z).to_numpy()
                    t = Transformer.from_crs(f"EPSG:{32600 + z_int}", "EPSG:4326", always_xy=True)
                    lon[m], lat[m] = t.transform(xv[m], yv[m])
            except Exception:
                pass
    meta["lat"], meta["lon"] = lat, lon

    # Extracción universal de columnas geoquímicas
    vals, cens, ld, au = {}, {}, {}, []
    for c in raw.columns:
        info_col = _parsear_columna_geoquimica(c[0], c[1])
        if info_col is None:
            continue
        e, u_det, f_ox_a_el = info_col
        v, k, l = _num(raw[c])
        if v.notna().sum() == 0:
            continue

        if u_det is None:
            u_det = 'ppm'
        f_uni = FACTOR_PPM.get(u_det, 1.0)
        f_total = f_uni * f_ox_a_el

        if e == "Au":
            au.append((v * f_total, k, l * f_total))
            continue
        if e not in vals:
            vals[e], cens[e], ld[e] = v * f_total, k, l * f_total
        else:
            v_prev, k_prev, l_prev = vals[e], cens[e], ld[e]
            vals[e] = v_prev.combine_first(v * f_total)
            cens[e] = k_prev.where(v_prev.notna(), k)
            ld[e] = l_prev.where(v_prev.notna(), l * f_total)

    if au:
        v, k, l = au[0]
        for v2, k2, l2 in au[1:]:
            k, l = k.where(v.notna(), k2), l.where(v.notna(), l2)
            v = v.combine_first(v2)
        vals["Au"], cens["Au"], ld["Au"] = v, k, l

    vals, cens, ld = pd.DataFrame(vals, index=raw.index), pd.DataFrame(cens, index=raw.index), pd.DataFrame(ld, index=raw.index)

    # Suma de Tierras Raras (Σ ETR)
    ree = [e for e in TIERRAS_RARAS if e in vals]
    if ree:
        vals["REE"] = vals[ree].fillna(0).sum(axis=1)
        cens["REE"], ld["REE"] = cens[ree].all(axis=1), np.nan

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
            pos = x[x > 0]
            lx = np.log10(pos) if not pos.empty else pd.Series(dtype=float)
            mu = lx.mean() if len(lx) else np.nan
            sd = lx.std(ddof=1) if len(lx) > 1 else 0.0
            filas.append(dict(
                grupo=g, sitio=m0.sitio, tipo=m0.tipo, estado=m0.estado, elemento=e,
                n=len(idx), n_val=len(x), pct_cens=pct,
                estatus="OK" if pct <= LIM_CENS_OK else ("Advertencia" if pct <= LIM_CENS_MAX else "Descartado"),
                media=x.mean(), gm=10 ** mu if not np.isnan(mu) else np.nan,
                p1=10 ** (mu + sd) if not np.isnan(mu) else np.nan,
                p2=10 ** (mu + 2 * sd) if not np.isnan(mu) else np.nan,
                maximo=x.max(),
                pct_sobre_min=(x >= _min_ppm(e, d_umb)).mean() if e in d_umb else np.nan))
    if not filas:
        return pd.DataFrame(columns=["grupo", "sitio", "tipo", "estado", "elemento", "n", "n_val", "pct_cens", "estatus", "media", "gm", "p1", "p2", "maximo", "pct_sobre_min"])
    return pd.DataFrame(filas)


def etiqueta_elemento(e, usar_oxidos=True):
    if e == "PGE":
        return "PGE (Pt+Pd+Au)"
    if usar_oxidos and e in FACTORES_OXIDO:
        return FACTORES_OXIDO[e][0]
    return e


def razones_min(est, stat="media", umb_dict=None, usar_oxidos=True):
    d_umb = umb_dict if umb_dict is not None else UMB_MINABLES
    if est.empty:
        return pd.DataFrame(columns=["grupo", "sitio", "tipo", "estado", "elemento", "n", "n_val", "pct_cens", "estatus", "media", "gm", "p1", "p2", "maximo", "pct_sobre_min", "unidad", "minimo", "etiqueta_elem", "valor", "razon", "razon_p1", "razon_p2"])
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
    if est.empty:
        return pd.DataFrame(columns=["grupo", "elemento", "media", "estatus", "lim_mx", "lim_eu", "razon_mx", "razon_eu"])
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
        a = rmin[rmin.grupo == g] if not rmin.empty else pd.DataFrame()
        t = rtox[rtox.grupo == g] if not rtox.empty else pd.DataFrame()
        rent = a[a.razon >= 1].sort_values("razon", ascending=False) if not a.empty else pd.DataFrame()
        mx = t[t.razon_mx > 1] if not t.empty else pd.DataFrame()
        eu = t[(t.razon_eu > 1) & ~(t.razon_mx > 1)] if not t.empty else pd.DataFrame()
        decl = {s.strip() for s in str(m.interes.iloc[0]).split(",") if s.strip() and s.strip() != "No especificado"}
        conf = set(rent.elemento) if not rent.empty else set()
        filas.append(dict(
            grupo=g, sitio=m.sitio.iloc[0], tipo=m.tipo.iloc[0], estado=m.estado.iloc[0],
            municipio=m.municipio.iloc[0], n=len(m), lat=m.lat.mean(), lon=m.lon.mean(),
            declarados=m.interes.iloc[0],
            vol_m3=m.vol_m3.sum(min_count=1),
            pinta=len(rent) > 0,
            elementos_rentables=", ".join(f"{r.etiqueta_elem} ({r.valor:.2f} {r.unidad})" for r in rent.itertuples()) if not rent.empty else "",
            no_declarados=", ".join(sorted(conf - decl)) if decl else "",
            sin_confirmar=", ".join(sorted((decl & set(rmin.elemento.unique())) - conf)) if (decl and not rmin.empty) else "",
            razon_econ=a.razon.max() if (not a.empty and a.razon.notna().any()) else np.nan,
            elem_econ=a.loc[a.razon.idxmax(), "etiqueta_elem"] if (not a.empty and a.razon.notna().any()) else "",
            nivel_tox="Supera NOM-147 (MX)" if len(mx) else ("Supera ref. europea" if len(eu) else "Debajo de límites"),
            excede_mx=", ".join(f"{r.elemento} ({r.media:.0f})" for r in mx.itertuples()) if not mx.empty else "",
            excede_eu=", ".join(f"{r.elemento} ({r.media:.0f})" for r in eu.itertuples()) if not eu.empty else "",
            razon_tox=t.razon_mx.max() if (not t.empty and t.razon_mx.notna().any()) else np.nan,
            elem_tox=t.loc[t.razon_mx.idxmax(), "elemento"] if (not t.empty and t.razon_mx.notna().any()) else ""))
    return pd.DataFrame(filas)


def clasificar_muestras(meta, vals_eval, vals_raw, umb_dict=None):
    """Clasificación muestra por muestra, resistente a cualquier subconjunto de elementos."""
    d_umb = umb_dict if umb_dict is not None else UMB_MINABLES
    out = meta.copy()
    elems_econ = [e for e in d_umb if e in vals_eval.columns]
    if elems_econ:
        Rm = pd.concat({e: vals_eval[e] * (1e-4 if d_umb[e][1] == "%" else 1.0) / d_umb[e][0]
                        for e in elems_econ}, axis=1)
        out["razon_econ"] = Rm.max(axis=1)
        out["elem_econ"] = Rm.fillna(-np.inf).idxmax(axis=1).where(out.razon_econ.notna(), "")
    else:
        out["razon_econ"] = np.nan
        out["elem_econ"] = ""

    elems_tox = [e for e in UMB_TOXICOS if e in vals_raw.columns]
    if elems_tox:
        Rx = pd.concat({e: vals_raw[e] / UMB_TOXICOS[e][0] for e in elems_tox}, axis=1)
        Re = pd.concat({e: vals_raw[e] / UMB_TOXICOS[e][1] for e in elems_tox}, axis=1)
        out["razon_mx"], out["razon_eu"] = Rx.max(axis=1), Re.max(axis=1)
    else:
        out["razon_mx"], out["razon_eu"] = np.nan, np.nan

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
    if est.empty:
        return pd.DataFrame(columns=["grupo", "elemento", "media", "estatus", "min_ppm", "razon", "percentil"])
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
    if rmin.empty:
        return pd.DataFrame(columns=["estado", "categoria", "rank_estado", "grupo", "n", "elemento", "valor", "unidad", "razon", "pct_sobre_min", "vol_m3", "nivel_tox", "declarados", "descartada", "confiable"])
    d = rmin[~rmin.elemento.isin(excluir) & (rmin.razon >= 1)]
    if d.empty:
        return pd.DataFrame(columns=["estado", "categoria", "rank_estado", "grupo", "n", "elemento", "valor", "unidad", "razon", "pct_sobre_min", "vol_m3", "nivel_tox", "declarados", "descartada", "confiable"])
    mejor = (d.sort_values("razon", ascending=False).groupby("grupo").first().reset_index()
             [["grupo", "elemento", "razon", "valor", "unidad", "pct_sobre_min"]])
    z = res.drop(columns=["elem_econ", "razon_econ"], errors="ignore").merge(mejor, on="grupo")
    z["vol_ok"] = np.where(z.vol_m3.isna(), "sin dato", np.where(z.vol_m3 >= vol_min, "sí", "no"))
    z["descartada"] = z.vol_ok == "no"
    z["confiable"] = z.n >= n_min
    z = z.sort_values(["estado", "descartada", "confiable", "razon"], ascending=[True, True, False, False])
    z["rank_estado"] = z.groupby("estado").cumcount() + 1
    z["categoria"] = np.where(z.descartada, "Descartada por volumen",
                      np.where((z.rank_estado == 1) & z.confiable, "Zona propuesta",
                      np.where(z.confiable, "Potencial adicional", "Potencial con n bajo (verificar)")))
    return z
