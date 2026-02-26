# =============================================================
#  app.py  |  Monitoramento de Reservatórios - Card Generator
#  GF Informática  |  Pedro Ferreira
#
#  Atualizações:
#  - Nome do Açude: 1 linha com reticências (sem quebrar)
#  - Município: 1 linha com reticências (sem quebrar)
#  - Google Sheets + Upload CSV
#  - Tema branco e sidebar azul claro
#  - Cards positivos em azul
#  - KPI Total / Com aporte / Sem aporte
#  - Volume e variações em milhões/m³ (com opção de conversão)
# =============================================================

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from zoneinfo import ZoneInfo
import re

BASE_LAYOUT_PATH = "base_card.png"
TZ_FORTALEZA = ZoneInfo("America/Fortaleza")


# ------------------------------
# Fontes
# ------------------------------
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    size = int(size) if size is not None else 14
    if size < 1:
        size = 1

    paths_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    paths_regular = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]

    for path in (paths_bold if bold else paths_regular):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError, ValueError):
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ------------------------------
# Parsing numérico robusto
# ------------------------------
def smart_to_float(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None

    s = s.replace("m³", "").replace("m3", "").replace("%", "")
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\-\+\,\.]", "", s)

    if s.count(",") > 0 and s.count(".") > 0:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(",") > 0 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        if s.count(".") > 1:
            s = s.replace(".", "")

    try:
        return float(s)
    except:
        return None


def to_num_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(smart_to_float), errors="coerce")


# ------------------------------
# Formatação
# ------------------------------
def fmt_m_2dp_dot(v) -> str:
    if pd.isna(v):
        return "N/A"
    try:
        return f"{float(v):.2f} m"
    except:
        return "N/A"


def fmt_milhoes_br(v, convert_raw_m3_to_millions: bool) -> str:
    """
    Exibe como "8,00 milhões/m³".
    Se convert_raw_m3_to_millions=True, converte m³ bruto -> milhões (divide 1.000.000).
    """
    if pd.isna(v):
        return "N/A"
    try:
        val = float(v)
        if convert_raw_m3_to_millions:
            val = val / 1_000_000.0
        s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} milhões/m³"
    except:
        return "N/A"


def fmt_pct_br(v) -> str:
    if pd.isna(v):
        return "N/A"
    try:
        return f"{float(v):.1f}".replace(".", ",")
    except:
        return "N/A"


# ------------------------------
# Helpers de texto
# ------------------------------
def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def ellipsize_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """
    Mantém texto em 1 linha, corta com reticências se passar do limite.
    """
    text = (text or "").strip()
    if not text:
        return "N/A"
    if text_width(draw, text, font) <= max_width:
        return text

    ell = "…"
    lo, hi = 0, len(text)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid].rstrip() + ell
        if text_width(draw, cand, font) <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ------------------------------
# Leitura CSV upload
# ------------------------------
def load_csv_from_upload(file) -> pd.DataFrame:
    data = file.read()
    try:
        df = pd.read_csv(BytesIO(data), sep=";", dtype=str, encoding="utf-8")
        if df.shape[1] == 1:
            raise ValueError("CSV com 1 coluna")
        return df
    except Exception:
        return pd.read_csv(BytesIO(data), sep=",", dtype=str, encoding="utf-8")


# ------------------------------
# Google Sheets helpers
# ------------------------------
def sheets_to_csv_url(sheet_url_or_id: str, gid: str = "0") -> str:
    s = (sheet_url_or_id or "").strip()
    if not s:
        return ""
    if "docs.google.com/spreadsheets" in s:
        m = re.search(r"/d/([a-zA-Z0-9\-_]+)", s)
        if not m:
            return ""
        sheet_id = m.group(1)
    else:
        sheet_id = s
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


@st.cache_data(ttl=300)
def load_data_from_sheets(csv_url: str) -> pd.DataFrame:
    resp = requests.get(csv_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return pd.read_csv(BytesIO(resp.content), dtype=str)


# ------------------------------
# Mapeamento colunas
# ------------------------------
def _norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip()).upper()


def find_date_cols(cols: list[str]) -> list[str]:
    date_like = []
    for c in cols:
        s = str(c).strip()
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", s):
            date_like.append(c)
    return date_like[:2]


def process_df(df_raw: pd.DataFrame):
    cols = list(df_raw.columns)
    norm_map = {_norm_col(c): c for c in cols}

    def col(name_upper: str):
        return norm_map.get(name_upper)

    c_ger = col("GERÊNCIA")
    c_bacia = col("BACIA")
    c_acude = col("AÇUDE")
    c_mun = col("MUNICÍPIO") or col("MUNICIPIO")
    c_var_m = col("VARIAÇÃO_M") or col("VARIAÇÃO EM M") or col("VARIACAO EM M")
    c_var_m3 = col("VARIAÇÃO_M³") or col("VARIAÇÃO EM M³") or col("VARIACAO EM M3") or col("VARIAÇÃO_M3")
    c_vol_atual = col("SITUAÇÃO ATUAL") or col("VOLUME ATUAL")
    c_pct_atual = col("PERCENTUAL ATUAL") or col("PERCENTUAL")

    date_cols = find_date_cols(cols)
    date_ant = date_cols[0] if len(date_cols) > 0 else ""
    date_atu = date_cols[1] if len(date_cols) > 1 else ""

    df = pd.DataFrame({
        "gerencia": (df_raw[c_ger].astype(str).str.strip() if c_ger else "N/A"),
        "bacia": (df_raw[c_bacia].astype(str).str.strip() if c_bacia else "N/A"),
        "nome": (df_raw[c_acude].astype(str).str.strip() if c_acude else df_raw.iloc[:, 0].astype(str).str.strip()),
        "municipio": (df_raw[c_mun].astype(str).str.strip() if c_mun else "N/A"),
        "data_anterior": str(date_ant).strip(),
        "data_atual": str(date_atu).strip(),
        "nivel_anterior": to_num_series(df_raw[date_ant]) if date_ant in df_raw.columns else pd.Series([None] * len(df_raw)),
        "nivel_atual": to_num_series(df_raw[date_atu]) if date_atu in df_raw.columns else pd.Series([None] * len(df_raw)),
        "variacao_m": to_num_series(df_raw[c_var_m]) if c_var_m else pd.Series([None] * len(df_raw)),
        "variacao_m3": to_num_series(df_raw[c_var_m3]) if c_var_m3 else pd.Series([None] * len(df_raw)),
        "volume_atual_m3": to_num_series(df_raw[c_vol_atual]) if c_vol_atual else pd.Series([None] * len(df_raw)),
        "percentual": to_num_series(df_raw[c_pct_atual]) if c_pct_atual else pd.Series([None] * len(df_raw)),
    })

    df = df[
        df["nome"].notna()
        & (df["nome"].astype(str).str.strip() != "")
        & (~df["nome"].astype(str).str.lower().isin(["nan", "none", "n/a"]))
    ].reset_index(drop=True)

    if df["variacao_m"].isna().all():
        df["variacao_m"] = (df["nivel_atual"] - df["nivel_anterior"]).round(2)

    for c in ["variacao_m", "variacao_m3", "volume_atual_m3", "percentual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    info = {"periodo": {"anterior": str(date_ant).strip(), "atual": str(date_atu).strip()}, "colunas": cols}
    return df, info


def build_fonte_gerencia(df: pd.DataFrame) -> str:
    uniques = [g for g in df.get("gerencia", pd.Series([])).dropna().astype(str).str.strip().unique().tolist() if g]
    if not uniques:
        return "Fonte: N/A"
    if len(uniques) <= 3:
        return "Fonte: " + " • ".join(uniques)
    return "Fonte: " + " • ".join(uniques[:3]) + f" • +{len(uniques) - 3}"


def build_bacia_label(df: pd.DataFrame) -> str:
    uniques = [b for b in df.get("bacia", pd.Series([])).dropna().astype(str).str.strip().unique().tolist() if b]
    if not uniques:
        return "N/A"
    if len(uniques) == 1:
        return uniques[0]
    if len(uniques) <= 3:
        return " / ".join(uniques)
    return " / ".join(uniques[:3]) + f" / +{len(uniques) - 3}"


# ------------------------------
# Desenho
# ------------------------------
def draw_rounded_rect(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                      r: int, fill, outline=None, width: int = 2):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, x: int, y: int, up: bool, size: int, color):
    w = size
    h = size
    if up:
        tri = [(x + w // 2, y), (x + w, y + h // 2), (x, y + h // 2)]
        shaft = [x + w // 2 - max(2, w // 10), y + h // 2,
                 x + w // 2 + max(2, w // 10), y + h]
    else:
        tri = [(x, y + h // 2), (x + w, y + h // 2), (x + w // 2, y + h)]
        shaft = [x + w // 2 - max(2, w // 10), y,
                 x + w // 2 + max(2, w // 10), y + h // 2]
    draw.polygon(tri, fill=color)
    draw.rectangle(shaft, fill=color)


def draw_kpi_pill(draw, x, y, w, h, label, value, outline, big=False):
    bg = (248, 250, 252, 255)
    text = (15, 23, 42, 255)
    sub = (71, 85, 105, 255)

    r = 20 if big else 18
    draw_rounded_rect(draw, x, y, w, h, r, fill=bg, outline=outline, width=3)

    f_lab = get_font(22 if big else 20, True)
    f_val = get_font(30 if big else 28, True)

    draw.text((x + 22, y + 10), label, fill=sub, font=f_lab)
    draw.text((x + w - 22, y + 6), str(value), fill=text, font=f_val, anchor="ra")


def draw_kpis_row(draw, x, y, total, up, down, big=False):
    gap = 18
    h = 54 if big else 50
    w = 300 if big else 290

    o_total = (148, 163, 184, 255)
    o_up = (59, 130, 246, 255)     # azul
    o_down = (244, 63, 94, 255)    # vermelho

    draw_kpi_pill(draw, x + 0*(w+gap), y, w, h, "Total", total, o_total, big)
    draw_kpi_pill(draw, x + 1*(w+gap), y, w, h, "Com aporte", up, o_up, big)
    draw_kpi_pill(draw, x + 2*(w+gap), y, w, h, "Sem aporte", down, o_down, big)

    return y + h


def draw_bacia_pill(draw, right_x, y, text_value, big=False):
    outline = (147, 197, 253, 255)
    bg = (255, 255, 255, 255)
    tx = (30, 64, 175, 255)

    f = get_font(22 if big else 20, True)
    label = f"Bacia: {text_value}"
    w = text_width(draw, label, f) + 34
    h = 44 if big else 40
    x = right_x - w

    draw_rounded_rect(draw, x, y, w, h, 18, fill=bg, outline=outline, width=3)
    draw.text((x + 18, y + 9), label, fill=tx, font=f)
    return x


def generate_image(df_all: pd.DataFrame, mode: str, date_anterior: str, date_atual: str,
                   ordenar: str, formato: str, convert_raw_m3_to_millions: bool) -> Image.Image:

    if mode == "Feed (1080x1350)":
        try:
            base = Image.open(BASE_LAYOUT_PATH).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1080, 1350), (255, 255, 255, 255))
        W, H = base.size
        img = base.copy()
        draw = ImageDraw.Draw(img)
        big = False
        cols_grid, rows_grid = 3, 6
    else:
        W, H = 1080, 1920
        img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        big = True
        cols_grid, rows_grid = 2, 9

    dark = (15, 23, 42, 255)
    gray = (71, 85, 105, 255)

    # positivos: azul
    blue_bg = (219, 234, 254, 255)
    blue_bd = (59, 130, 246, 255)
    blue_tx = (29, 78, 216, 255)

    # negativos: vermelho
    red_bg = (255, 241, 242, 255)
    red_bd = (251, 113, 133, 255)
    red_tx = (225, 29, 72, 255)

    # neutro: cinza
    neutral_bg = (241, 245, 249, 255)
    neutral_bd = (148, 163, 184, 255)
    neutral_tx = (51, 65, 85, 255)

    f_sub = get_font(34 if big else 28, False)

    # tamanhos
    f_name_base = 22 if big else 18
    f_line_base = 17 if big else 15
    f_var_base = 22 if big else 18

    pad = 70

    total = int(len(df_all))
    up = int((df_all["variacao_m"] > 0).sum()) if "variacao_m" in df_all.columns else 0
    down = int((df_all["variacao_m"] < 0).sum()) if "variacao_m" in df_all.columns else 0

    bacia_txt = build_bacia_label(df_all)

    y = 70
    if big:
        f_title = get_font(66, True)
        draw.text((pad, y), "Monitoramento dos Reservatórios", fill=dark, font=f_title)
        y += 92

    if not big:
        y = 150

    comparativo = f"Comparativo  {date_anterior}  →  {date_atual}"
    draw.text((pad, y), comparativo, fill=gray, font=f_sub)

    bacia_y = y - (4 if big else 2)
    bacia_x = draw_bacia_pill(draw, right_x=W - pad, y=bacia_y, text_value=bacia_txt, big=big)
    if bacia_x < pad + 540:
        bacia_y2 = y + (52 if big else 48)
        draw_bacia_pill(draw, right_x=W - pad, y=bacia_y2, text_value=bacia_txt, big=big)
        y += (52 if big else 48)

    y += 64 if big else 56

    y = draw_kpis_row(draw, pad, y, total=total, up=up, down=down, big=big)
    y += 20

    draw.line((pad, y, W - pad, y), fill=(226, 232, 240, 255), width=3)
    y += 24

    df = df_all.copy()
    df_pos = df[df["variacao_m"] > 0].copy()
    df_neg = df[df["variacao_m"] < 0].copy()
    df_zero = df[df["variacao_m"] == 0].copy()

    if ordenar == "Maior variação positiva":
        df_pos = df_pos.sort_values("variacao_m", ascending=False)
        df_neg = df_neg.sort_values("variacao_m", ascending=True)
    elif ordenar == "Maior variação negativa":
        df_neg = df_neg.sort_values("variacao_m", ascending=True)
        df_pos = df_pos.sort_values("variacao_m", ascending=False)
    elif ordenar == "Maior variação absoluta":
        tmp = df.assign(_abs=df["variacao_m"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
        df_pos = tmp[tmp["variacao_m"] > 0]
        df_neg = tmp[tmp["variacao_m"] < 0]
        df_zero = tmp[tmp["variacao_m"] == 0]

    ordered = pd.concat([df_pos, df_neg, df_zero], ignore_index=True).head(18).reset_index(drop=True)

    gap_x = 18
    gap_y = 16

    grid_x = pad
    grid_y = y
    grid_w = W - 2 * pad
    grid_h = H - grid_y - (110 if big else 95)

    card_w = int((grid_w - (cols_grid - 1) * gap_x) / cols_grid)
    card_h = int((grid_h - (rows_grid - 1) * gap_y) / rows_grid)

    def draw_item(ix: int, row: pd.Series, x: int, y: int):
        nome = str(row.get("nome", "N/A")).strip()
        municipio = str(row.get("municipio", "N/A")).strip()

        var_m = row.get("variacao_m", None)
        var_m3 = row.get("variacao_m3", None)
        vol = row.get("volume_atual_m3", None)
        pct = row.get("percentual", None)

        is_pos = (not pd.isna(var_m)) and (float(var_m) > 0)
        is_neg = (not pd.isna(var_m)) and (float(var_m) < 0)

        if is_pos:
            bg, bd, tx = blue_bg, blue_bd, blue_tx
            up_arrow = True
        elif is_neg:
            bg, bd, tx = red_bg, red_bd, red_tx
            up_arrow = False
        else:
            bg, bd, tx = neutral_bg, neutral_bd, neutral_tx
            up_arrow = True

        draw_rounded_rect(draw, x, y, card_w, card_h, 22, fill=bg, outline=bd, width=2)

        # badge rank
        rank_w = 44
        draw_rounded_rect(draw, x + card_w - rank_w - 10, y + 10, rank_w, 30, 14, fill=bd, outline=None, width=0)
        draw.text((x + card_w - 10 - rank_w / 2, y + 25), str(ix + 1),
                  fill=(255, 255, 255, 255), font=get_font(16, True), anchor="mm")

        # --- AÇUDE: 1 linha com reticências ---
        name_area_w = card_w - 28 - 54  # deixa espaço do badge
        f_name = get_font(f_name_base, True)
        nome_1linha = ellipsize_text(draw, nome.upper(), f_name, name_area_w)
        draw.text((x + 14, y + 10), nome_1linha, fill=(15, 23, 42, 255), font=f_name)

        # --- MUNICÍPIO: 1 linha com reticências ---
        f_mun = get_font(14 if big else 13, False)
        muni_text = f"Município: {municipio}"
        muni_max_w = card_w - 28
        muni_text = ellipsize_text(draw, muni_text, f_mun, muni_max_w)
        y_mun = y + 10 + f_name.size + 2
        draw.text((x + 14, y_mun), muni_text, fill=(100, 116, 139, 255), font=f_mun)

        # variação principal
        f_var = get_font(f_var_base, True)
        arrow_x = x + 14
        arrow_y = y + (58 if big else 54)
        draw_arrow(draw, arrow_x, arrow_y, up_arrow, 22 if big else 20, tx)

        if pd.isna(var_m):
            var_txt = "N/A"
        else:
            sign = "+" if float(var_m) > 0 else ""
            var_txt = f"{sign}{fmt_m_2dp_dot(var_m)}"
        draw.text((x + 44, arrow_y - 2), var_txt, fill=tx, font=f_var)

        # linhas
        f_line = get_font(f_line_base, False)
        l1 = f"Var. m³: {fmt_milhoes_br(var_m3, convert_raw_m3_to_millions)}"
        l2 = f"Vol: {fmt_milhoes_br(vol, convert_raw_m3_to_millions)}"
        l3 = f"%: {fmt_pct_br(pct)}"

        base_y = y + (86 if big else 78)
        draw.text((x + 14, base_y), l1, fill=(51, 65, 85, 255), font=f_line)
        draw.text((x + 14, base_y + (22 if big else 20)), l2, fill=(51, 65, 85, 255), font=f_line)
        draw.text((x + 14, base_y + (44 if big else 40)), l3, fill=(51, 65, 85, 255), font=f_line)

    for i in range(min(18, len(ordered))):
        ri = i // cols_grid
        ci = i % cols_grid
        cx = grid_x + ci * (card_w + gap_x)
        cy = grid_y + ri * (card_h + gap_y)
        draw_item(i, ordered.iloc[i], cx, cy)

    # rodapé
    fonte_txt = build_fonte_gerencia(df_all)
    foot_y = H - (72 if big else 70)
    draw.line((pad, foot_y - 18, W - pad, foot_y - 18), fill=(226, 232, 240, 255), width=2)

    f_foot = get_font(26 if big else 22, False)
    draw.text((pad, foot_y), fonte_txt, fill=(100, 116, 139, 255), font=f_foot)

    ts = datetime.now(TZ_FORTALEZA).strftime("%d/%m/%Y %H:%M")
    draw.text((W - pad, foot_y), f"Gerado em {ts}", fill=(100, 116, 139, 255), font=f_foot, anchor="ra")

    return img.convert("RGB") if formato.upper() == "JPG" else img


# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.set_page_config(
        page_title="Reservatórios - Card Generator",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Tema branco e sidebar azul claro
    st.markdown(
        """
        <style>
            .stApp { background-color: #ffffff; color: #0f172a; }
            section[data-testid="stSidebar"] { background-color: #dbeafe !important; }
            .stMarkdown, .stText, label, p, span, div { color: #0f172a; }
            h1, h2, h3 { color: #0b2a5b !important; }
            .stButton > button {
                background:#2563eb; color:#fff; border-radius:10px;
                font-weight:800; border:none;
            }
            .stButton > button:hover { background:#1d4ed8; }
            .stDownloadButton > button {
                background:#0ea5e9; color:#fff; border-radius:10px;
                font-weight:800; border:none;
            }
            .stDownloadButton > button:hover { background:#0284c7; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("💧 Gerador de Card. Monitoramento de Reservatórios")
    st.caption("Automatize com Google Sheets ou use CSV quando precisar.")
    st.divider()

    with st.sidebar:
        st.markdown("## Fonte de dados")
        fonte = st.radio("Escolha", ["Google Sheets", "Upload CSV"], index=0)

        df_raw = None

        if fonte == "Google Sheets":
            st.markdown("### Google Sheets")
            sheet_link = st.text_input(
                "Link (ou ID) da planilha",
                value="https://docs.google.com/spreadsheets/d/1fbaYqjee8h4dAA8ew0RXbHOKdnSDoHIB2xPpdveYMDU/edit?usp=sharing"
            )
            gid = st.text_input("GID", value="0", help="Aba da planilha. Geralmente 0.")
            csv_url = sheets_to_csv_url(sheet_link, gid=gid)
            st.caption("URL CSV gerada automaticamente.")
            st.code(csv_url or "Informe um link/ID válido", language="text")

            if st.button("Atualizar dados", use_container_width=True):
                load_data_from_sheets.clear()
                st.rerun()

            if csv_url:
                try:
                    df_raw = load_data_from_sheets(csv_url)
                except Exception as e:
                    st.error(f"Erro ao ler planilha: {e}")

        else:
            st.markdown("### Upload CSV")
            uploaded = st.file_uploader("Envie o arquivo .csv", type=["csv"])
            if uploaded is not None:
                try:
                    df_raw = load_csv_from_upload(uploaded)
                except Exception as e:
                    st.error(f"Erro lendo CSV: {e}")

        st.divider()
        mode = st.selectbox("Formato", ["Feed (1080x1350)", "Stories (1080x1920)"], index=0)
        ordenar = st.selectbox(
            "Ordenação",
            ["Manter ordem", "Maior variação absoluta", "Maior variação positiva", "Maior variação negativa"],
            index=0
        )
        formato = st.selectbox("Saída", ["PNG", "JPG"], index=0)

        convert_raw = st.checkbox(
            "Converter m³ bruto para milhões/m³",
            value=True,
            help="Marque se vier em m³ (ex: 8000000). Desmarque se já vier em milhões (ex: 8,00)."
        )

        debug = st.toggle("Mostrar prévia", value=False)
        st.caption("GF Informática")

    if df_raw is None or df_raw.empty:
        st.info("Escolha a fonte na lateral e carregue os dados.")
        return

    try:
        df_proc, info = process_df(df_raw)
    except Exception as e:
        st.error(f"Erro processando dados: {e}")
        st.stop()

    total = len(df_proc)
    up = int((df_proc["variacao_m"] > 0).sum()) if "variacao_m" in df_proc.columns else 0
    down = int((df_proc["variacao_m"] < 0).sum()) if "variacao_m" in df_proc.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Com aporte", up)
    c3.metric("Sem aporte", down)

    if debug:
        st.subheader("Prévia processada")
        st.dataframe(df_proc, use_container_width=True)

    st.divider()

    if st.button("🎨 Gerar imagem", type="primary", use_container_width=True):
        if df_proc.empty:
            st.warning("Sem dados para renderizar.")
            return

        d_ant = info.get("periodo", {}).get("anterior", "")
        d_atu = info.get("periodo", {}).get("atual", "")

        with st.spinner("Renderizando..."):
            img_final = generate_image(
                df_all=df_proc,
                mode=mode,
                date_anterior=d_ant,
                date_atual=d_atu,
                ordenar=ordenar,
                formato=formato,
                convert_raw_m3_to_millions=convert_raw
            )

        st.image(img_final, caption=f"Preview {mode}", use_container_width=True)

        buf = BytesIO()
        save_fmt = "JPEG" if formato.upper() == "JPG" else "PNG"
        if save_fmt == "JPEG":
            img_final.save(buf, format=save_fmt, quality=95, optimize=True)
            mime = "image/jpeg"
        else:
            img_final.save(buf, format=save_fmt, optimize=True)
            mime = "image/png"
        buf.seek(0)

        ts_name = datetime.now(TZ_FORTALEZA).strftime("%Y%m%d_%H%M%S")
        fname = f"monitoramento_reservatorios_{'stories' if 'Stories' in mode else 'feed'}_{ts_name}.{formato.lower()}"

        st.download_button(
            label=f"📥 Baixar ({formato})",
            data=buf,
            file_name=fname,
            mime=mime,
            use_container_width=True
        )
        st.success("Pronto.")

    st.caption("Para Feed com layout, mantenha base_card.png na mesma pasta do app.py.")


if __name__ == "__main__":
    main()
