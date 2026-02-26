# =============================================================
#  app.py  |  Monitoramento de Reservatórios - Card Generator
#  GF Informática  |  Pedro Ferreira
#
#  Ajustes visuais
#  - KPI em cards melhores
#  - Subiu/Desceu => Com aporte / Sem aporte
#  - Comparativo com data inicial -> data final
#  - Bacia em pílula separada com retângulo
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

DEFAULT_SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/"
    "1fbaYqjee8h4dAA8ew0RXbHOKdnSDoHIB2xPpdveYMDU"
    "/export?format=csv&gid=0"
)

TZ_FORTALEZA = ZoneInfo("America/Fortaleza")


# -----------------------------
# Fontes
# -----------------------------
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


# -----------------------------
# Parser robusto de número
# -----------------------------
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


# -----------------------------
# Formatação final
# -----------------------------
def fmt_m_2dp_dot(v) -> str:
    if pd.isna(v):
        return "N/A"
    try:
        return f"{float(v):.2f} m"
    except:
        return "N/A"


def fmt_milhoes_br(v, convert_raw_m3_to_millions: bool) -> str:
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


# -----------------------------
# Desenho utilitários
# -----------------------------
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


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def wrap_name_two_lines(draw: ImageDraw.ImageDraw, name: str, max_width: int,
                        base_font_size: int, bold: bool) -> tuple[str, str, ImageFont.FreeTypeFont]:
    name = (name or "").strip()
    words = [w for w in re.split(r"\s+", name) if w]
    if not words:
        font = get_font(base_font_size, bold=bold)
        return "N/A", "", font

    min_font = max(12, base_font_size - 6)
    size = base_font_size

    while size >= min_font:
        font = get_font(size, bold=bold)

        line1 = ""
        line2 = ""
        i = 0

        while i < len(words):
            test = (line1 + " " + words[i]).strip()
            if text_width(draw, test, font) <= max_width:
                line1 = test
                i += 1
            else:
                break

        while i < len(words):
            test = (line2 + " " + words[i]).strip()
            if text_width(draw, test, font) <= max_width:
                line2 = test
                i += 1
            else:
                break

        if i == len(words):
            return line1, line2, font

        size -= 1

    font = get_font(min_font, bold=bold)
    line1 = ""
    i = 0
    while i < len(words):
        test = (line1 + " " + words[i]).strip()
        if text_width(draw, test, font) <= max_width:
            line1 = test
            i += 1
        else:
            break

    line2 = ""
    while i < len(words):
        test = (line2 + " " + words[i]).strip()
        if text_width(draw, test + "…", font) <= max_width:
            line2 = test
            i += 1
        else:
            break
    if line2:
        line2 = line2 + "…"
    return line1, line2, font


# -----------------------------
# Leitura dados
# -----------------------------
@st.cache_data(ttl=300)
def load_csv_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(BytesIO(resp.content))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_csv_from_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def process_df(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = list(df_raw.columns)

    def find_col_exact(name: str):
        for c in cols:
            if str(c).strip().lower() == name.strip().lower():
                return c
        return None

    date_anterior = cols[4] if len(cols) > 4 else ""
    date_atual = cols[5] if len(cols) > 5 else ""

    col_ger = find_col_exact("Gerência")
    col_bacia = find_col_exact("Bacia")
    col_nome = find_col_exact("Nome do reservatório") or (cols[2] if len(cols) > 2 else cols[0])

    col_var_m = find_col_exact("Variação em m")
    col_var_m3 = find_col_exact("Variação em m³") or find_col_exact("Variação em m3")
    col_vol = find_col_exact("Volume atual")
    col_pct = find_col_exact("Percentual atual")

    col_lvl_ant = cols[4] if len(cols) > 4 else None
    col_lvl_atu = cols[5] if len(cols) > 5 else None

    df = pd.DataFrame({
        "gerencia": df_raw[col_ger].astype(str).str.strip() if col_ger else "N/A",
        "bacia": df_raw[col_bacia].astype(str).str.strip() if col_bacia else "N/A",
        "nome": df_raw[col_nome].astype(str).str.strip(),
        "data_anterior": date_anterior,
        "data_atual": date_atual,
        "nivel_anterior": to_num_series(df_raw[col_lvl_ant]) if col_lvl_ant else pd.Series([None] * len(df_raw)),
        "nivel_atual": to_num_series(df_raw[col_lvl_atu]) if col_lvl_atu else pd.Series([None] * len(df_raw)),
        "variacao_m": to_num_series(df_raw[col_var_m]) if col_var_m else pd.Series([None] * len(df_raw)),
        "variacao_m3": to_num_series(df_raw[col_var_m3]) if col_var_m3 else pd.Series([None] * len(df_raw)),
        "volume_atual_m3": to_num_series(df_raw[col_vol]) if col_vol else pd.Series([None] * len(df_raw)),
        "percentual": to_num_series(df_raw[col_pct]) if col_pct else pd.Series([None] * len(df_raw)),
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

    info = {
        "colunas": cols,
        "shape": df_raw.shape,
        "periodo": {"anterior": date_anterior, "atual": date_atual},
    }
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


# -----------------------------
# KPI melhorado
# -----------------------------
def draw_kpi_card(draw, x, y, w, h, title, value, accent, big=False):
    bg = (248, 250, 252, 255)
    bd = (203, 213, 225, 255)
    text = (15, 23, 42, 255)
    sub = (71, 85, 105, 255)

    r = 18 if big else 16
    draw_rounded_rect(draw, x, y, w, h, r, fill=bg, outline=bd, width=2)

    # Faixa/acento no topo
    draw_rounded_rect(draw, x + 10, y + 10, w - 20, 8, 6, fill=accent, outline=None, width=0)

    f_t = get_font(20 if big else 16, True)
    f_v = get_font(38 if big else 30, True)

    draw.text((x + 16, y + 22), title, fill=sub, font=f_t)
    draw.text((x + w - 16, y + 16), str(value), fill=text, font=f_v, anchor="ra")


def draw_kpis(draw, x, y, total, up, down, big=False):
    gap = 16
    w = 300 if big else 280
    h = 92 if big else 78

    accent_total = (56, 189, 248, 255)  # azul
    accent_up = (16, 185, 129, 255)     # verde
    accent_down = (244, 63, 94, 255)    # vermelho

    draw_kpi_card(draw, x + 0 * (w + gap), y, w, h, "Total", total, accent_total, big)
    draw_kpi_card(draw, x + 1 * (w + gap), y, w, h, "Com aporte", up, accent_up, big)
    draw_kpi_card(draw, x + 2 * (w + gap), y, w, h, "Sem aporte", down, accent_down, big)

    return y + h


def draw_badge(draw, x, y, label, value, outline, text_color, big=False):
    f = get_font(22 if big else 20, True)
    pad_x = 16
    pad_y = 10
    text = f"{label} {value}"
    tw = text_width(draw, text, f)
    w = tw + pad_x * 2
    h = (44 if big else 40)
    draw_rounded_rect(draw, x, y, w, h, 18, fill=(255, 255, 255, 255), outline=outline, width=3)
    draw.text((x + pad_x, y + (10 if big else 9)), text, fill=text_color, font=f)
    return x + w + 14


# -----------------------------
# Geração
# -----------------------------
def generate_image(
    df_all: pd.DataFrame,
    mode: str,
    date_anterior: str,
    date_atual: str,
    ordenar: str,
    formato: str,
    convert_raw_m3_to_millions: bool,
) -> Image.Image:

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

    green_bg = (220, 252, 231, 255)
    green_bd = (16, 185, 129, 255)
    green_tx = (5, 150, 105, 255)

    red_bg = (255, 241, 242, 255)
    red_bd = (251, 113, 133, 255)
    red_tx = (225, 29, 72, 255)

    neutral_bg = (241, 245, 249, 255)
    neutral_bd = (148, 163, 184, 255)
    neutral_tx = (51, 65, 85, 255)

    f_sub = get_font(34 if big else 28, False)

    f_name_pos_base = 22 if big else 18
    f_line_pos_base = 17 if big else 15
    f_var_pos_base = 22 if big else 18

    f_name_neg_base = 20 if big else 16
    f_line_neg_base = 15 if big else 13
    f_var_neg_base = 20 if big else 16

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

    # Comparativo com data inicial -> data final (como antes)
    comparativo = f"Comparativo  {date_anterior}  →  {date_atual}"
    draw.text((pad, y), comparativo, fill=gray, font=f_sub)

    # Bacia em retângulo/pílula separada ao lado
    bx = pad + (720 if big else 680)
    by = y - (4 if big else 2)

    # se ficar apertado, joga pra linha de baixo no stories
    if bx > W - pad - 200:
        bx = pad
        by = y + (50 if big else 46)

    x_after = draw_badge(
        draw,
        bx,
        by,
        "Bacia:",
        bacia_txt,
        outline=(147, 197, 253, 255),
        text_color=(30, 64, 175, 255),
        big=big
    )

    y += 64 if big else 56

    # KPIs melhorados
    y = draw_kpis(draw, pad, y, total=total, up=up, down=down, big=big) + 18

    # linha separadora
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
        var_m = row.get("variacao_m", None)
        var_m3 = row.get("variacao_m3", None)
        vol = row.get("volume_atual_m3", None)
        pct = row.get("percentual", None)

        is_pos = (not pd.isna(var_m)) and (float(var_m) > 0)
        is_neg = (not pd.isna(var_m)) and (float(var_m) < 0)

        if is_pos:
            bg, bd, tx = green_bg, green_bd, green_tx
            up_arrow = True
            base_name = f_name_pos_base
            base_line = f_line_pos_base
            base_var = f_var_pos_base
        elif is_neg:
            bg, bd, tx = red_bg, red_bd, red_tx
            up_arrow = False
            base_name = f_name_neg_base
            base_line = f_line_neg_base
            base_var = f_var_neg_base
        else:
            bg, bd, tx = neutral_bg, neutral_bd, neutral_tx
            up_arrow = True
            base_name = f_name_neg_base
            base_line = f_line_neg_base
            base_var = f_var_neg_base

        draw_rounded_rect(draw, x, y, card_w, card_h, 22, fill=bg, outline=bd, width=2)

        rank_w = 44
        draw_rounded_rect(draw, x + card_w - rank_w - 10, y + 10, rank_w, 30, 14, fill=bd, outline=None, width=0)
        draw.text((x + card_w - 10 - rank_w / 2, y + 25), str(ix + 1),
                  fill=(255, 255, 255, 255), font=get_font(16, True), anchor="mm")

        name_area_w = card_w - 28 - 54
        line1, line2, f_name = wrap_name_two_lines(draw, nome.upper(), name_area_w, base_name, True)
        draw.text((x + 14, y + 12), line1, fill=dark, font=f_name)
        if line2:
            draw.text((x + 14, y + 12 + (f_name.size + 2)), line2, fill=dark, font=f_name)

        f_var = get_font(base_var, True)
        arrow_x = x + 14
        arrow_y = y + (56 if big else 52)
        draw_arrow(draw, arrow_x, arrow_y, up_arrow, 22 if big else 20, tx)

        if pd.isna(var_m):
            var_txt = "N/A"
        else:
            sign = "+" if float(var_m) > 0 else ""
            var_txt = f"{sign}{fmt_m_2dp_dot(var_m)}"
        draw.text((x + 44, arrow_y - 2), var_txt, fill=tx, font=f_var)

        f_line = get_font(base_line, False)
        l1 = f"Var. m³: {fmt_milhoes_br(var_m3, convert_raw_m3_to_millions)}"
        l2 = f"Vol: {fmt_milhoes_br(vol, convert_raw_m3_to_millions)}"
        l3 = f"%: {fmt_pct_br(pct)}"

        base_y = y + (86 if big else 76)
        draw.text((x + 14, base_y), l1, fill=(51, 65, 85, 255), font=f_line)
        draw.text((x + 14, base_y + (22 if big else 20)), l2, fill=(51, 65, 85, 255), font=f_line)
        draw.text((x + 14, base_y + (44 if big else 40)), l3, fill=(51, 65, 85, 255), font=f_line)

    for i in range(min(18, len(ordered))):
        ri = i // cols_grid
        ci = i % cols_grid
        cx = grid_x + ci * (card_w + gap_x)
        cy = grid_y + ri * (card_h + gap_y)
        draw_item(i, ordered.iloc[i], cx, cy)

    fonte_txt = build_fonte_gerencia(df_all)
    foot_y = H - (72 if big else 70)
    draw.line((pad, foot_y - 18, W - pad, foot_y - 18), fill=(226, 232, 240, 255), width=2)

    f_foot = get_font(26 if big else 22, False)
    draw.text((pad, foot_y), fonte_txt, fill=(100, 116, 139, 255), font=f_foot)

    ts = datetime.now(TZ_FORTALEZA).strftime("%d/%m/%Y %H:%M")
    draw.text((W - pad, foot_y), f"Gerado em {ts}", fill=(100, 116, 139, 255), font=f_foot, anchor="ra")

    return img.convert("RGB") if formato.upper() == "JPG" else img


# -----------------------------
# Streamlit
# -----------------------------
def main():
    st.set_page_config(
        page_title="Reservatórios - Card Generator",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        """
        <style>
            .stApp { background-color: #0a1628; color: #e0e0e0; }
            section[data-testid="stSidebar"] { background-color: #0d1f35 !important; }
            h1, h2, h3 { color: #4FC3F7 !important; }
            div[data-testid="metric-container"] {
                background: rgba(0,200,83,.08);
                border: 1px solid rgba(0,200,83,.28);
                border-radius: 10px;
                padding: 8px;
            }
            .stButton > button {
                background:#00C853; color:#fff; border-radius:8px;
                font-weight:700; border:none;
            }
            .stButton > button:hover { background:#00a844; }
            .stDownloadButton > button {
                background:#1565C0; color:#fff; border-radius:8px;
                font-weight:700; border:none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("💧 Gerador de Card. Monitoramento de Reservatórios")
    st.caption("KPIs premium + Bacia em pílula separada + comparativo com data inicial → data final.")
    st.divider()

    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        st.divider()

        fonte = st.radio("Fonte de dados", ["Google Sheets", "Upload CSV"], index=0)
        uploaded = None
        sheet_url = DEFAULT_SHEET_CSV

        if fonte == "Upload CSV":
            uploaded = st.file_uploader("Envie o .csv", type=["csv"])
        else:
            sheet_url = st.text_input("Link CSV do Google Sheets", value=DEFAULT_SHEET_CSV)

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
            value=False,
            help="Marque se a planilha vier com valores em m³ (ex: 8000000). Desmarque se já vier em milhões (ex: 8,00)."
        )

        debug = st.toggle("Mostrar prévia", value=False)

        st.divider()
        if st.button("Atualizar dados", use_container_width=True):
            load_csv_from_url.clear()
            st.rerun()

        st.caption("GF Informática")

    try:
        if fonte == "Upload CSV":
            if uploaded is None:
                st.info("Envia um CSV na lateral e eu gero o card.")
                return
            df_raw = load_csv_from_upload(uploaded)
        else:
            df_raw = load_csv_from_url(sheet_url)

        df_proc, info = process_df(df_raw)

    except Exception as e:
        st.error(f"Erro carregando dados: {e}")
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
        st.dataframe(df_proc.head(30), use_container_width=True)

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
