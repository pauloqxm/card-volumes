import io
import re
import math
import requests
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Monitoramento dos Reservatórios", layout="wide")


SHEET_URL = "https://docs.google.com/spreadsheets/d/1fbaYqjee8h4dAA8ew0RXbHOKdnSDoHIB2xPpdveYMDU/edit?usp=sharing"

# Cabeçalhos esperados (da tua tabela)
COL_NAME = "Nome do reservatório"
COL_DATE_START = "09/02/2026"     # coluna E
COL_DATE_END = "24/02/26"         # coluna F
COL_VAR_M = "Variação em m"
COL_VAR_M3 = "Variação em m³"
COL_VOL = "Volume atual"
COL_PCT = "Percentual atual"


# -----------------------------
# Google Sheets -> CSV
# -----------------------------
def sheets_to_csv_url(url: str, gid: str | None = None) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Não consegui achar o ID da planilha no link.")
    sheet_id = m.group(1)

    if gid is None:
        gid_match = re.search(r"gid=([0-9]+)", url)
        gid = gid_match.group(1) if gid_match else "0"

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


@st.cache_data(ttl=300)
def load_sheet_df(sheet_url: str, gid: str | None = None) -> pd.DataFrame:
    csv_url = sheets_to_csv_url(sheet_url, gid)
    r = requests.get(csv_url, timeout=30)
    r.raise_for_status()
    data = r.content.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(data))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_number(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace("%", "").replace("m³", "").replace("m3", "").replace("m", "")
    s = s.replace(" ", "")
    if s.count(",") == 1 and s.count(".") >= 1:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-\+]", "", s)
    try:
        return float(s)
    except:
        return None


# -----------------------------
# Render imagem
# -----------------------------
def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except:
            pass
    return ImageFont.load_default()


def draw_round_rect(draw, xy, r, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def draw_up_arrow(draw, x, y, size, color):
    w = size
    h = size
    # triângulo
    draw.polygon([(x + w // 2, y), (x + w, y + h // 2), (x, y + h // 2)], fill=color)
    # haste
    shaft_w = max(4, w // 5)
    sx1 = x + (w - shaft_w) // 2
    sy1 = y + h // 2
    draw.rectangle([sx1, sy1, sx1 + shaft_w, y + h], fill=color)


def format_num(v, suffix=""):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if s.endswith(",00"):
        s = s[:-3]
    return f"{s}{suffix}"


def generate_card(title: str, period_text: str, items: list[dict], w=1080, h=1350):
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    green = (22, 163, 74)
    green_dark = (12, 129, 58)
    green_soft = (220, 252, 231)
    text_dark = (15, 23, 42)
    gray = (71, 85, 105)

    font_title = load_font(58, bold=True)
    font_period = load_font(28, bold=False)
    font_card_title = load_font(34, bold=True)
    font_label = load_font(24, bold=False)
    font_value = load_font(26, bold=True)

    pad = 72
    y = 70

    draw.text((pad, y), title, fill=text_dark, font=font_title)
    y += 80

    if period_text.strip():
        draw.text((pad, y), period_text, fill=gray, font=font_period)
        y += 55
    else:
        y += 20

    draw.line((pad, y, w - pad, y), fill=(226, 232, 240), width=3)
    y += 35

    cols = 1 if len(items) <= 4 else 2
    gap = 26
    card_w = (w - 2 * pad - (gap if cols == 2 else 0)) // cols
    card_h = 185

    def draw_item(x, yy, item):
        draw_round_rect(draw, (x, yy, x + card_w, yy + card_h), r=28, fill=green_soft, outline=(187, 247, 208), width=3)
        draw_round_rect(draw, (x, yy, x + card_w, yy + 52), r=28, fill=green, outline=None, width=0)
        draw.rectangle((x, yy + 28, x + card_w, yy + 52), fill=green)

        name = str(item["name"]).strip()
        name = name[:40] + "…" if len(name) > 41 else name
        draw.text((x + 20, yy + 10), name, fill="white", font=font_card_title)

        body_x = x + 20
        body_y = yy + 70

        draw.text((body_x, body_y), "Variação do nível", fill=text_dark, font=font_label)
        draw_up_arrow(draw, x + card_w - 60, body_y + 2, size=34, color=green_dark)
        draw.text((body_x, body_y + 30), f"{format_num(item['var_m'], ' m')}", fill=green_dark, font=font_value)

        line_y = body_y + 78

        draw.text((body_x, line_y), "Variação", fill=gray, font=font_label)
        draw.text((body_x, line_y + 26), format_num(item["var_m3"], " m³"), fill=text_dark, font=font_value)

        right_x = x + card_w // 2 + 10
        draw.text((right_x, line_y), "Volume atual", fill=gray, font=font_label)
        draw.text((right_x, line_y + 26), format_num(item["vol"], " m³"), fill=text_dark, font=font_value)

        draw.text((right_x, line_y + 64), "Percentual", fill=gray, font=font_label)
        pct = item["pct"]
        pct_txt = "-" if pct is None else (f"{pct:.2f}".replace(".", ",") + "%").replace(",00%", "%")
        draw.text((right_x, line_y + 90), pct_txt, fill=text_dark, font=font_value)

    for i, it in enumerate(items[:8]):
        row = i // cols
        col = i % cols
        x = pad + col * (card_w + (gap if cols == 2 else 0))
        yy = y + row * (card_h + gap)
        draw_item(x, yy, it)

    draw.text((pad, h - 60), "Fonte: Monitoramento dos reservatórios", fill=(148, 163, 184), font=load_font(22))
    return img


def image_to_bytes(img: Image.Image, fmt: str):
    buf = io.BytesIO()
    if fmt.upper() == "JPG":
        img.save(buf, format="JPEG", quality=92, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.title("Gerador de Arte. Monitoramento dos Reservatórios")

with st.sidebar:
    st.subheader("Configurações")
    sheet_url = st.text_input("Link do Google Sheets", value=SHEET_URL)
    gid = st.text_input("GID (opcional)", value="")
    min_var = st.number_input("Variação mínima (m) para destacar", min_value=0.0, value=0.01, step=0.01)
    top_n = st.slider("Quantidade máxima na arte", 3, 8, 6)
    out_fmt = st.selectbox("Formato de saída", ["PNG", "JPG"])

# Carregar
try:
    df = load_sheet_df(sheet_url, gid if gid.strip() else None)
except Exception as e:
    st.error(f"Erro ao ler a planilha: {e}")
    st.stop()

# Confere colunas
required = [COL_NAME, COL_DATE_START, COL_DATE_END, COL_VAR_M, COL_VAR_M3, COL_VOL, COL_PCT]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Faltam colunas no Sheet. Achei diferente do esperado:")
    st.write(missing)
    st.write("Colunas encontradas:")
    st.write(list(df.columns))
    st.stop()

st.write("Prévia dos dados")
st.dataframe(df.head(30), use_container_width=True)

# Período usando colunas E e F (como tu pediu)
# Pega o primeiro valor não vazio dessas colunas
a = df[COL_DATE_START].dropna().astype(str).head(1).tolist()
b = df[COL_DATE_END].dropna().astype(str).head(1).tolist()
a = a[0].strip() if a else COL_DATE_START
b = b[0].strip() if b else COL_DATE_END
period_text = f"Comparativo {a} até {b}"

work = df.copy()
work["_name"] = work[COL_NAME].astype(str)
work["_var_m"] = work[COL_VAR_M].apply(to_number)
work["_var_m3"] = work[COL_VAR_M3].apply(to_number)
work["_vol"] = work[COL_VOL].apply(to_number)
work["_pct"] = work[COL_PCT].apply(to_number)

filtered = work[(work["_var_m"].notna()) & (work["_var_m"] > 0) & (work["_var_m"] >= float(min_var))].copy()
filtered = filtered.sort_values("_var_m", ascending=False).head(int(top_n))

st.subheader("Reservatórios que entram na arte")
st.dataframe(
    filtered[["_name", "_var_m", "_var_m3", "_vol", "_pct"]].rename(
        columns={"_name": "Reservatório", "_var_m": "Variação (m)", "_var_m3": "Variação (m³)", "_vol": "Volume atual", "_pct": "Percentual"}
    ),
    use_container_width=True
)

items = [
    {"name": r["_name"], "var_m": r["_var_m"], "var_m3": r["_var_m3"], "vol": r["_vol"], "pct": r["_pct"]}
    for _, r in filtered.iterrows()
]

if not items:
    st.warning("Nada passou no filtro. Baixa a variação mínima.")
    st.stop()

img = generate_card("Monitoramento dos Reservatórios", period_text, items)
st.subheader("Pré-visualização")
st.image(img, use_container_width=True)

img_bytes = image_to_bytes(img, out_fmt)
mime = "image/png" if out_fmt == "PNG" else "image/jpeg"
st.download_button(
    f"Baixar imagem ({out_fmt})",
    data=img_bytes,
    file_name=f"monitoramento_reservatorios.{out_fmt.lower()}",
    mime=mime
)
