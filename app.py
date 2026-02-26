# =============================================================
#  app.py  |  Monitoramento de Reservatórios — Card Generator
#  GF Informática  |  Paulo Ferreira
#  Atualizado: unidades (m com ponto), milhões/m³, positivos em
#  destaque, negativos abaixo, fonte via Gerência, horário
#  Fortaleza-CE, upload CSV
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


# ─────────────────────────────────────────────────────────────
#  FONTES
# ─────────────────────────────────────────────────────────────
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
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
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────
#  PARSER ROBUSTO
# ─────────────────────────────────────────────────────────────
def smart_to_float(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace("m³", "").replace("m3", "").replace("m", "").replace("%", "")
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


# ─────────────────────────────────────────────────────────────
#  FORMATAÇÃO FINAL (EXATAMENTE COMO TU PEDIU)
# ─────────────────────────────────────────────────────────────
def fmt_m_2dp_dot(v) -> str:
    """0,2 -> 0.20 m (ponto e 2 casas)."""
    if pd.isna(v):
        return "N/A"
    try:
        return f"{float(v):.2f} m"
    except:
        return "N/A"


def fmt_milhoes_br(v) -> str:
    """
    8,000 -> 8,00 milhões/m³
    428,17 -> 428,17 milhões/m³
    AQUI assume que o valor já está em milhões (como no teu exemplo).
    """
    if pd.isna(v):
        return "N/A"
    try:
        s = f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} milhões/m³"
    except:
        return "N/A"


def fmt_pct_br(v) -> str:
    if pd.isna(v):
        return "N/A"
    try:
        s = f"{float(v):.1f}".replace(".", ",")
        return s
    except:
        return "N/A"


# ─────────────────────────────────────────────────────────────
#  DESENHO
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  LEITURA: GOOGLE SHEETS / UPLOAD CSV
# ─────────────────────────────────────────────────────────────
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
    col_nome = find_col_exact("Nome do reservatório") or (cols[1] if len(cols) > 1 else cols[0])
    col_var_m = find_col_exact("Variação em m")
    col_var_m3 = find_col_exact("Variação em m³") or find_col_exact("Variação em m3")
    col_vol = find_col_exact("Volume atual")
    col_pct = find_col_exact("Percentual atual")

    col_lvl_ant = cols[4] if len(cols) > 4 else None
    col_lvl_atu = cols[5] if len(cols) > 5 else None

    df = pd.DataFrame({
        "gerencia": df_raw[col_ger].astype(str).str.strip() if col_ger else "N/A",
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
        df["nome"].notna() &
        (df["nome"].astype(str).str.strip() != "") &
        (~df["nome"].astype(str).str.lower().isin(["nan", "none", "n/a"]))
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
    if "gerencia" not in df.columns:
        return "Fonte: N/A"
    uniques = [g for g in df["gerencia"].dropna().astype(str).str.strip().unique().tolist() if g]
    if not uniques:
        return "Fonte: N/A"
    if len(uniques) <= 3:
        return "Fonte: " + " • ".join(uniques)
    return "Fonte: " + " • ".join(uniques[:3]) + f" • +{len(uniques) - 3}"


# ─────────────────────────────────────────────────────────────
#  IMAGEM FINAL (18 cards)
# ─────────────────────────────────────────────────────────────
def generate_image_layout(df_all: pd.DataFrame, titulo: str, date_anterior: str, date_atual: str,
                          ordenar: str, formato: str) -> Image.Image:
    try:
        base = Image.open(BASE_LAYOUT_PATH).convert("RGBA")
    except Exception:
        base = Image.new("RGBA", (1080, 1350), (255, 255, 255, 255))

    W, H = base.size
    img = base.copy()
    draw = ImageDraw.Draw(img)

    dark = (15, 23, 42)
    gray = (71, 85, 105)

    green_bg = (220, 252, 231, 255)
    green_bd = (16, 185, 129, 255)
    green_tx = (5, 150, 105, 255)

    red_bg = (255, 241, 242, 255)
    red_bd = (251, 113, 133, 255)
    red_tx = (225, 29, 72, 255)

    neutral_bg = (241, 245, 249, 255)
    neutral_bd = (148, 163, 184, 255)
    neutral_tx = (51, 65, 85, 255)

    f_title = get_font(58, True)
    f_sub = get_font(28, False)
    f_legend = get_font(24, True)

    f_name_pos = get_font(18, True)
    f_line_pos = get_font(15, False)
    f_var_pos = get_font(18, True)

    f_name_neg = get_font(16, True)
    f_line_neg = get_font(13, False)
    f_var_neg = get_font(16, True)

    pad = 70
    y = 70
    draw.text((pad, y), titulo, fill=dark, font=f_title)
    y += 80

    period_text = ""
    if str(date_anterior).strip() and str(date_atual).strip():
        period_text = f"Comparativo {date_anterior} até {date_atual}"
    elif str(date_atual).strip():
        period_text = f"Data de referência: {date_atual}"

    if period_text:
        draw.text((pad, y), period_text, fill=gray, font=f_sub)
        y += 52
    else:
        y += 12

    chip_y = y + 8
    chip_h = 40

    draw_rounded_rect(draw, pad, chip_y, 230, chip_h, 18, fill=green_bg, outline=green_bd, width=2)
    draw_arrow(draw, pad + 16, chip_y + 8, True, 22, green_tx)
    draw.text((pad + 46, chip_y + 6), "Subiu", fill=green_tx, font=f_legend)

    draw_rounded_rect(draw, pad + 245, chip_y, 230, chip_h, 18, fill=red_bg, outline=red_bd, width=2)
    draw_arrow(draw, pad + 261, chip_y + 8, False, 22, red_tx)
    draw.text((pad + 291, chip_y + 6), "Desceu", fill=red_tx, font=f_legend)

    y = chip_y + chip_h + 18
    draw.line((pad, y, W - pad, y), fill=(226, 232, 240, 255), width=3)
    y += 26

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

    cols_grid = 3
    rows_grid = 6
    gap_x = 18
    gap_y = 16

    grid_x = pad
    grid_y = y
    grid_w = W - 2 * pad
    grid_h = H - grid_y - 95

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
            up = True
            f_name, f_line, f_var = f_name_pos, f_line_pos, f_var_pos
        elif is_neg:
            bg, bd, tx = red_bg, red_bd, red_tx
            up = False
            f_name, f_line, f_var = f_name_neg, f_line_neg, f_var_neg
        else:
            bg, bd, tx = neutral_bg, neutral_bd, neutral_tx
            up = True
            f_name, f_line, f_var = f_name_neg, f_line_neg, f_var_neg

        draw_rounded_rect(draw, x, y, card_w, card_h, 22, fill=bg, outline=bd, width=2)

        rank_w = 44
        draw_rounded_rect(draw, x + card_w - rank_w - 10, y + 10, rank_w, 30, 14, fill=bd, outline=None, width=0)
        draw.text((x + card_w - 10 - rank_w / 2, y + 25), str(ix + 1),
                  fill=(255, 255, 255), font=get_font(16, True), anchor="mm")

        nome_show = nome.upper()
        if len(nome_show) > 18:
            nome_show = nome_show[:18] + "…"
        draw.text((x + 14, y + 12), nome_show, fill=(15, 23, 42), font=f_name)

        arrow_x = x + 14
        arrow_y = y + 42
        draw_arrow(draw, arrow_x, arrow_y, up, 20, tx)

        # Variação em m (ponto e 2 casas)
        var_txt = "N/A" if pd.isna(var_m) else f"{'+' if float(var_m) > 0 else ''}{fmt_m_2dp_dot(var_m)}"
        draw.text((x + 40, y + 40), var_txt, fill=tx, font=f_var)

        # Variação em m³ e Volume atual em milhões/m³
        l1 = f"Var. m³: {fmt_milhoes_br(var_m3)}"
        l2 = f"Vol: {fmt_milhoes_br(vol)}"
        l3 = f"%: {fmt_pct_br(pct)}"

        draw.text((x + 14, y + 68), l1, fill=(51, 65, 85), font=f_line)
        draw.text((x + 14, y + 88), l2, fill=(51, 65, 85), font=f_line)
        draw.text((x + 14, y + 108), l3, fill=(51, 65, 85), font=f_line)

    for i in range(min(18, len(ordered))):
        ri = i // cols_grid
        ci = i % cols_grid
        cx = grid_x + ci * (card_w + gap_x)
        cy = grid_y + ri * (card_h + gap_y)
        draw_item(i, ordered.iloc[i], cx, cy)

    fonte_txt = build_fonte_gerencia(df_all)

    foot_y = H - 70
    draw.line((pad, foot_y - 18, W - pad, foot_y - 18), fill=(226, 232, 240, 255), width=2)
    f_foot = get_font(22, False)
    draw.text((pad, foot_y), fonte_txt, fill=(100, 116, 139), font=f_foot)

    ts = datetime.now(TZ_FORTALEZA).strftime("%d/%m/%Y %H:%M")
    draw.text((W - pad, foot_y), f"Gerado em {ts}", fill=(100, 116, 139), font=f_foot, anchor="ra")

    return img.convert("RGB") if formato.upper() == "JPG" else img


# ─────────────────────────────────────────────────────────────
#  STREAMLIT
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Reservatórios — Card Generator",
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
    st.caption("Formato final: variação m com ponto. Variação m³ e Volume em milhões/m³.")
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

        titulo_custom = st.text_input("📝 Título", value="Monitoramento dos Reservatórios")
        ordenar = st.selectbox(
            "Ordenação",
            ["Manter ordem", "Maior variação absoluta", "Maior variação positiva", "Maior variação negativa"],
            index=0
        )
        formato = st.selectbox("🖼️ Formato de saída", ["PNG", "JPG"])
        debug = st.toggle("🔍 Mostrar prévia do CSV", value=False)

        st.divider()
        if st.button("🔄 Atualizar dados", use_container_width=True):
            load_csv_from_url.clear()
            st.rerun()

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
    pos = int((df_proc["variacao_m"] > 0).sum()) if "variacao_m" in df_proc.columns else 0
    neg = int((df_proc["variacao_m"] < 0).sum()) if "variacao_m" in df_proc.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("💾 Total", f"{total}")
    c2.metric("📈 Subiram", f"{pos}")
    c3.metric("📉 Desceram", f"{neg}")

    if debug:
        st.subheader("Prévia processada")
        st.dataframe(df_proc.head(30), use_container_width=True)

    st.divider()

    if st.button("🎨 Gerar imagem do card", type="primary", use_container_width=True):
        if df_proc.empty:
            st.warning("Sem dados para renderizar.")
            return

        d_ant = info.get("periodo", {}).get("anterior", "")
        d_atu = info.get("periodo", {}).get("atual", "")

        with st.spinner("Renderizando..."):
            img_final = generate_image_layout(
                df_all=df_proc,
                titulo=titulo_custom,
                date_anterior=d_ant,
                date_atual=d_atu,
                ordenar=ordenar,
                formato=formato
            )

        st.image(img_final, caption="Preview. 1080×1350 px", use_container_width=True)

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
        fname = f"monitoramento_reservatorios_{ts_name}.{formato.lower()}"
        st.download_button(
            label=f"📥 Baixar ({formato})",
            data=buf,
            file_name=fname,
            mime=mime,
            use_container_width=True
        )

        st.success("Pronto. Unidades ajustadas do jeito que tu pediu.")

    st.caption("Coloque base_card.png na mesma pasta do app.py para manter o layout.")


if __name__ == "__main__":
    main()
