# =============================================================
#  app.py  |  Monitoramento de Reservatórios — Card Generator
#  GF Informática  |  Paulo Ferreira
#  Atualizado: layout por base_card.png, 18 reservatórios,
#  positivos e negativos, upload CSV
# =============================================================

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import math
import re

# Caminho da imagem de layout (mesmo diretório do app.py)
BASE_LAYOUT_PATH = "base_card.png"

# Link padrão do Google Sheets
DEFAULT_SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/"
    "1fbaYqjee8h4dAA8ew0RXbHOKdnSDoHIB2xPpdveYMDU"
    "/export?format=csv&gid=0"
)


# ─────────────────────────────────────────────────────────────
#  UTILITÁRIOS DE FONTE
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
#  FORMATAÇÃO PT-BR
# ─────────────────────────────────────────────────────────────

def format_number(value, decimals: int = 0, prefix: str = "", suffix: str = "") -> str:
    try:
        if pd.isna(value):
            return "N/A"
        val = float(value)
        if decimals > 0:
            formatted = (
                f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
        else:
            formatted = f"{int(round(val)):,.0f}".replace(",", ".")
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError):
        return str(value)


def to_num_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("m³", "", regex=False).str.replace("m3", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(" ", "", regex=False)

    # se vier 1.234,56 (BR), vira 1234.56
    # remove milhar e troca decimal
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)

    # remove lixo
    s = s.str.replace(r"[^0-9\.\-\+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


# ─────────────────────────────────────────────────────────────
#  DESENHO: RETÂNGULO ARREDONDADO + SETAS
# ─────────────────────────────────────────────────────────────

def draw_rounded_rect(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                      r: int, fill, outline=None, width: int = 2):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, x: int, y: int, up: bool, size: int, color):
    # seta simples (triângulo + haste)
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
#  LEITURA: GOOGLE SHEETS OU CSV UPLOAD
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
    """
    Cabeçalho esperado:
    Gerência | Nome do reservatório | Capacidade (hm³) | Cota Sangria |
    09/02/2026 | 24/02/26 | Variação em m | Variação em m³ | Volume atual | Percentual atual
    """
    cols = list(df_raw.columns)

    def find_col_exact(name: str):
        for c in cols:
            if str(c).strip().lower() == name.strip().lower():
                return c
        return None

    # Período pelas colunas E e F (índices 4 e 5)
    date_anterior = cols[4] if len(cols) > 4 else ""
    date_atual = cols[5] if len(cols) > 5 else ""

    col_nome = find_col_exact("Nome do reservatório") or (cols[1] if len(cols) > 1 else cols[0])
    col_var_m = find_col_exact("Variação em m")
    col_var_m3 = find_col_exact("Variação em m³") or find_col_exact("Variação em m3")
    col_vol = find_col_exact("Volume atual")
    col_pct = find_col_exact("Percentual atual")

    # níveis nas datas (se existirem e forem numéricos)
    col_lvl_ant = cols[4] if len(cols) > 4 else None
    col_lvl_atu = cols[5] if len(cols) > 5 else None

    df = pd.DataFrame({
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

    # limpa linhas inválidas
    df = df[
        df["nome"].notna() &
        (df["nome"].astype(str).str.strip() != "") &
        (~df["nome"].astype(str).str.lower().isin(["nan", "none", "n/a"]))
    ].reset_index(drop=True)

    # se variacao_m vier toda vazia, tenta calcular: nivel_atual - nivel_anterior
    if df["variacao_m"].isna().all():
        if ("nivel_anterior" in df.columns) and ("nivel_atual" in df.columns):
            df["variacao_m"] = (df["nivel_atual"] - df["nivel_anterior"]).round(2)

    # garante numéricos
    for c in ["variacao_m", "variacao_m3", "volume_atual_m3", "percentual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    info = {
        "colunas": cols,
        "shape": df_raw.shape,
        "periodo": {"anterior": date_anterior, "atual": date_atual},
    }
    return df, info


# ─────────────────────────────────────────────────────────────
#  GERAÇÃO DA IMAGEM (BASE + GRID 18)
# ─────────────────────────────────────────────────────────────

def generate_image_layout(
    df_all: pd.DataFrame,
    titulo: str,
    date_anterior: str,
    date_atual: str,
    ordenar: str,
    formato: str,
) -> Image.Image:
    # base layout 1080x1350
    try:
        base = Image.open(BASE_LAYOUT_PATH).convert("RGBA")
    except Exception:
        base = Image.new("RGBA", (1080, 1350), (255, 255, 255, 255))

    W, H = base.size
    img = base.copy()
    draw = ImageDraw.Draw(img)

    # cores
    dark = (15, 23, 42)
    gray = (71, 85, 105)

    green_bg = (220, 252, 231, 255)
    green_bd = (16, 185, 129, 255)
    green_tx = (5, 150, 105, 255)

    red_bg = (254, 226, 226, 255)
    red_bd = (239, 68, 68, 255)
    red_tx = (220, 38, 38, 255)

    neutral_bg = (241, 245, 249, 255)
    neutral_bd = (148, 163, 184, 255)
    neutral_tx = (51, 65, 85, 255)

    # fontes
    f_title = get_font(58, bold=True)
    f_sub = get_font(28, bold=False)
    f_legend = get_font(24, bold=True)

    f_name = get_font(18, bold=True)
    f_line = get_font(15, bold=False)
    f_var = get_font(18, bold=True)

    # header
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

    # legenda de cores
    chip_y = y + 8
    chip_h = 40

    # chip verde
    draw_rounded_rect(draw, pad, chip_y, 230, chip_h, 18, fill=green_bg, outline=green_bd, width=2)
    draw_arrow(draw, pad + 16, chip_y + 8, True, 22, green_tx)
    draw.text((pad + 46, chip_y + 6), "Subiu", fill=green_tx, font=f_legend)

    # chip vermelho
    draw_rounded_rect(draw, pad + 245, chip_y, 230, chip_h, 18, fill=red_bg, outline=red_bd, width=2)
    draw_arrow(draw, pad + 261, chip_y + 8, False, 22, red_tx)
    draw.text((pad + 291, chip_y + 6), "Desceu", fill=red_tx, font=f_legend)

    # linha
    y = chip_y + chip_h + 18
    draw.line((pad, y, W - pad, y), fill=(226, 232, 240, 255), width=3)
    y += 26

    # ordenação
    df = df_all.copy()

    if ordenar == "Maior variação positiva":
        df = df.sort_values("variacao_m", ascending=False)
    elif ordenar == "Maior variação negativa":
        df = df.sort_values("variacao_m", ascending=True)
    elif ordenar == "Maior variação absoluta":
        df = df.assign(_abs=df["variacao_m"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
    else:
        # manter ordem
        pass

    # garante 18
    df = df.head(18).reset_index(drop=True)

    # grid 3 x 6 = 18
    cols = 3
    rows = 6
    gap_x = 18
    gap_y = 16

    grid_x = pad
    grid_y = y
    grid_w = W - 2 * pad
    grid_h = H - grid_y - 95

    card_w = int((grid_w - (cols - 1) * gap_x) / cols)
    card_h = int((grid_h - (rows - 1) * gap_y) / rows)

    # mini-card render
    def draw_item(ix: int, row: pd.Series, x: int, y: int):
        nome = str(row.get("nome", "N/A")).strip()
        var_m = row.get("variacao_m", None)
        var_m3 = row.get("variacao_m3", None)
        vol = row.get("volume_atual_m3", None)
        pct = row.get("percentual", None)

        # define cor
        if pd.isna(var_m):
            bg, bd, tx = neutral_bg, neutral_bd, neutral_tx
            up = True
        else:
            if float(var_m) > 0:
                bg, bd, tx = green_bg, green_bd, green_tx
                up = True
            elif float(var_m) < 0:
                bg, bd, tx = red_bg, red_bd, red_tx
                up = False
            else:
                bg, bd, tx = neutral_bg, neutral_bd, neutral_tx
                up = True

        draw_rounded_rect(draw, x, y, card_w, card_h, 22, fill=bg, outline=bd, width=2)

        # rank pequeno
        rank_w = 44
        draw_rounded_rect(draw, x + card_w - rank_w - 10, y + 10, rank_w, 30, 14, fill=bd, outline=None, width=0)
        draw.text((x + card_w - 10 - rank_w / 2, y + 25), str(ix + 1), fill=(255, 255, 255), font=get_font(16, True), anchor="mm")

        # nome
        nome_show = nome.upper()
        if len(nome_show) > 18:
            nome_show = nome_show[:18] + "…"
        draw.text((x + 14, y + 12), nome_show, fill=(15, 23, 42), font=f_name)

        # var m com seta
        arrow_x = x + 14
        arrow_y = y + 42
        draw_arrow(draw, arrow_x, arrow_y, up, 20, tx)

        sinal = ""
        if not pd.isna(var_m) and float(var_m) > 0:
            sinal = "+"

        var_txt = "N/A" if pd.isna(var_m) else f"{sinal}{format_number(var_m, 2)} m"
        draw.text((x + 40, y + 40), var_txt, fill=tx, font=f_var)

        # linhas compactas
        l1 = f"Var. m³: {'N/A' if pd.isna(var_m3) else format_number(var_m3, 0)}"
        l2 = f"Vol: {'N/A' if pd.isna(vol) else format_number(vol, 0)} m³"
        l3 = f"%: {'N/A' if pd.isna(pct) else format_number(pct, 1)}"

        draw.text((x + 14, y + 68), l1, fill=(51, 65, 85), font=f_line)
        draw.text((x + 14, y + 88), l2, fill=(51, 65, 85), font=f_line)
        draw.text((x + 14, y + 108), l3, fill=(51, 65, 85), font=f_line)

    # desenha os 18
    for i in range(min(18, len(df))):
        ri = i // cols
        ci = i % cols
        cx = grid_x + ci * (card_w + gap_x)
        cy = grid_y + ri * (card_h + gap_y)
        draw_item(i, df.iloc[i], cx, cy)

    # rodapé
    foot_y = H - 70
    draw.line((pad, foot_y - 18, W - pad, foot_y - 18), fill=(226, 232, 240, 255), width=2)
    f_foot = get_font(22, False)
    draw.text((pad, foot_y), "Fonte: Monitoramento dos reservatórios", fill=(100, 116, 139), font=f_foot)
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")
    draw.text((W - pad, foot_y), f"Gerado em {ts}", fill=(100, 116, 139), font=f_foot, anchor="ra")

    # saída
    if formato.upper() == "JPG":
        return img.convert("RGB")
    return img


# ─────────────────────────────────────────────────────────────
#  STREAMLIT APP
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
    st.caption("Gera imagem 1080×1350 com 18 reservatórios, positivos e negativos. Layout baseado no arquivo base_card.png.")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        st.divider()

        fonte = st.radio("Fonte de dados", ["Google Sheets", "Upload CSV"], index=0)

        uploaded = None
        sheet_url = DEFAULT_SHEET_CSV

        if fonte == "Upload CSV":
            uploaded = st.file_uploader("Envie o .csv", type=["csv"])
            st.caption("Dica: mantenha as colunas iguais à planilha oficial.")
        else:
            sheet_url = st.text_input("Link CSV do Google Sheets", value=DEFAULT_SHEET_CSV)

        st.divider()

        titulo_custom = st.text_input("📝 Título", value="Monitoramento dos Reservatórios")

        ordenar = st.selectbox(
            "Ordenação",
            ["Manter ordem", "Maior variação absoluta", "Maior variação positiva", "Maior variação negativa"],
            index=1
        )

        formato = st.selectbox("🖼️ Formato de saída", ["PNG", "JPG"])

        debug = st.toggle("🔍 Mostrar prévia do CSV", value=False)

        st.divider()
        if st.button("🔄 Atualizar dados", use_container_width=True):
            load_csv_from_url.clear()
            st.rerun()

        st.caption("GF Informática · Paulo Ferreira")

    # Carregar dados
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

    # métricas
    total = len(df_proc)
    pos = int((df_proc["variacao_m"] > 0).sum()) if "variacao_m" in df_proc.columns else 0
    neg = int((df_proc["variacao_m"] < 0).sum()) if "variacao_m" in df_proc.columns else 0
    maior_pos = df_proc["variacao_m"].max() if "variacao_m" in df_proc.columns else None
    maior_neg = df_proc["variacao_m"].min() if "variacao_m" in df_proc.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💾 Total", f"{total}")
    c2.metric("📈 Subiram", f"{pos}")
    c3.metric("📉 Desceram", f"{neg}")
    c4.metric("🏁 Extremos", f"+{format_number(maior_pos, 2)} m | {format_number(maior_neg, 2)} m")

    if debug:
        st.subheader("Prévia do CSV")
        st.dataframe(df_raw.head(30), use_container_width=True)

    st.divider()

    # Geração
    if st.button("🎨 Gerar imagem do card", type="primary", use_container_width=True):
        if df_proc.empty:
            st.warning("Sem dados para renderizar.")
            return

        # precisa ter pelo menos 18 linhas, se tiver menos, mostra todas mesmo assim
        if len(df_proc) < 18:
            st.warning(f"Seu CSV tem {len(df_proc)} linhas. Vou renderizar mesmo assim, mas o ideal é ter 18 reservatórios.")

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

        fname = f"monitoramento_reservatorios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{formato.lower()}"
        st.download_button(
            label=f"📥 Baixar ({formato})",
            data=buf,
            file_name=fname,
            mime=mime,
            use_container_width=True
        )

        st.success("Pronto. Card gerado com positivos e negativos.")

    st.caption("Obs: coloque o arquivo base_card.png na mesma pasta do app.py para manter o layout de referência.")


if __name__ == "__main__":
    main()
