# =============================================================
#  app.py  |  Monitoramento de Reservatórios — Card Generator
#  GF Informática  |  Paulo Ferreira
# =============================================================

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import math


# ─────────────────────────────────────────────────────────────
#  UTILITÁRIOS DE FONTE
# ─────────────────────────────────────────────────────────────

def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Carrega fonte TTF com fallback para múltiplos sistemas operacionais."""
    paths_bold = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    ]
    paths_regular = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/TTF/DejaVuSans.ttf',
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
#  FORMATAÇÃO DE NÚMEROS (PT-BR)
# ─────────────────────────────────────────────────────────────

def format_number(value, decimals: int = 0,
                  prefix: str = '', suffix: str = '') -> str:
    """Formata número no padrão brasileiro: 1.234.567,89"""
    try:
        if pd.isna(value):
            return 'N/A'
        val = float(value)
        if decimals > 0:
            formatted = (f"{val:,.{decimals}f}"
                         .replace(',', 'X')
                         .replace('.', ',')
                         .replace('X', '.'))
        else:
            formatted = f"{int(val):,}".replace(',', '.')
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError):
        return str(value)


# ─────────────────────────────────────────────────────────────
#  RETÂNGULO ARREDONDADO COM ANTI-ALIASING
# ─────────────────────────────────────────────────────────────

def draw_rounded_rect(base_img: Image.Image,
                      x: int, y: int, width: int, height: int,
                      radius: int, fill_color: tuple,
                      border_color: tuple = None,
                      border_width: int = 2) -> Image.Image:
    """Desenha retângulo arredondado via máscara 4x com LANCZOS."""
    if base_img.mode != 'RGBA':
        base_img = base_img.convert('RGBA')

    scale = 4
    mask = Image.new('L', (width * scale, height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, width * scale - 1, height * scale - 1],
        radius=radius * scale, fill=255
    )
    mask = mask.resize((width, height), Image.LANCZOS)

    card_layer = Image.new('RGBA', (width, height), fill_color)
    base_img.paste(card_layer, (x, y), mask)

    if border_color:
        border_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(border_layer).rounded_rectangle(
            [0, 0, width - 1, height - 1],
            radius=radius, outline=border_color, width=border_width
        )
        base_img.paste(border_layer, (x, y), border_layer)

    return base_img


# ─────────────────────────────────────────────────────────────
#  CARREGAMENTO DOS DADOS DO GOOGLE SHEETS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data(url: str):
    """
    Lê planilha do Google Sheets via exportação CSV.
    Mapeamento de colunas por posição (robusto a mudanças de cabeçalho):
      A(0)=nome  B(1)=capacidade  C(2)=volume_atual_m3  D(3)=percentual
      E(4)=data_anterior  F(5)=data_atual  G(6)=variacao_m  H(7)=variacao_m3
    """
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df_raw = pd.read_csv(BytesIO(resp.content))

        num_cols = df_raw.shape[1]

        def safe_col(idx):
            """Retorna coluna pelo índice ou série de N/A se não existir."""
            if idx < num_cols:
                return df_raw.iloc[:, idx]
            return pd.Series(['N/A'] * len(df_raw))

        def to_num(series):
            return pd.to_numeric(
                series.astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            )

        df = pd.DataFrame({
            'nome':           safe_col(0).astype(str).str.strip(),
            'capacidade':     to_num(safe_col(1)),
            'volume_atual_m3':to_num(safe_col(2)),
            'percentual':     to_num(safe_col(3)),
            'data_anterior':  safe_col(4).astype(str).str.strip(),
            'data_atual':     safe_col(5).astype(str).str.strip(),
            'variacao_m':     to_num(safe_col(6)),
            'variacao_m3':    to_num(safe_col(7)),
        })

        # Remove linhas sem nome válido
        df = df[
            df['nome'].notna() &
            (df['nome'] != '') &
            (~df['nome'].str.lower().isin(['nan', 'none', 'n/a']))
        ].reset_index(drop=True)

        return df_raw, df, {'colunas': list(df_raw.columns),
                            'shape': df_raw.shape}

    except Exception as err:
        st.warning(f"⚠️ Planilha indisponível: {err} — exibindo dados de exemplo.")
        mock = pd.DataFrame({
            'nome':           ['Açude Castanhão', 'Açude Orós',
                               'Barragem Banabuiú', 'Açude Araras'],
            'capacidade':     [6_700_000_000, 1_940_000_000,
                               1_601_000_000,   891_000_000],
            'volume_atual_m3':[2_100_000_000,   486_000_000,
                                 640_000_000,   445_000_000],
            'percentual':     [31.3, 25.0, 40.0, 49.9],
            'data_anterior':  ['10/06/2025'] * 4,
            'data_atual':     ['11/06/2025'] * 4,
            'variacao_m':     [0.85, 0.42, 0.31, 1.10],
            'variacao_m3':    [310_000, 52_000, 28_000, 95_000],
        })
        return pd.DataFrame(), mock, {'erro': str(err)}


# ─────────────────────────────────────────────────────────────
#  CARD INDIVIDUAL
# ─────────────────────────────────────────────────────────────

def draw_card(img: Image.Image,
              x: int, y: int, w: int, h: int,
              data: pd.Series, rank: int) -> Image.Image:
    """Desenha um card verde com todas as informações do reservatório."""

    # Fundo e borda do card
    img = draw_rounded_rect(
        img, x, y, w, h,
        radius=14,
        fill_color=(0, 70, 35, 215),
        border_color=(0, 200, 83, 255),
        border_width=2
    )
    # Glow externo sutil
    img = draw_rounded_rect(
        img, x - 2, y - 2, w + 4, h + 4,
        radius=16,
        fill_color=(0, 0, 0, 0),
        border_color=(105, 240, 174, 45),
        border_width=5
    )

    draw = ImageDraw.Draw(img)

    # ── Badge de ranking ──────────────────────────────────
    br = 22
    bx = x + w - br - 12
    by = y + 12
    draw.ellipse([bx, by, bx + br * 2, by + br * 2], fill=(0, 200, 83))
    draw.text((bx + br, by + br), str(rank),
              font=get_font(18, bold=True), fill=(255, 255, 255), anchor='mm')

    # ── Fontes ────────────────────────────────────────────
    f_nome  = get_font(20, bold=True)
    f_label = get_font(15)
    f_valor = get_font(18, bold=True)
    f_var   = get_font(26, bold=True)
    f_pct   = get_font(20, bold=True)

    px = x + 16
    py = y + 14

    # ── Nome ──────────────────────────────────────────────
    nome = str(data.get('nome', 'N/A')).upper()
    nome = (nome[:21] + '…') if len(nome) > 21 else nome
    draw.text((px, py), nome, font=f_nome, fill=(255, 255, 255))

    # Linha divisória
    ly = py + 28
    draw.line([(px, ly), (x + w - 16, ly)], fill=(0, 200, 83), width=1)

    # ── Variação em metros (destaque) ─────────────────────
    var_m   = float(data.get('variacao_m', 0) or 0)
    positiv = var_m >= 0
    cor_var = (105, 240, 174) if positiv else (244, 100, 100)
    seta    = '▲' if positiv else '▼'
    sinal   = '+' if positiv else ''

    vy = ly + 10
    draw.text((px, vy),       seta,                          font=f_var, fill=cor_var)
    draw.text((px + 30, vy),  f"{sinal}{format_number(var_m, 2)} m",
              font=f_var, fill=cor_var)
    draw.text((px, vy + 32),  "Variação do nível",
              font=f_label, fill=(165, 214, 167))

    # ── Variação em m³ ────────────────────────────────────
    l2y    = vy + 54
    var_m3 = float(data.get('variacao_m3', 0) or 0)
    s3     = '+' if var_m3 >= 0 else ''
    draw.text((px, l2y),      "Var. Volume:", font=f_label, fill=(144, 164, 174))
    draw.text((px, l2y + 17), f"{s3}{format_number(var_m3, 0)} m³",
              font=f_valor, fill=(224, 224, 224))

    # ── Volume atual ──────────────────────────────────────
    l3y = l2y + 40
    draw.text((px, l3y),      "Volume Atual:", font=f_label, fill=(144, 164, 174))
    draw.text((px, l3y + 17),
              f"{format_number(data.get('volume_atual_m3', 0), 0)} m³",
              font=f_valor, fill=(224, 224, 224))

    # ── Percentual + barra ────────────────────────────────
    l4y = l3y + 44
    draw.text((px, l4y), "Capacidade:", font=f_label, fill=(144, 164, 174))

    pct = max(0.0, min(100.0, float(data.get('percentual', 0) or 0)))
    draw.text((x + w - 16, l4y), f"{format_number(pct, 1)}%",
              font=f_pct, fill=(105, 240, 174), anchor='ra')

    bx2  = px
    by2  = l4y + 20
    bw2  = w - 32
    bh2  = 8
    draw.rounded_rectangle([bx2, by2, bx2 + bw2, by2 + bh2],
                            radius=4, fill=(20, 45, 30))
    fw = max(0, int(bw2 * pct / 100))
    if fw > 0:
        draw.rounded_rectangle([bx2, by2, bx2 + fw, by2 + bh2],
                                radius=4, fill=(0, 200, 83))

    return img


# ─────────────────────────────────────────────────────────────
#  GERAÇÃO DA IMAGEM PRINCIPAL (1080 × 1080)
# ─────────────────────────────────────────────────────────────

def generate_image(df_top: pd.DataFrame,
                   date_anterior: str, date_atual: str,
                   formato: str, titulo_custom: str) -> Image.Image:
    """Gera card 1080×1080 com fundo gradiente e cards dos reservatórios."""
    W, H = 1080, 1080

    # ── Gradiente de fundo ────────────────────────────────
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for row in range(H):
        t = row / H
        arr[row] = [
            int(13 + (22 - 13) * t),
            int(27 + (38 - 27) * t),
            int(42 + (58 - 42) * t),
        ]
    img = Image.fromarray(arr, 'RGB').convert('RGBA')

    # ── Overlay com glows nos cantos ─────────────────────
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for cx, cy, r, alpha in [
        (0,    0,    350, 20),
        (W,    0,    280, 14),
        (0,    H,    220, 12),
        (W//2, H//2, 520, 7),
    ]:
        od.ellipse([cx - r, cy - r, cx + r, cy + r],
                   fill=(79, 195, 247, alpha))
    img = Image.alpha_composite(img, ov)
    draw = ImageDraw.Draw(img)

    # ── Ícone gota ────────────────────────────────────────
    draw.ellipse([52, 52, 104, 104], fill=(79, 195, 247))
    draw.polygon([(78, 34), (52, 68), (104, 68)], fill=(79, 195, 247))

    # ── Título em 2 linhas ────────────────────────────────
    palavras = titulo_custom.strip().split()
    if len(palavras) >= 2:
        l1 = ' '.join(palavras[:2])
        l2 = ' '.join(palavras[2:])
    else:
        l1, l2 = titulo_custom, ''

    draw.text((124, 42), l1,
              font=get_font(44, bold=True), fill=(255, 255, 255))
    if l2:
        draw.text((124, 90), l2,
                  font=get_font(52, bold=True), fill=(79, 195, 247))

    # ── Subtítulo de período ──────────────────────────────
    sub_y = 162 if l2 else 102
    draw.text((55, sub_y),
              f"Período de referência:  {date_anterior}  →  {date_atual}",
              font=get_font(26), fill=(144, 164, 174))

    # ── Linha divisória verde ─────────────────────────────
    line_y = sub_y + 40
    draw.rectangle([55, line_y, W - 55, line_y + 4], fill=(0, 200, 83))

    # ── Layout dinâmico de cards ──────────────────────────
    start_y  = line_y + 18
    margin   = 55
    gap      = 14
    num      = min(len(df_top), 6)

    if num <= 1:
        cols = 1
    elif num <= 4:
        cols = 2
    else:
        cols = 3

    rows   = math.ceil(num / cols)
    avail_h = 985 - start_y
    card_w  = (W - 2 * margin - (cols - 1) * gap) // cols
    card_h  = (avail_h - (rows - 1) * gap) // rows

    for i, (_, row) in enumerate(df_top.head(num).iterrows()):
        ci = i % cols
        ri = i // cols
        cx = margin + ci * (card_w + gap)
        cy = start_y + ri * (card_h + gap)
        img = draw_card(img, cx, cy, card_w, card_h, row, i + 1)

    # ── Rodapé ────────────────────────────────────────────
    draw = ImageDraw.Draw(img)
    draw.line([(55, 1018), (W - 55, 1018)], fill=(50, 70, 80), width=1)
    f_foot = get_font(21)
    draw.text((55,     1024), "📊 COGERH / SRH-CE",
              font=f_foot, fill=(84, 110, 122))
    ts = datetime.now().strftime('%d/%m/%Y às %H:%M')
    draw.text((W - 55, 1024), f"Gerado em {ts}",
              font=f_foot, fill=(84, 110, 122), anchor='ra')

    return img.convert('RGB') if formato.upper() == 'JPG' else img


# ─────────────────────────────────────────────────────────────
#  APLICATIVO STREAMLIT
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Reservatórios — Card Generator",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .stApp               { background-color: #0a1628; color: #e0e0e0; }
        section[data-testid="stSidebar"] { background-color: #0d1f35 !important; }
        h1, h2, h3           { color: #4FC3F7 !important; }
        div[data-testid="metric-container"] {
            background: rgba(0,200,83,.08);
            border: 1px solid rgba(0,200,83,.28);
            border-radius: 10px; padding: 8px;
        }
        .stButton > button   { background:#00C853; color:#fff;
                               border-radius:8px; font-weight:700; border:none; }
        .stButton > button:hover { background:#00a844; }
        .stDownloadButton > button { background:#1565C0; color:#fff;
                                     border-radius:8px; font-weight:700; border:none; }
    </style>
    """, unsafe_allow_html=True)

    SHEET_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1fbaYqjee8h4dAA8ew0RXbHOKdnSDoHIB2xPpdveYMDU"
        "/export?format=csv&gid=0"
    )

    # ── Carregar dados ────────────────────────────────────
    df_raw, df_proc, info = load_data(SHEET_URL)

    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        st.divider()

        titulo_custom = st.text_input(
            "📝 Título",
            value="Monitoramento dos Reservatórios"
        )
        min_var = st.slider(
            "📏 Variação mínima (m)",
            min_value=0.0, max_value=5.0, value=0.1, step=0.05,
            help="Exibe apenas reservatórios com variação ≥ este valor"
        )
        max_res = st.slider(
            "🔢 Máx. reservatórios no card",
            min_value=1, max_value=6, value=4
        )
        formato = st.selectbox("🖼️ Formato de saída", ["PNG", "JPG"])
        st.divider()
        debug = st.toggle("🔍 Dados brutos", value=False)
        if st.button("🔄 Atualizar planilha", use_container_width=True):
            load_data.clear()
            st.rerun()
        st.divider()
        st.caption("GF Informática · Paulo Ferreira")

    # ── CABEÇALHO ─────────────────────────────────────────
    st.title("💧 Gerador de Cards — Monitoramento de Reservatórios")
    st.caption("Lê dados em tempo real do Google Sheets e gera imagens "
               "1080×1080 prontas para redes sociais.")
    st.divider()

    # ── FILTRO ────────────────────────────────────────────
    df_filtrado = (
        df_proc[df_proc['variacao_m'] >= min_var]
        .sort_values('variacao_m', ascending=False)
        .head(max_res)
        .reset_index(drop=True)
    )

    # ── MÉTRICAS ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💾 Total carregado",   f"{len(df_proc)} reservatórios")
    c2.metric("📈 Variação positiva", f"{len(df_proc[df_proc['variacao_m'] > 0])}")
    c3.metric("🏆 Maior variação",    f"{format_number(df_proc['variacao_m'].max(), 2)} m")
    c4.metric("🎯 No card",           f"{len(df_filtrado)}")

    # ── DEBUG ─────────────────────────────────────────────
    if debug:
        with st.expander("📋 Colunas detectadas"):
            st.write(info.get('colunas', []))
            if not df_raw.empty:
                st.dataframe(df_raw.head(10), use_container_width=True)
        with st.expander("📊 Dados filtrados"):
            st.dataframe(df_filtrado, use_container_width=True)

    st.divider()

    # ── BOTÃO GERAR ───────────────────────────────────────
    if st.button("🎨 Gerar Imagem para Redes Sociais",
                 type="primary", use_container_width=True):

        if df_filtrado.empty:
            st.warning("⚠️ Nenhum reservatório encontrado. "
                       "Reduza a variação mínima na barra lateral.")
        else:
            with st.spinner("🎨 Renderizando imagem..."):
                d_ant = df_filtrado['data_anterior'].iloc[0]
                d_atu = df_filtrado['data_atual'].iloc[0]
                img_final = generate_image(
                    df_top        = df_filtrado,
                    date_anterior = d_ant,
                    date_atual    = d_atu,
                    formato       = formato,
                    titulo_custom = titulo_custom,
                )

            # Preview
            st.image(img_final,
                     caption="Preview — 1080×1080 px",
                     use_container_width=True)

            # Download
            buf = BytesIO()
            mime_fmt = 'JPEG' if formato.upper() == 'JPG' else 'PNG'
            img_final.save(buf, format=mime_fmt, quality=95)
            buf.seek(0)

            fname = (f"reservatorios_"
                     f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                     f".{formato.lower()}")

            st.download_button(
                label          = f"📥 Baixar Imagem ({formato})",
                data           = buf,
                file_name      = fname,
                mime           = f"image/{'jpeg' if formato=='JPG' else 'png'}",
                use_container_width = True,
            )
            st.success(f"✅ Pronto! {len(df_filtrado)} reservatório(s) exibido(s).")


if __name__ == '__main__':
    main()
