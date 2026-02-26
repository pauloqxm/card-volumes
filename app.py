import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from datetime import datetime
import math

# 
# UTILITÁRIOS DE FONTE
# 

def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    Tenta carregar fonte TrueType de vários caminhos (Linux, Windows, macOS).
    Fallback para fonte padrão do PIL se nenhuma for encontrada.
    """
    font_paths_bold = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/Arial Bold.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    ]
    font_paths_regular = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/TTF/DejaVuSans.ttf',
    ]
    paths = font_paths_bold if bold else font_paths_regular
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    # Fallback seguro
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# 
# FORMATAÇÃO DE NÚMEROS (PT-BR)
# 

def format_number(value, decimals: int = 0, prefix: str = '', suffix: str = '') -> str:
    """
    Formata número no padrão brasileiro: 1.234.567,89
    """
    try:
        if pd.isna(value):
            return 'N/A'
        val = float(value)
        if decimals > 0:
            formatted = f"{val:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else:
            formatted = f"{int(val):,}".replace(',', '.')
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError):
        return str(value)


# 
# RETÂNGULO ARREDONDADO COM ANTI-ALIASING
# 

def draw_rounded_rect_on_image(
    base_img: Image.Image,
    x: int, y: int,
    width: int, height: int,
    radius: int,
    fill_color: tuple,
    border_color: tuple = None,
    border_width: int = 2
):
    """
    Desenha retângulo arredondado com anti-aliasing via upscale 4x.
    Suporta fill_color com canal alpha (RGBA).
    """
    # Garantir RGBA na imagem base
    if base_img.mode != 'RGBA':
        base_img = base_img.convert('RGBA')

    scale = 4
    # Máscara de alta qualidade
    mask = Image.new('L', (width * scale, height * scale), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        [0, 0, width * scale - 1, height * scale - 1],
        radius=radius * scale, fill=255
    )
    mask = mask.resize((width, height), Image.LANCZOS)

    # Fundo do card
    card_layer = Image.new('RGBA', (width, height), fill_color)
    base_img.paste(card_layer, (x, y), mask)

    # Borda
    if border_color:
        border_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        border_draw = ImageDraw.Draw(border_layer)
        border_draw.rounded_rectangle(
            [0, 0, width - 1, height - 1],
            radius=radius,
            outline=border_color,
            width=border_width
        )
        base_img.paste(border_layer, (x, y), border_layer)


# 
# CARREGAMENTO DOS DADOS DO GOOGLE SHEETS
# 

@st.cache_data(ttl=300)
def load_data(url: str):
    """
    Lê planilha do Google Sheets via URL de exportação CSV.
    Retorna: (df_original, df_processado, info_colunas)
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        df = pd.read_csv(BytesIO(response.content))

        # ── Info de debug ──────────────────────────────
        info = {
            'colunas': list(df.columns),
            'shape': df.shape,
        }

        # ── Mapeamento por posição das colunas ─────────
        # A planilha pode ter linha de cabeçalho variada.
        # Usar .iloc para robustez.
        num_cols = df.shape[1]

        def safe_col(idx):
            return df.iloc[:, idx] if idx &lt; num_cols else pd.Series(['N/A'] * len(df))

        df_proc = pd.DataFrame()
        df_proc['nome']           = safe_col(0).astype(str).str.strip()
        df_proc['capacidade']     = pd.to_numeric(safe_col(1).astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df_proc['volume_atual_m3']= pd.to_numeric(safe_col(2).astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df_proc['percentual']     = pd.to_numeric(safe_col(3).astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df_proc['data_anterior']  = safe_col(4).astype(str).str.strip()
        df_proc['data_atual']     = safe_col(5).astype(str).str.strip()
        df_proc['variacao_m']     = pd.to_numeric(safe_col(6).astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df_proc['variacao_m3']    = pd.to_numeric(safe_col(7).astype(str).str.replace(',', '.', regex=False), errors='coerce')

        # Remover linhas inválidas
        df_proc = df_proc[
            df_proc['nome'].notna() &
            (df_proc['nome'] != '') &
            (df_proc['nome'].str.lower() != 'nan') &
            (df_proc['nome'].str.lower() != 'none')
        ].reset_index(drop=True)

        return df, df_proc, info

    except Exception as e:
        # ── DADOS MOCK para não travar o app ───────────
        st.warning(f"⚠️ Não foi possível carregar a planilha: {e}. Usando dados de exemplo.")
        mock = pd.DataFrame({
            'nome':          ['Açude Orós', 'Açude Castanhão', 'Barragem Banabuiú', 'Açude Araras'],
            'capacidade':    [1940000000, 6700000000, 1601000000, 891000000],
            'volume_atual_m3':[486000000, 2100000000, 640000000, 445000000],
            'percentual':    [25.0, 31.3, 40.0, 49.9],
            'data_anterior': ['10/06/2025'] * 4,
            'data_atual':    ['11/06/2025'] * 4,
            'variacao_m':    [0.42, 0.85, 0.31, 1.10],
            'variacao_m3':   [52000, 310000, 28000, 95000],
        })
        return pd.DataFrame(), mock, {'erro': str(e)}


# 
# DESENHO DO CARD INDIVIDUAL
# 

def draw_card(
    img: Image.Image,
    x: int, y: int,
    w: int, h: int,
    data: pd.Series,
    rank: int
):
    """
    Desenha um card de reservatório em destaque com todos os campos visuais.
    """
    draw = ImageDraw.Draw(img)

    # ── Fundo do card (verde escuro semi-transparente) ──
    draw_rounded_rect_on_image(
        img, x, y, w, h,
        radius=14,
        fill_color=(0, 70, 35, 210),
        border_color=(0, 200, 83, 255),
        border_width=2
    )

    # ── Glow sutil na borda ─────────────────────────────
    draw_rounded_rect_on_image(
        img, x - 1, y - 1, w + 2, h + 2,
        radius=15,
        fill_color=(0, 0, 0, 0),
        border_color=(105, 240, 174, 60),
        border_width=4
    )

    # ── Badge de ranking ────────────────────────────────
    badge_cx = x + w - 32
    badge_cy = y + 28
    badge_r  = 22
    draw.ellipse(
        [badge_cx - badge_r, badge_cy - badge_r,
         badge_cx + badge_r, badge_cy + badge_r],
        fill=(0, 200, 83)
    )
    font_rank = get_font(18, bold=True)
    draw.text((badge_cx, badge_cy), str(rank), font=font_rank,
              fill=(255, 255, 255), anchor='mm')

    # ── Fontes 
    f_nome   = get_font(20, bold=True)
    f_label  = get_font(16)
    f_valor  = get_font(19, bold=True)
    f_var    = get_font(28, bold=True)
    f_pct    = get_font(21, bold=True)

    px = x + 16  # padding horizontal
    py = y + 14  # padding vertical inicial

    # ── Nome do reservatório ────────────────────────────
    nome = str(data.get('nome', 'N/A')).upper()
    nome = (nome[:22] + '…') if len(nome) > 22 else nome
    draw.text((px, py), nome, font=f_nome, fill=(255, 255, 255))

    # Linha divisória interna
    div_y = py + 28
    draw.line([(px, div_y), (x + w - 16, div_y)],
              fill=(0, 200, 83), width=1)

    # ── Variação do nível (destaque) ────────────────────
    var_m    = data.get('variacao_m', 0) or 0
    positivo = var_m >= 0
    cor_var  = (105, 240, 174) if positivo else (244, 100, 100)
    seta     = '▲' if positivo else '▼'
    sinal    = '+' if positivo else ''

    vy = div_y + 10
    draw.text((px, vy), seta, font=f_var, fill=cor_var)
    draw.text((px + 32, vy), f"{sinal}{format_number(var_m, 2)} m",
              font=f_var, fill=cor_var)
    draw.text((px, vy + 34), "Variação do nível",
              font=f_label, fill=(165, 214, 167))

    # ── Variação em m³ ──────────────────────────────────
    l2y = vy + 56
    draw.text((px, l2y), "Var. Volume:", font=f_label, fill=(144, 164, 174))
    var_m3 = data.get('variacao_m3', 0) or 0
    sinal3 = '+' if var_m3 >= 0 else ''
    draw.text((px, l2y + 18), f"{sinal3}{format_number(var_m3, 0)} m³",
              font=f_valor, fill=(224, 224, 224))

    # ── Volume atual ────────────────────────────────────
    l3y = l2y + 42
    draw.text((px, l3y), "Volume Atual:", font=f_label, fill=(144, 164, 174))
    draw.text((px, l3y + 18), f"{format_number(data.get('volume_atual_m3', 0), 0)} m³",
              font=f_valor, fill=(224, 224, 224))

    # ── Percentual + barra de progresso ─────────────────
    l4y = l3y + 48
    draw.text((px, l4y), "Capacidade:", font=f_label, fill=(144, 164, 174))

    pct = float(data.get('percentual', 0) or 0)
    pct = max(0, min(100, pct))

    # Percentual (direita)
    draw.text((x + w - 16, l4y), f"{format_number(pct, 1)}%",
              font=f_pct, fill=(105, 240, 174), anchor='ra')

    # Barra
    bar_y  = l4y + 22
    bar_x  = px
    bar_w  = w - 32
    bar_h  = 8
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                            radius=4, fill=(30, 50, 40))
    fill_w = max(0, int(bar_w * pct / 100))
    if fill_w > 0:
        draw.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h],
                                radius=4, fill=(0, 200, 83))


# 
# GERAÇÃO DA IMAGEM PRINCIPAL
# 

def generate_image(
    df_top: pd.DataFrame,
    date_anterior: str,
    date_atual: str,
    formato: str,
    titulo_custom: str
) -> Image.Image:
    """
    Gera a imagem 1080x1080 com todos os cards de reservatórios.
    """
    W, H = 1080, 1080

    # ── 1. Fundo com gradiente vertical ─────────────────
    img_arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        img_arr[y, :] = [
            int(13 + (20 - 13) * t),   # R: #0D → #14
            int(27 + (35 - 27) * t),   # G: #1B → #23
            int(42 + (55 - 42) * t),   # B: #2A → #37
        ]
    img = Image.fromarray(img_arr, 'RGB').convert('RGBA')

    # ── 2. Glow decorativo nos cantos ───────────────────
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for cx, cy, r, alpha in [
        (0,    0,    320, 18),
        (W,    0,    250, 12),
        (0,    H,    200, 10),
        (W//2, H//2, 500, 8),
    ]:
        od.ellipse([cx - r, cy - r, cx + r, cy + r],
                   fill=(79, 195, 247, alpha))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)

    # ── 3. Ícone de gota (círculo + triângulo) ──────────
    # Corpo da gota
    draw.ellipse([52, 48, 108, 104], fill=(79, 195, 247))
    # "ponta" da gota
    drop_points = [(80, 30), (52, 65), (108, 65)]
    draw.polygon(drop_points, fill=(79, 195, 247))

    # ── 4. Título ───────────────────────────────────────
    partes = titulo_custom.strip().split(' ', 3)
    linha1 = ' '.join(partes[:2]) if len(partes) >= 2 else titulo_custom
    linha2 = ' '.join(partes[2:]) if len(partes) > 2 else ''

    f_tit1 = get_font(44, bold=True)
    f_tit2 = get_font(52, bold=True)
    f_sub  = get_font(26)
    f_foot = get_font(21)

    draw.text((128, 40), linha1, font=f_tit1, fill=(255, 255, 255))
    if linha2:
        draw.text((128, 88), linha2, font=f_tit2, fill=(79, 195, 247))

    # ── 5. Subtítulo de período ──────────────────────────
    sub_y = 160 if linha2 else 100
    draw.text((55, sub_y),
              f"Período de referência: {date_anterior}  →  {date_atual}",
              font=f_sub, fill=(144, 164, 174))

    # ── 6. Linha divisória verde ─────────────────────────
    line_y = sub_y + 38
    draw.rectangle([55, line_y, W - 55, line_y + 4], fill=(0, 200, 83))
    # Pontos decorativos
    for i in range(55, W - 55, 18):
        draw.rectangle([i, line_y, i + 8, line_y + 4], fill=(0, 200, 83))

    # ── 7. Cards dinâmicos ───────────────────────────────
    start_y  = line_y + 16
    avail_h  = 980 - start_y      # altura disponível para cards
    margin   = 55
    gap      = 14
    num      = min(len(df_top), 6)

    cols = 1 if num &lt;= 1 else (2 if num &lt;= 4 else 3)
    rows = math.ceil(num / cols)

    card_w = (W - 2 * margin - (cols - 1) * gap) // cols
    card_h = (avail_h - (rows - 1) * gap) // rows

    for i, (_, row) in enumerate(df_top.head(num).iterrows()):
        col_i = i % cols
        row_i = i // cols
        cx = margin + col_i * (card_w + gap)
        cy = start_y + row_i * (card_h + gap)
        draw_card(img, cx, cy, card_w, card_h, row, i + 1)

    # ── 8. Rodapé 
    draw.line([(55, 1018), (W - 55, 1018)], fill=(50, 70, 80), width=1)
    draw.text((55, 1022), "📊 COGERH / SRH-CE",
              font=f_foot, fill=(84, 110, 122))
    ts = datetime.now().strftime('%d/%m/%Y às %H:%M')
    draw.text((W - 55, 1022), f"Gerado em {ts}",
              font=f_foot, fill=(84, 110, 122), anchor='ra')

    # ── 9. Converter formato de saída ────────────────────
    final = img.convert('RGB') if formato.upper() == 'JPG' else img
    return final


# 
# APLICATIVO STREAMLIT — FUNÇÃO PRINCIPAL
# 

def main():
    st.set_page_config(
        page_title="Reservatórios — Gerador de Cards",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS customizado
    st.markdown("""
    <style>
        .stApp            { background-color: #0a1628; color: #e0e0e0; }
        .stSidebar        { background-color: #0d1f35 !important; }
        h1, h2, h3        { color: #4FC3F7 !important; }
        .stMetric         { background: rgba(0,200,83,.08);
                            border: 1px solid #00C85344;
                            border-radius: 10px; padding: 8px; }
        .stButton > button { background: #00C853; color: #fff;
                             border-radius: 8px; font-weight: 700; }
        .stButton > button:hover { background: #00a844; }
        .stDownloadButton > button { background: #1565C0; color: #fff;
                                     border-radius: 8px; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    # URL de exportação CSV do Google Sheets
    SHEET_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1fbaYqjee8h4dAA8ew0RXbHOKdnSDoHIB2xPpdveYMDU"
        "/export?format=csv&gid=0"
    )

    # ── Carregamento de dados ────────────────────────────
    df_original, df_proc, info_cols = load_data(SHEET_URL)

    # ── SIDEBAR 
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        st.divider()

        titulo_custom = st.text_input(
            "📝 Título da imagem",
            value="Monitoramento dos Reservatórios"
        )
        min_variacao = st.slider(
            "📏 Variação mínima (m)",
            min_value=0.0, max_value=5.0,
            value=0.1, step=0.05,
            help="Filtra reservatórios com variação ≥ este valor"
        )
        max_reserv = st.slider(
            "🔢 Nº máximo de reservatórios",
            min_value=1, max_value=6, value=4
        )
        formato = st.selectbox("🖼️ Formato de saída", ["PNG", "JPG"])
        st.divider()
        mostrar_debug = st.toggle("🔍 Mostrar dados brutos", value=False)
        if st.button("🔄 Atualizar dados", use_container_width=True):
            load_data.clear()
            st.rerun()

        st.divider()
        st.caption("v1.0 · GF Informática")

    # ── CABEÇALHO 
    st.title("💧 Gerador de Cards — Monitoramento de Reservatórios")
    st.caption("Lê dados em tempo real do Google Sheets e gera imagens prontas para redes sociais.")
    st.divider()

    # ── FILTRO DOS DADOS ─────────────────────────────────
    df_filtrado = (
        df_proc[df_proc['variacao_m'] >= min_variacao]
        .sort_values('variacao_m', ascending=False)
        .head(max_reserv)
        .reset_index(drop=True)
    )

    # ── MÉTRICAS RÁPIDAS ─────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💾 Total carregado",    f"{len(df_proc)} reservatórios")
    c2.metric("📈 Variação positiva",  f"{len(df_proc[df_proc['variacao_m'] > 0])}")
    c3.metric("🏆 Maior variação",     f"{format_number(df_proc['variacao_m'].max(), 2)} m")
    c4.metric("🎯 Exibindo no card",   f"{len(df_filtrado)}")

    # ── DEBUG (expansível) ───────────────────────────────
    if mostrar_debug:
        with st.expander("📋 Colunas detectadas na planilha"):
            st.write("**Colunas:**", info_cols.get('colunas', []))
            if not df_original.empty:
                st.dataframe(df_original.head(10), use_container_width=True)

        with st.expander("📊 Dados processados e filtrados"):
            st.dataframe(df_filtrado, use_container_width=True)

    st.divider()

    # ── BOTÃO GERAR ──────────────────────────────────────
    if st.button("🎨 Gerar Imagem para Redes Sociais",
                 type="primary", use_container_width=True):

        if df_filtrado.empty:
            st.warning(
                "⚠️ Nenhum reservatório encontrado com os filtros atuais. "
                "Reduza a variação mínima na barra lateral."
            )
            # Gera imagem de aviso mesmo assim
            img_aviso = Image.new('RGBA', (1080, 1080), (13, 27, 42, 255))
            da = ImageDraw.Draw(img_aviso)
            fa = get_font(38, bold=True)
            da.text((540, 480), "Nenhum reservatório encontrado",
                    font=fa, fill=(200, 200, 200), anchor='mm')
            da.text((540, 540), "Reduza a variação mínima nos filtros",
                    font=get_font(26), fill=(144, 164, 174), anchor='mm')
            img_final = img_aviso.convert('RGB')
        else:
            with st.spinner("🎨 Gerando imagem premium..."):
                date_ant = df_filtrado['data_anterior'].iloc[0]
                date_atu = df_filtrado['data_atual'].iloc[0]
                img_final = generate_image(
                    df_top=df_filtrado,
                    date_anterior=date_ant,
                    date_atual=date_atu,
                    formato=formato,
                    titulo_custom=titulo_custom
                )

        # ── Preview ──────────────────────────────────────
        st.image(img_final, caption="Preview — 1080×1080 px", use_container_width=True)

        # ── Download ─────────────────────────────────────
        buf = BytesIO()
        mime_format = 'JPEG' if formato.upper() == 'JPG' else 'PNG'
        img_final.save(buf, format=mime_format, quality=95)
        buf.seek(0)

        nome_arquivo = (
            f"reservatorios_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f".{formato.lower()}"
        )
        st.download_button(
            label=f"📥 Baixar Imagem ({formato})",
            data=buf,
            file_name=nome_arquivo,
            mime=f"image/{'jpeg' if formato == 'JPG' else 'png'}",
            use_container_width=True,
            key=f"dl_{datetime.now().timestamp()}"
        )
        st.success(f"✅ Imagem gerada com sucesso! {len(df_filtrado)} reservatório(s) exibido(s).")


if __name__ == '__main__':
    main()
