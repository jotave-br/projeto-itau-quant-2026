"""Gera o relatório técnico anônimo em cinco páginas 16:9."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from pypdf import PdfReader, PdfWriter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import Paragraph  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/runs/2026-08-07_025233_oficial/tabelas"
V2_PANEL = ROOT / "outputs/runs/2026-08-12_211858_v2_painel/tabelas"
V2_TEST = ROOT / "outputs/runs/2026-08-13_223020_v2_backtest/tabelas"
V2_DOCS = ROOT / "outputs/runs/2026-08-11_204130_v2_documentos/tabelas"
MASCOT = ROOT / "assets/relatorio/mimir_mascote_v3.png"

PAGE_W = 960
PAGE_H = 540
MARGIN = 44

PAPER = "#F5F2EC"
WHITE = "#FCFBF8"
INK = "#1D2927"
INK_2 = "#4A5652"
MUTED = "#77817D"
RULE = "#C9CDC7"
STONE = "#E8E4DC"
INDIGO = "#40506F"
TEAL = "#3F7D6D"
TEAL_LIGHT = "#D8E7E1"
BRICK = "#A64B40"
BRICK_LIGHT = "#ECD8D3"
OCHRE = "#C18A42"
LILAC = "#82739B"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saida",
        type=Path,
        default=ROOT / "output/pdf/relatorio_final.pdf",
    )
    return parser.parse_args()


def _register_fonts() -> None:
    fonts = {
        "Segoe": "C:/Windows/Fonts/segoeui.ttf",
        "Segoe-Semibold": "C:/Windows/Fonts/seguisb.ttf",
        "Segoe-Bold": "C:/Windows/Fonts/segoeuib.ttf",
        "Georgia": "C:/Windows/Fonts/georgia.ttf",
        "Georgia-Bold": "C:/Windows/Fonts/georgiab.ttf",
        "Georgia-Italic": "C:/Windows/Fonts/georgiai.ttf",
        "Consolas": "C:/Windows/Fonts/consola.ttf",
        "Consolas-Bold": "C:/Windows/Fonts/consolab.ttf",
    }
    for name, raw_path in fonts.items():
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        pdfmetrics.registerFont(TTFont(name, str(path)))


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": RULE,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "mathtext.fontset": "stix",
        }
    )


def _read_data() -> dict[str, pd.DataFrame]:
    files = {
        "lags": V1 / "lags_alternativos.csv",
        "fdr": V1 / "fdr_resumo_por_faixa.csv",
        "placebo_v1": V1 / "placebo_resumo.csv",
        "portfolios": V1 / "resumo_carteiras.csv",
        "universe": V1 / "universo_por_janela.csv",
        "network": V1 / "rede_por_janela.csv",
        "stale": V1 / "reestimativa_sem_preco_velho.csv",
        "always_traded": V1 / "rede_sempre_negociado.csv",
        "matrix": V2_PANEL / "matriz_confusao.csv",
        "panel": V2_PANEL / "resumo_painel.csv",
        "v2_summary": V2_TEST / "resumo_bracos.csv",
        "v2_placebo": V2_TEST / "placebo_seguidora_distribuicao.csv",
        "v2_operations": V2_TEST / "operacoes_principal.csv",
        "funnel": V2_DOCS / "funil_documentos.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("arquivos ausentes: " + ", ".join(missing))
    out = {name: pd.read_csv(path, encoding="utf-8-sig") for name, path in files.items()}
    out["matrix"] = pd.read_csv(files["matrix"], index_col=0, encoding="utf-8-sig")
    classifications = ROOT / "data/processed/cvm_ipe/classificacoes_ia.parquet"
    if not classifications.exists():
        raise FileNotFoundError(classifications)
    out["classifications"] = pd.read_parquet(classifications)
    return out


def _clean_axes(ax, *, grid_x: bool = False, grid_y: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.tick_params(length=0)
    if grid_x:
        ax.xaxis.grid(True, color=RULE, linewidth=0.7, alpha=0.8)
    if grid_y:
        ax.yaxis.grid(True, color=RULE, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _save_fig(fig, path: Path, *, transparent: bool = False, dpi: int = 600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=transparent,
        metadata={"Author": "", "Creator": "", "Title": ""},
    )
    plt.close(fig)
    return path


def _fig_formula_v1(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 0.58))
    ax.axis("off")
    formula = r"$r^{F}_{j,t}=\alpha_{ij}+\beta^{(k)}_{ij}\,r^{L}_{i,t-k}+\varepsilon_{ij,t}$"
    ax.text(0.5, 0.5, formula, fontsize=29, ha="center", va="center", color=INK)
    return _save_fig(fig, path, transparent=True, dpi=900)


def _fig_formula_weights(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 0.60))
    ax.axis("off")
    formula = (
        r"$u_{j,c}=\sum_{s\in c}\frac{d_s}{|F_s|},\qquad "
        r"w_{j,c}=\frac{\operatorname{sgn}(u_{j,c})"
        r"\min\!\left(|u_{j,c}|/\sum_k|u_{k,c}|,\,10\%\right)}{H}$"
    )
    ax.text(0.5, 0.5, formula, fontsize=20, ha="center", va="center", color=INK)
    return _save_fig(fig, path, transparent=True, dpi=900)


def _fig_lags(lags: pd.DataFrame, path: Path) -> Path:
    series = {}
    for limit in (20, 40):
        series[limit] = (
            lags.loc[lags["faixa_minima"].le(limit)]
            .groupby("lag")["beta"]
            .median()
            .reindex([0, 1, 2, 3])
        )

    fig, ax = plt.subplots(figsize=(6.35, 2.55))
    specs = {20: (TEAL, "o", "Top 20"), 40: (INDIGO, "s", "Top 40")}
    offsets = {20: 11, 40: -15}
    for limit, values in series.items():
        color, marker, label = specs[limit]
        ax.plot(
            values.index,
            values.values,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            label=label,
        )
        for lag, value in values.items():
            ax.annotate(
                f"{value:+.3f}".replace(".", ","),
                xy=(lag, value),
                xytext=(0, offsets[limit]),
                textcoords="offset points",
                ha="center",
                va="bottom" if offsets[limit] > 0 else "top",
                color=color,
                fontsize=9.5,
                fontweight="semibold" if lag == 0 else "normal",
                bbox={"boxstyle": "round,pad=0.10", "facecolor": PAPER, "edgecolor": "none", "alpha": 0.92},
            )
    ax.axhline(0, color=INK_2, linewidth=0.9)
    ax.set_xticks(
        [0, 1, 2, 3],
        ["k=0\nmesmo pregão", "k=1\n1 pregão depois", "k=2\n2 pregões depois", "k=3\n3 pregões depois"],
    )
    ax.set_xlabel("Momento medido na seguidora")
    ax.set_ylabel("Coeficiente β mediano")
    ax.set_ylim(-0.095, 0.76)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _clean_axes(ax, grid_y=True)
    return _save_fig(fig, path)


def _fig_fdr(fdr: pd.DataFrame, path: Path) -> Path:
    base = fdr.loc[fdr["faixa"].eq("top40")].copy()
    positive = base["aprovados_fdr_beta_positivo"].astype(int)
    negative = base["aprovados_fdr"].astype(int) - positive
    x = np.arange(len(base))

    fig, ax = plt.subplots(figsize=(5.0, 2.35))
    ax.bar(x, positive, color=TEAL, width=0.72, label="β > 0")
    ax.bar(x, -negative, color=BRICK, width=0.72, hatch="////", label="β < 0")
    ax.axhline(0, color=INK_2, linewidth=0.9)
    tick_idx = np.arange(0, len(base), 4)
    tick_labels = [str(base.iloc[i]["janela"])[:4] for i in tick_idx]
    ax.set_xticks(tick_idx, tick_labels, fontsize=8.8)
    ax.set_ylabel("Aprovações par x janela")
    ax.set_xlabel("Trimestres de teste; ausência de barra = zero aprovação")
    ax.set_ylim(-6.8, 4.2)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.8)
    _clean_axes(ax, grid_y=True)
    return _save_fig(fig, path)


def _fig_ai_audit(matrix: pd.DataFrame, path: Path) -> Path:
    rows = ["especifico_positiva", "especifico_negativa", "especifico_neutra"]
    cols = ["especifico_positiva", "especifico_negativa", "especifico_neutra", "nao_especifico"]
    m = matrix.reindex(index=rows, columns=cols).fillna(0).astype(int)
    labels_y = ["Painel: positiva (n=16)", "Painel: negativa (n=18)", "Painel: neutra (n=56)"]
    labels_x = ["Qwen3:\npositiva", "Qwen3:\nnegativa", "Qwen3:\nneutra", "Qwen3:\nnão específica"]
    cmap = LinearSegmentedColormap.from_list("mimir_heat", [WHITE, TEAL_LIGHT, TEAL, INDIGO])

    fig, ax = plt.subplots(figsize=(5.7, 2.2))
    image = ax.imshow(m.to_numpy(), cmap=cmap, vmin=0, vmax=int(m.to_numpy().max()), aspect="auto")
    del image
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            value = int(m.iat[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold" if value else "normal",
                color=WHITE if value >= 14 else INK_2,
            )
    ax.set_xticks(np.arange(4), labels_x, fontsize=8.8)
    ax.set_yticks(np.arange(3), labels_y, fontsize=9.0)
    ax.tick_params(length=0)
    ax.set_xlabel("Rótulo produzido pelo classificador local")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save_fig(fig, path)


def _fig_placebo(distribution: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    values = distribution["retorno_total"] * 100
    row = summary.loc[summary["braco"].eq("ia_mais_rede") & summary["h"].eq(3)].iloc[0]
    actual = float(row["retorno_total"] * 100)
    median = float(values.median())
    fig, ax = plt.subplots(figsize=(6.0, 2.55))
    ax.hist(values, bins=22, color=TEAL_LIGHT, edgecolor=PAPER, linewidth=0.9)
    ax.axvline(median, color=INDIGO, linestyle="--", linewidth=1.7,
               label=f"Mediana placebo {median:+.2f}%".replace(".", ","))
    ax.axvline(actual, color=BRICK, linewidth=2.2,
               label=f"Estratégia {actual:+.2f}%".replace(".", ","))
    ax.set_xlabel("Retorno líquido acumulado em H=3 (%)")
    ax.set_ylabel("Redes aleatórias")
    ax.legend(frameon=False, fontsize=9.2, loc="upper left")
    _clean_axes(ax, grid_y=True)
    return _save_fig(fig, path)


def _make_figures(data: dict[str, pd.DataFrame], dest: Path) -> dict[str, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    return {
        "formula_v1": _fig_formula_v1(dest / "01_formula_v1.png"),
        "formula_weights": _fig_formula_weights(dest / "02_formula_pesos.png"),
        "lags": _fig_lags(data["lags"], dest / "03_lags.png"),
        "fdr": _fig_fdr(data["fdr"], dest / "04_fdr.png"),
        "ai": _fig_ai_audit(data["matrix"], dest / "05_auditoria_ia.png"),
        "placebo": _fig_placebo(data["v2_placebo"], data["v2_summary"], dest / "06_placebo.png"),
    }


def _color(hex_value: str):
    return colors.HexColor(hex_value)


def _paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    top: float,
    width: float,
    height: float,
    *,
    size: float = 9.4,
    leading: float | None = None,
    font: str = "Georgia",
    color: str = INK_2,
    align: int = TA_LEFT,
) -> float:
    style = ParagraphStyle(
        "body",
        fontName=font,
        fontSize=size,
        leading=leading or size * 1.33,
        textColor=_color(color),
        alignment=align,
        allowWidows=0,
        allowOrphans=0,
        spaceAfter=0,
    )
    p = Paragraph(text, style)
    _, used_h = p.wrap(width, height)
    if used_h > height:
        raise ValueError(f"texto excede a área: {text[:90]}")
    p.drawOn(c, x, top - used_h)
    return used_h


def _rule(c: canvas.Canvas, x1: float, y: float, x2: float, *, color: str = RULE, width: float = 0.7) -> None:
    c.setStrokeColor(_color(color))
    c.setLineWidth(width)
    c.line(x1, y, x2, y)


def _header(c: canvas.Canvas, page: int, section: str) -> None:
    c.setFillColor(_color(PAPER))
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    if not MASCOT.exists():
        raise FileNotFoundError(MASCOT)
    _image(c, MASCOT, MARGIN, PAGE_H - 34, 24, 27)
    c.setFillColor(_color(INK))
    c.setFont("Segoe-Bold", 11)
    c.drawString(MARGIN + 29, PAGE_H - 28, "MÍMIR")
    c.setFillColor(_color(MUTED))
    c.setFont("Segoe", 7.5)
    c.drawString(MARGIN + 74, PAGE_H - 27, "LEAD-LAG SETORIAL NA B3")
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 27, f"{section.upper()}  |  {page}/5")
    _rule(c, MARGIN, PAGE_H - 38, PAGE_W - MARGIN)


def _page_title(c: canvas.Canvas, title: str, subtitle: str | None = None) -> None:
    _paragraph(c, title, MARGIN, PAGE_H - 57, PAGE_W - 2 * MARGIN, 48,
               size=23.0, leading=25.0, font="Segoe-Semibold", color=INK)
    if subtitle:
        _paragraph(c, subtitle, MARGIN, PAGE_H - 88, PAGE_W - 2 * MARGIN, 28,
                   size=10.0, leading=12.5, font="Georgia-Italic", color=MUTED)


def _section(c: canvas.Canvas, title: str, x: float, top: float, width: float, *, color: str = INDIGO) -> None:
    c.setFillColor(_color(color))
    c.rect(x, top - 3, 18, 2, fill=1, stroke=0)
    c.setFillColor(_color(INK))
    c.setFont("Segoe-Semibold", 9.7)
    c.drawString(x + 24, top - 6, title.upper())
    _rule(c, x, top - 13, x + width, color=RULE, width=0.6)


def _caption(c: canvas.Canvas, label: str, text: str, x: float, top: float, width: float) -> None:
    _paragraph(c, f"<b>{label}</b> {text}", x, top, width, 35,
               size=8.7, leading=10.7, font="Georgia", color=MUTED)


def _image(c: canvas.Canvas, path: Path, x: float, y: float, width: float, height: float) -> None:
    c.drawImage(ImageReader(str(path)), x, y, width=width, height=height,
                preserveAspectRatio=True, anchor="c", mask="auto")


def _table(
    c: canvas.Canvas,
    x: float,
    top: float,
    widths: list[float],
    headers: list[str],
    rows: list[list[str]],
    *,
    row_h: float = 25,
    header_h: float = 23,
    size: float = 8.1,
    header_size: float = 8.3,
    fonts: list[str] | None = None,
) -> float:
    total_w = sum(widths)
    c.setFillColor(_color(STONE))
    c.rect(x, top - header_h, total_w, header_h, fill=1, stroke=0)
    cursor = x
    for i, (header, width) in enumerate(zip(headers, widths, strict=True)):
        _paragraph(c, header, cursor + 6, top - 5, width - 12, header_h - 5,
                   size=header_size, leading=header_size * 1.16,
                   font="Segoe-Semibold", color=INK)
        cursor += width
    y = top - header_h
    for row_idx, row in enumerate(rows):
        if row_idx % 2:
            c.setFillColor(_color(WHITE))
            c.rect(x, y - row_h, total_w, row_h, fill=1, stroke=0)
        _rule(c, x, y - row_h, x + total_w, color=RULE, width=0.35)
        cursor = x
        for col_idx, (value, width) in enumerate(zip(row, widths, strict=True)):
            font = fonts[col_idx] if fonts else "Georgia"
            _paragraph(c, value, cursor + 6, y - 6, width - 12, row_h - 7,
                       size=size, leading=size * 1.15, font=font, color=INK_2)
            cursor += width
        y -= row_h
    return header_h + len(rows) * row_h


def _metric_line(c: canvas.Canvas, x: float, y: float, width: float,
                 label: str, value: str, *, color: str = INK) -> None:
    c.setFont("Georgia", 8.0)
    c.setFillColor(_color(MUTED))
    c.drawString(x, y, label)
    c.setFont("Segoe-Semibold", 10.5)
    c.setFillColor(_color(color))
    c.drawRightString(x + width, y - 1, value)
    _rule(c, x, y - 6, x + width, color=RULE, width=0.35)


def _card(
    c: canvas.Canvas,
    x: float,
    top: float,
    width: float,
    height: float,
    *,
    label: str | None = None,
    fill: str = WHITE,
    accent: str = INDIGO,
) -> None:
    c.setFillColor(_color(fill))
    c.setStrokeColor(_color(RULE))
    c.setLineWidth(0.55)
    c.rect(x, top - height, width, height, fill=1, stroke=1)
    c.setFillColor(_color(accent))
    c.rect(x, top - 3, width, 3, fill=1, stroke=0)
    if label:
        c.setFont("Segoe-Semibold", 8.3)
        c.setFillColor(_color(accent))
        c.drawString(x + 10, top - 17, label.upper())


def _flow_strip(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    labels: list[str],
    *,
    colors_flow: list[str] | None = None,
    size: float = 8.6,
) -> None:
    colors_used = colors_flow or [INDIGO] * len(labels)
    if len(colors_used) != len(labels):
        raise ValueError("uma cor por etapa")
    centers = np.linspace(x + 44, x + width - 44, len(labels))
    for start, end in zip(centers[:-1], centers[1:], strict=True):
        c.setStrokeColor(_color(RULE))
        c.setLineWidth(1.2)
        c.line(start + 7, y, end - 10, y)
        c.setFillColor(_color(RULE))
        arrow = c.beginPath()
        arrow.moveTo(end - 5, y)
        arrow.lineTo(end - 12, y + 4)
        arrow.lineTo(end - 12, y - 4)
        arrow.close()
        c.drawPath(arrow, stroke=0, fill=1)
    cell_w = width / len(labels)
    for center, label, color in zip(centers, labels, colors_used, strict=True):
        c.setFillColor(_color(color))
        c.circle(center, y, 5.3, stroke=0, fill=1)
        _paragraph(
            c,
            label,
            center - cell_w * 0.46,
            y - 10,
            cell_w * 0.92,
            34,
            size=size,
            leading=size * 1.18,
            font="Segoe-Semibold",
            color=INK,
            align=TA_CENTER,
        )


def _validate_data(data: dict[str, pd.DataFrame]) -> None:
    network = data["network"]
    top20 = network.loc[network["faixa_minima"].le(20)]
    unique_cols = ["lider", "seguidora"]
    expected = {
        "top40_edge_windows": (len(network), 1247),
        "top20_edge_windows": (len(top20), 452),
        "top40_unique_pairs": (len(network[unique_cols].drop_duplicates()), 131),
        "top20_unique_pairs": (len(top20[unique_cols].drop_duplicates()), 42),
        "windows": (network["janela"].nunique(), 38),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"{label}: esperado {wanted}, encontrado {actual}")

    matrix = data["matrix"]
    checks = {
        ("especifico_positiva", "especifico_positiva"): 16,
        ("especifico_negativa", "especifico_negativa"): 18,
        ("especifico_neutra", "especifico_positiva"): 14,
        ("especifico_neutra", "especifico_negativa"): 12,
        ("especifico_neutra", "especifico_neutra"): 25,
        ("especifico_neutra", "nao_especifico"): 5,
    }
    for key, wanted in checks.items():
        actual = int(matrix.at[key[0], key[1]])
        if actual != wanted:
            raise ValueError(f"matriz {key}: esperado {wanted}, encontrado {actual}")

    classifications = data["classifications"]
    truncated = classifications["texto_truncado_llm"].astype(bool)
    errors = classifications["status_ia"].eq("erro")
    if (int(truncated.sum()), int((truncated & errors).sum()), int(errors.sum())) != (23, 16, 87):
        raise ValueError("contagens de falha da IA mudaram")

    summary = data["v2_summary"]
    h3 = summary.loc[summary["braco"].eq("ia_mais_rede") & summary["h"].eq(3)].iloc[0]
    if int(h3["operacoes"]) != 99 or not np.isclose(h3["retorno_total"], -0.0008049030274807165):
        raise ValueError("resultado principal V2 mudou")


def _research_flow(c: canvas.Canvas, x: float, top: float, width: float) -> None:
    labels = [
        "Intuição\nplausível",
        "Teste agregado\n— V1",
        "Ausência de lead-lag\nrobusto",
        "Último teste\ncondicional — V2",
        "IA não validada +\nestratégia não negociável",
        "Decisão de\nabandono",
    ]
    colors_flow = [INDIGO, INDIGO, BRICK, LILAC, BRICK, TEAL]
    node_w = 122
    gap = (width - node_w * len(labels)) / (len(labels) - 1)
    y = top - 10
    centers = [x + node_w / 2 + idx * (node_w + gap) for idx in range(len(labels))]
    for idx in range(len(centers) - 1):
        start = centers[idx] + 7
        end = centers[idx + 1] - 9
        c.setStrokeColor(_color(RULE))
        c.setLineWidth(1.15)
        c.line(start, y, end, y)
        c.setFillColor(_color(RULE))
        arrow = c.beginPath()
        arrow.moveTo(end + 3, y)
        arrow.lineTo(end - 4, y + 4)
        arrow.lineTo(end - 4, y - 4)
        arrow.close()
        c.drawPath(arrow, stroke=0, fill=1)
    for idx, (center, label, node_color) in enumerate(zip(centers, labels, colors_flow, strict=True), start=1):
        c.setFillColor(_color(node_color))
        c.circle(center, y, 5.2, stroke=0, fill=1)
        c.setFillColor(_color(WHITE))
        c.setFont("Segoe-Bold", 5.7)
        c.drawCentredString(center, y - 2.0, str(idx))
        text = label.replace("\n", "<br/>")
        _paragraph(c, text, center - node_w / 2, top - 23, node_w, 34,
                   size=8.3, leading=9.7, font="Segoe-Semibold", color=INK,
                   align=TA_CENTER)


def _page_1(c: canvas.Canvas, data: dict[str, pd.DataFrame]) -> None:
    _header(c, 1, "Pergunta, identidade e amostra")
    _page_title(
        c,
        "Lead-lag setorial na B3: existe atraso negociável?",
        "Mímir testa se a ação mais líquida incorpora informação antes de empresas relacionadas menos líquidas.",
    )

    left_x, left_w = MARGIN, 548
    right_x, right_w = 620, 296

    _section(c, "Pergunta econômica", left_x, 410, left_w)
    question = (
        "A pergunta nasce de uma intuição sobre imitação e aprendizado social. Em sistemas humanos, informações de "
        "agentes relevantes podem influenciar as decisões dos demais. No mercado financeiro, surge uma pergunta "
        "testável: informações sobre uma empresa líder são incorporadas primeiro ao preço de sua ação e apenas depois "
        "aos preços de empresas economicamente relacionadas?<br/><br/>"
        "<b>Condição negociável.</b> Para sustentar uma estratégia, esse atraso precisa durar ao menos um pregão, "
        "repetir-se fora da amostra e permanecer rentável após custos de negociação."
    )
    _paragraph(c, question, left_x, 381, left_w, 103, size=10.1, leading=13.0)

    _section(c, "Hipótese operacional", left_x, 280, left_w, color=OCHRE)
    _flow_strip(
        c,
        left_x,
        248,
        left_w,
        [
            "Universo PIT<br/>por liquidez",
            "Líder move no<br/>fechamento t",
            "Seguidora entra<br/>em t+1",
            "H=3, giro<br/>e custos",
        ],
        colors_flow=[INDIGO, TEAL, OCHRE, BRICK],
        size=8.4,
    )

    _section(c, "Exemplos recorrentes na rede V1 top 40", left_x, 193, left_w)
    pairs_rows = [
        ["Financeiro", "ITUB4 → BBDC4 / BBAS3 / ITSA4", "38 / 38 / 38"],
        ["Petróleo", "PETR4 → PRIO3 / UGPA3", "20 / 26"],
        ["Siderurgia", "GGBR4 → CSNA3 / USIM5", "35 / 30"],
        ["Varejo", "MGLU3 → LREN3", "27"],
    ]
    _table(
        c,
        left_x,
        169,
        [104, 322, 122],
        ["GRUPO", "PAR DIRECIONADO", "Nº DE JANELAS"],
        pairs_rows,
        row_h=22,
        header_h=20,
        size=9.0,
        header_size=8.7,
        fonts=["Georgia", "Consolas", "Consolas"],
    )

    _section(c, "Por que Mímir?", right_x, 410, right_w, color=BRICK)
    _image(c, MASCOT, right_x + 4, 290, 88, 96)
    mimir = (
        "Na mitologia nórdica, Mímir guarda o poço da sabedoria. O nome representa a estratégia, que utiliza os "
        "movimentos das ações mais líquidas como sinais antecipados das ações seguidoras, considerando que esse "
        "conhecimento só tem valor se superar os custos de negociação."
    )
    _paragraph(c, mimir, right_x + 100, 381, right_w - 100, 100, size=9.25, leading=11.8)

    _section(c, "Amostra e unidade de contagem", right_x, 263, right_w, color=TEAL)
    _paragraph(
        c,
        "<b>Observação par x janela</b> é um par direcionado líder para seguidora, elegível e congelado para um trimestre de teste. A rede é refeita antes de cada janela.",
        right_x,
        235,
        right_w,
        43,
        size=9.2,
        leading=11.5,
    )
    sample_rows = [
        ["Treino inicial", "jan/2015 a dez/2016"],
        ["Testes OOS", "2017-Q1 a 2026-Q2"],
        ["Walk-forward", "38 redes; 24m / 3m"],
        ["V1 top 40", "1.247 observações; 131 pares únicos"],
        ["V2 top 20", "452 observações; 42 pares únicos"],
    ]
    _table(
        c,
        right_x,
        184,
        [105, 191],
        ["CAMPO", "DEFINIÇÃO"],
        sample_rows,
        row_h=25,
        header_h=20,
        size=8.8,
        header_size=8.7,
        fonts=["Segoe-Semibold", "Georgia"],
    )
    c.showPage()


def _page_2(c: canvas.Canvas, figs: dict[str, Path]) -> None:
    _header(c, 2, "Dados, relógio e método")
    _page_title(
        c,
        "Dados e método: um teste sem relógio inventado",
        "O desenho reduz look-ahead e não sincronia; fator comum e falta de timestamp ainda limitam a identificação.",
    )

    left_x, left_w = MARGIN, 420
    right_x, right_w = 490, 426
    _section(c, "Fontes: função e limite", left_x, 410, left_w)
    source_rows = [
        ["B3 COTAHIST", "Identidade, calendário, volume e PREULT.", "Preço bruto; mascarar eventos corporativos."],
        ["Yahoo", "Robustez com preço ajustado.", "Nunca define universo ou identidade."],
        ["B3 + CVM/FCA", "Setor histórico, ISIN e deslistados.", "Validade histórica exige curadoria."],
        ["CVM IPE", "Fatos Relevantes originais da V2.", "Data_Entrega não informa hora negociável."],
    ]
    _table(
        c,
        left_x,
        386,
        [92, 164, 164],
        ["FONTE", "PAPEL", "LIMITE"],
        source_rows,
        row_h=41,
        header_h=20,
        size=8.8,
        header_size=8.7,
        fonts=["Segoe-Semibold", "Georgia", "Georgia"],
    )

    _section(c, "Regra temporal", right_x, 410, right_w, color=OCHRE)
    timing = (
        "O desafio foi demonstrar <b>quando</b> a informação se tornou pública: fontes gratuitas raramente preservam "
        "horário e revisões. Como a CVM fornece Data_Entrega sem hora, um documento em D só foi considerado conhecido "
        "na primeira abertura B3 posterior. Isso pode perder o movimento do evento, mas evita inventar um timestamp."
    )
    _paragraph(c, timing, right_x, 381, right_w, 92, size=9.3, leading=11.8)
    _flow_strip(
        c,
        right_x,
        263,
        right_w,
        ["Documento<br/>em D", "Hora não<br/>auditável", "1ª abertura<br/>depois de D"],
        colors_flow=[INDIGO, BRICK, TEAL],
        size=8.5,
    )

    _section(c, "Modelo por par e por lag", left_x, 196, left_w, color=TEAL)
    _image(c, figs["formula_v1"], left_x + 8, 120, left_w - 16, 58)
    _paragraph(
        c,
        "i = líder | j = seguidora | k estimado separadamente em 0, 1, 2 e 3 pregões",
        left_x,
        113,
        left_w,
        18,
        size=8.7,
        leading=10.2,
        font="Consolas",
        color=MUTED,
        align=TA_CENTER,
    )
    _paragraph(
        c,
        "Retorno simples: PREULT<sub>t</sub>/PREULT<sub>t-1</sub>-1, somente com negociação em pregões consecutivos, sem interpolação e com fronteiras de eventos corporativos removidas.",
        left_x,
        88,
        left_w,
        48,
        size=8.9,
        leading=11.1,
    )

    _section(c, "Seleção antes do teste e controles", right_x, 196, right_w, color=BRICK)
    method_rows = [
        ["Universo PIT", "Cobertura e negociação ≥95%; ≥400 pregões; ISIN; setor vigente; liquidez mediana só do treino."],
        ["Regressão", "OLS: intercepto + retorno defasado da líder; sem mercado, setor ou lags próprios; HAC(5); ≥100 dias; k=1 principal."],
        ["FDR", "BH 10% em p HAC bicaudais por treino top 40: 38 famílias de 24-42 testes, nunca 1.247 juntos; só beta>0 negocia; rede congelada."],
        ["Não sincronia", "Stale = volume 0, &lt;10 negócios, volume &lt;P10 ou 2º fechamento igual; remove o dia e o seguinte. COTAHIST não traz hora do último negócio."],
    ]
    _table(
        c,
        right_x,
        172,
        [102, 324],
        ["ETAPA", "REGRA"],
        method_rows,
        row_h=29,
        header_h=18,
        size=8.6,
        header_size=8.5,
        fonts=["Segoe-Semibold", "Georgia"],
    )
    _paragraph(
        c,
        "Referências: Hou (2007, RFS); Chordia e Swaminathan (2000, JF); Scholes e Williams (1977, JFE); Benjamini e Hochberg (1995, JRSS-B).",
        MARGIN,
        24,
        PAGE_W - 2 * MARGIN,
        16,
        size=8.7,
        leading=9.8,
        color=MUTED,
        align=TA_CENTER,
    )
    c.showPage()


def _page_3(c: canvas.Canvas, data: dict[str, pd.DataFrame], figs: dict[str, Path]) -> None:
    _header(c, 3, "V1: resultados")
    _page_title(
        c,
        "V1: associação contemporânea, sem atraso persistente",
        "O teste agregado combina 38 redes trimestrais, seleção somente no treino, placebo setorial e execução fora da amostra.",
    )
    _section(c, "Coeficiente por defasagem", MARGIN, 405, 540)
    _image(c, figs["lags"], MARGIN, 190, 540, 198)
    _caption(
        c,
        "FIGURA 1.",
        "Observações par x janela. Em k=1: base -0,0152; sem stale -0,0170; sempre negociados -0,0157; direção inversa -0,0213. São proxies, não prova de fechamento síncrono.",
        MARGIN,
        183,
        540,
    )

    right_x, right_w = 610, 306
    _section(c, "FDR ao longo das 38 janelas", right_x, 405, right_w, color=BRICK)
    _image(c, figs["fdr"], right_x, 228, right_w, 160)
    _caption(
        c,
        "FIGURA 2.",
        "26 aprovações em 9 janelas, todas a partir de 2023. Placebo: 500 sorteios por janela (19.000); p mediano 0,632, faixa 0,320-0,910; 0/38 abaixo de 5%.",
        right_x,
        219,
        right_w,
    )
    _paragraph(
        c,
        "<b>Leitura.</b> São 26 aprovações par x janela, mas só 12 pares únicos: 11 betas positivos negociáveis e 15 negativos diagnósticos. As primeiras 24 janelas zeraram; não há persistência.",
        right_x,
        177,
        right_w,
        54,
        size=8.9,
        leading=11.1,
    )

    _section(c, "Backtest V1: soma do P&L diário OOS, 2017-Q1 a 2026-Q2", MARGIN, 139, 552, color=OCHRE)
    portfolios = data["portfolios"].set_index("carteira")
    order = ["fdr_long_only", "fdr_long_short", "top_k_long_only", "top_k_long_short"]
    names = {
        "fdr_long_only": "FDR | long-only",
        "fdr_long_short": "FDR | long-short",
        "top_k_long_only": "Top-k | long-only",
        "top_k_long_short": "Top-k | long-short",
    }
    rows = []
    for key in order:
        row = portfolios.loc[key]
        rows.append([
            names[key],
            f"{row.pnl_bruto_total * 100:+.2f}%".replace(".", ","),
            f"{row.pnl_liquido_total * 100:+.2f}%".replace(".", ","),
            f"{row.turnover_medio_diario * 100:.2f}%".replace(".", ","),
        ])
    _table(
        c,
        MARGIN,
        112,
        [224, 105, 105, 118],
        ["CARTEIRA", "BRUTO", "LÍQUIDO", "GIRO/PREGÃO"],
        rows,
        row_h=20,
        header_h=19,
        size=8.7,
        header_size=8.5,
        fonts=["Segoe", "Consolas", "Consolas", "Consolas"],
    )
    _paragraph(
        c,
        "<b>Seleção.</b> FDR usa só beta&gt;0. Top-k: até 20 maiores t positivos no treino, não o universo top 20.<br/><b>Execução.</b> Sinal no fechamento t; entrada da seguidora no fechamento t+1; P&amp;L em t+2...t+4. Pesos por volatilidade de 60 pregões, normalização bruta, teto de 10% sem reelevar e safras 1/H. Long-only renormaliza compras.<br/><b>Custos.</b> 13,25 bps/ponta (5 spread + 5 slippage + 3,25 emolumentos), duas pontas; aluguel 5%/252 no short. PREULT sem dividendos/JCP; total não anualizado nem composto.",
        right_x,
        112,
        right_w,
        100,
        size=8.9,
        leading=10.4,
        color=INK_2,
    )
    c.showPage()


def _page_4(c: canvas.Canvas, data: dict[str, pd.DataFrame], figs: dict[str, Path]) -> None:
    _header(c, 4, "V2: eventos e IA")
    _page_title(
        c,
        "V2: o classificador reprovou; o backtest é exploratório",
        "O teste condicional tinha propósito e protocolo próprios; o gate impediu que o resultado fosse tratado como evidência confirmatória.",
    )
    _card(c, MARGIN, 410, PAGE_W - 2 * MARGIN, 68, label="STATUS INFERENCIAL", accent=BRICK)
    _paragraph(
        c,
        "H=3 principal e H=1/H=5 como robustez foram pré-especificados em protocolo versionado antes do P&amp;L. O gate reprovou. Uma emenda registrada antes do primeiro backtest permitiu somente análise <b>exploratória</b>, sem mudar modelo, prompt, top 20, holding ou limiares.",
        MARGIN + 10,
        384,
        PAGE_W - 2 * MARGIN - 20,
        44,
        size=9.5,
        leading=11.8,
        color=INK,
    )

    left_x, left_w = MARGIN, 300
    right_x, right_w = 370, 546
    _section(c, "Funil de Fatos Relevantes", left_x, 320, left_w)
    funnel_rows = [
        ["Metadados das líderes", "3.280"],
        ["Apresentações originais", "3.085"],
        ["Mapeadas / em janelas", "3.071 / 2.668"],
        ["Top 20 / com texto", "884 / 882"],
        ["Classificações válidas", "795"],
        ["Eventos direcionais", "76"],
        ["Sinais líder-dia", "73"],
    ]
    _table(
        c,
        left_x,
        296,
        [210, 90],
        ["ETAPA", "N"],
        funnel_rows,
        row_h=19,
        header_h=18,
        size=9.0,
        header_size=8.7,
        fonts=["Georgia", "Consolas-Bold"],
    )

    _section(c, "Classificador e usos de IA", left_x, 135, left_w, color=LILAC)
    _paragraph(
        c,
        "<b>Qwen3:14B local</b>: temperatura 0, seed 20260811, prompt ia-eventos-1.2.5. Sem preços ou P&amp;L; JSON: não específico ou específico positivo, negativo ou neutro.<br/><br/>"
        "<b>Exemplo real.</b> PETR4 informou impacto de cerca de R$ 350 milhões na receita. Saída: <font name='Consolas'>{especifico_empresa:true, direcao:negativa}</font>.<br/><br/>"
        "Outras LLMs apoiaram parsing, painel cego, mascote e conferência de cálculos; as etapas quantitativas posteriores foram determinísticas.",
        left_x,
        108,
        left_w,
        104,
        size=8.8,
        leading=10.5,
    )

    _section(c, "Amostra de estresse por previsão (n=90)", right_x, 320, right_w, color=TEAL)
    _image(c, figs["ai"], right_x, 169, right_w, 133)
    _caption(
        c,
        "FIGURA 3.",
        "Contagens, não percentuais. Entre 56 neutros do painel, 14 viraram positivos, 12 negativos e 5 não específicos.",
        right_x,
        161,
        right_w,
    )

    _section(c, "Painel, gate e falhas técnicas", right_x, 142, right_w, color=BRICK)
    gate_rows = [
        ["Gold por maioria", "4 LLMs + 1 humano", "humano só desempata; 62/28/0"],
        ["Fleiss κ", "0,703", "convergência, não verdade"],
        ["Macro-F1 com suporte", "0,688", "reprova <0,70"],
        ["Classes com suporte", "3 de 4", "reprova"],
        ["Recusas técnicas", "87 = 78 + 9", "evidência / retic.; trunc. 16/23 vs 71/859"],
    ]
    _table(
        c,
        right_x,
        118,
        [190, 145, 211],
        ["MÉTRICA", "VALOR", "LEITURA"],
        gate_rows,
        row_h=18,
        header_h=18,
        size=8.0,
        header_size=8.2,
        fonts=["Georgia", "Consolas", "Georgia"],
    )
    c.showPage()


def _page_5(c: canvas.Canvas, data: dict[str, pd.DataFrame], figs: dict[str, Path]) -> None:
    _header(c, 5, "V2: backtest e decisão")
    _page_title(
        c,
        "V2 exploratória: 99 operações não superam redes aleatórias",
        "H=3 é principal; H=1 e H=5 são testes de robustez pré-declarados. Promover H=1 depois do resultado seria data snooping.",
    )
    left_x, left_w = MARGIN, 510
    right_x, right_w = 582, 334
    _section(c, "Randomização de seguidoras", left_x, 410, left_w, color=TEAL)
    _image(c, figs["placebo"], left_x, 219, left_w, 172)
    _caption(
        c,
        "FIGURA 4.",
        "500 redes preservam janela, líder, setor, top 20 e nº de arestas; sem self-pairs. 159/500 ≥ estratégia; p=(159+1)/(500+1)=0,319.",
        left_x,
        212,
        left_w,
    )

    _section(c, "Peso, execução e custos", right_x, 410, right_w, color=OCHRE)
    _image(c, figs["formula_weights"], right_x + 3, 341, right_w - 6, 48)
    _paragraph(
        c,
        "d = direção textual | F = seguidoras<br/>abre na 1ª abertura após D | fecha no H-ésimo pregão desde a entrada",
        right_x,
        334,
        right_w,
        21,
        size=8.3,
        leading=9.5,
        font="Consolas",
        color=MUTED,
        align=TA_CENTER,
    )
    _paragraph(
        c,
        "No H=3, cada operação pesa ±3,33%; exposição bruta máxima 13,33%. P&amp;L bruto +0,895%; entrada e saída -0,875%; aluguel -0,101%; líquido -0,080%. Custos: 13,25 bps por lado e aluguel-base de 5% a.a. no short.",
        right_x,
        307,
        right_w,
        46,
        size=8.6,
        leading=10.7,
    )

    _section(c, "Horizontes e inferência", right_x, 259, right_w, color=INDIGO)
    summary = data["v2_summary"]
    rows = []
    for h in (1, 3, 5):
        row = summary.loc[summary["braco"].eq("ia_mais_rede") & summary["h"].eq(h)].iloc[0]
        ci = f"[{row.ic_inferior * 10000:+.3f}; {row.ic_superior * 10000:+.3f}]".replace(".", ",")
        rows.append([
            f"H={h} | n={int(row.operacoes)}" + (" principal" if h == 3 else ""),
            f"{row.retorno_total * 100:+.2f}%".replace(".", ","),
            f"{row.sharpe:+.3f}".replace(".", ","),
            ci,
        ])
    _table(
        c,
        right_x,
        235,
        [92, 58, 58, 126],
        ["H (OPERAÇÕES)", "RET.", "SHARPE", "IC 95% BP/PREGÃO"],
        rows,
        row_h=25,
        header_h=20,
        size=8.2,
        header_size=8.0,
        fonts=["Segoe", "Consolas", "Consolas", "Consolas"],
    )
    _paragraph(
        c,
        "H=3: 206 dias ativos; IC por pregão em 2.870 sessões, zeros incluídos. Bootstrap móvel: 10.000 amostras, bloco 10; Newey-West. Todos cruzam zero.",
        right_x,
        132,
        right_w,
        28,
        size=8.5,
        leading=10.4,
    )

    _section(c, "O que o nulo permite concluir", MARGIN, 94, 422, color=BRICK)
    limitations = (
        "As 99 posições partem de 73 choques em 72 datas e não são independentes; faltaram potência/MDE e saída por reversão. O P&amp;L usa retorno de preço, não anormal, e omite a resposta contemporânea da líder por falta de hora auditável. Com o gate reprovado, não identifica difusão."
    )
    _paragraph(c, limitations, MARGIN, 66, 422, 35, size=8.7, leading=10.3)

    _section(c, "Decisão e aprendizado", 494, 94, 422, color=TEAL)
    decision = (
        "O fracasso é da hipótese operacional, não da pesquisa. Lead-lag pode existir, mas não foi verificável nem negociável neste desenho. A decisão é não implementar: abandonar um sinal frágil antes de alocar capital também é controle de risco."
    )
    _paragraph(c, decision, 494, 66, 422, 35, size=8.7, leading=10.3)

    _flow_strip(
        c,
        MARGIN,
        24,
        PAGE_W - 2 * MARGIN,
        ["Intuição plausível", "Teste agregado - V1", "Sem atraso robusto", "Teste condicional - V2", "IA reprovada + não negociável", "Decisão de abandono"],
        colors_flow=[INDIGO, INDIGO, BRICK, LILAC, BRICK, TEAL],
        size=7.3,
    )
    c.showPage()


def _scrub_metadata(source: Path, target: Path) -> None:
    reader = PdfReader(str(source))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": "Mímir - lead-lag setorial na B3",
            "/Author": "",
            "/Subject": "Relatório técnico anônimo",
            "/Creator": "",
            "/Producer": "",
        }
    )
    with target.open("wb") as stream:
        writer.write(stream)


def _word_count(path: Path) -> int:
    text = " ".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    return len(re.findall(r"\b[\wÀ-ÿ]+(?:-[\wÀ-ÿ]+)?\b", text))


def _build_pdf(path: Path, data: dict[str, pd.DataFrame], figs: dict[str, Path]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + "_temp.pdf")
    c = canvas.Canvas(str(temp), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setAuthor("")
    c.setCreator("")
    c.setProducer("")
    c.setTitle("Mímir - lead-lag setorial na B3")
    _page_1(c, data)
    _page_2(c, figs)
    _page_3(c, data, figs)
    _page_4(c, data, figs)
    _page_5(c, data, figs)
    c.save()
    _scrub_metadata(temp, path)
    temp.unlink()
    return _word_count(path)


def main() -> int:
    args = _args()
    _register_fonts()
    _plot_style()
    data = _read_data()
    _validate_data(data)
    figures = _make_figures(data, ROOT / "tmp/pdfs/relatorio_final_figuras")
    words = _build_pdf(args.saida, data, figures)
    reader = PdfReader(str(args.saida))
    if len(reader.pages) != 5:
        raise ValueError(f"esperadas 5 páginas; encontrado {len(reader.pages)}")
    # A extração inclui eixos, legendas e todas as células das tabelas. O limite
    # protege contra densidade excessiva sem cortar contexto necessário à banca.
    if not 900 <= words <= 1450:
        raise ValueError(f"contagem textual fora do alvo: {words}")
    print(f"PDF: {args.saida}")
    print(f"Páginas: {len(reader.pages)}")
    print(f"Palavras aproximadas: {words}")
    for figure in figures.values():
        print(f"Figura: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
