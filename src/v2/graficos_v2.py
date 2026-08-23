"""Figuras da V2 para o relatório de cinco páginas.

Como em `src/graficos`, nada de estatística nova: todo número aqui saiu de uma
tabela em outputs/runs. As figuras são desenhadas para 16:9 e leitura em tela
cheia, sem zoom.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

AZUL = "#2a78d6"
LARANJA = "#eb6834"
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

RAMPA_AZUL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab"]
CMAP_AZUL = LinearSegmentedColormap.from_list("azul_seq", RAMPA_AZUL)

FONTE = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def _base(largura=10.0, altura=5.625):
    """Tela 16:9, superfície e tipografia do sistema."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONTE,
            "text.color": TINTA,
            "axes.labelcolor": TINTA_2,
            "xtick.color": MUDO,
            "ytick.color": MUDO,
            "axes.edgecolor": EIXO,
        }
    )
    fig, ax = plt.subplots(figsize=(largura, altura), dpi=200)
    fig.patch.set_facecolor(SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_linewidth(0.8)
    return fig, ax


def _salvar(fig, caminho):
    # Faixa superior reservada: o cabeçalho é desenhado fora dos eixos.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    fig.savefig(caminho, dpi=200, facecolor=SUPERFICIE)
    plt.close(fig)
    return caminho


def _titulo(ax, titulo: str, subtitulo: str | None = None) -> None:
    ax.text(0.0, 1.14, titulo, transform=ax.transAxes, fontsize=14,
            fontweight="600", color=TINTA, va="bottom")
    if subtitulo:
        ax.text(0.0, 1.045, subtitulo, transform=ax.transAxes, fontsize=10,
                color=TINTA_2, va="bottom")


ROTULO_CLASSE = {
    "nao_especifico": "não específico",
    "especifico_positiva": "positiva",
    "especifico_negativa": "negativa",
    "especifico_neutra": "neutra",
}


def fig_matriz_confusao(matriz: pd.DataFrame, caminho):
    """Heatmap sequencial de uma hue. As duas primeiras linhas contam a história."""
    ordem = ["especifico_positiva", "especifico_negativa", "especifico_neutra",
             "nao_especifico"]
    m = matriz.reindex(index=ordem, columns=ordem).fillna(0).astype(int)
    fig, ax = _base(9.0, 5.4)

    maximo = int(m.to_numpy().max())
    ax.imshow(m.to_numpy(), cmap=CMAP_AZUL, vmin=0, vmax=maximo, aspect="auto")

    rotulos = [ROTULO_CLASSE[c] for c in ordem]
    ax.set_xticks(range(len(ordem)), rotulos, fontsize=10)
    ax.set_yticks(range(len(ordem)), rotulos, fontsize=10)
    ax.set_xlabel("classificação da IA", fontsize=10, color=TINTA_2, labelpad=10)
    ax.set_ylabel("gold do painel", fontsize=10, color=TINTA_2, labelpad=10)

    ax.set_xticks(np.arange(-0.5, len(ordem), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordem), 1), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)

    for i in range(len(ordem)):
        for j in range(len(ordem)):
            valor = int(m.iat[i, j])
            claro = valor > maximo * 0.55
            ax.text(
                j,
                i,
                str(valor),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="600" if i == j else "400",
                color=SUPERFICIE if claro else (TINTA if valor else MUDO),
            )

    _titulo(
        ax,
        "A amostra auditada revela excesso de sinais direcionais",
        "Dos 56 neutros no gold, 26 viraram direcionais; foram 31 erros no total",
    )
    return _salvar(fig, caminho)


def fig_escada_conservadorismo(contagens: pd.Series, destaque: str, caminho):
    """Barras de um valor por rotulador, com o classificador em destaque."""
    serie = contagens.sort_values()
    fig, ax = _base(9.0, 5.0)

    cores = [LARANJA if nome == destaque else AZUL for nome in serie.index]
    barras = ax.barh(range(len(serie)), serie.to_numpy(), color=cores, height=0.62)
    for barra in barras:
        barra.set_capstyle("round")

    ax.set_yticks(range(len(serie)), list(serie.index), fontsize=11)
    ax.set_xlabel("documentos com direção atribuída (de 90)", fontsize=10,
                  color=TINTA_2, labelpad=10)
    ax.xaxis.grid(True, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)

    for indice, valor in enumerate(serie.to_numpy()):
        ax.text(valor + 1.0, indice, str(int(valor)), va="center", fontsize=11,
                color=TINTA, fontweight="600")

    ax.set_xlim(0, float(serie.max()) * 1.16)
    _titulo(
        ax,
        "O classificador local atribui mais direções que o painel",
        "Mesmos 90 documentos e protocolo; as avaliações por LLM não são independentes",
    )
    return _salvar(fig, caminho)


def fig_curvas_bracos(series: dict[str, pd.Series], principal: str,
                      placebo: str, caminho):
    """Acumulado dos braços. Ênfase em dois; os demais recuam para cinza."""
    fig, ax = _base(10.0, 5.2)

    for nome, serie in series.items():
        if nome in (principal, placebo):
            continue
        ax.plot(serie.index, serie.cumsum().to_numpy() * 100, linewidth=1.1,
                color=MUDO, alpha=0.55, zorder=1)

    estilos = {principal: (AZUL, "IA + rede (sinal real)"),
               placebo: (LARANJA, "seguidora aleatória (placebo)")}
    for nome, (cor, rotulo) in estilos.items():
        acumulado = series[nome].cumsum() * 100
        ax.plot(acumulado.index, acumulado.to_numpy(), linewidth=2.0, color=cor,
                label=rotulo, zorder=3)
        ax.annotate(
            f"{acumulado.iloc[-1]:+.1f}%",
            xy=(acumulado.index[-1], acumulado.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=11,
            fontweight="600",
            color=cor,
            va="center",
        )

    ax.axhline(0, color=EIXO, linewidth=0.9, zorder=2)
    ax.yaxis.grid(True, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("P&L acumulado (% do capital)", fontsize=10, color=TINTA_2)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.tick_params(labelsize=9)
    ax.margins(x=0.06)
    _titulo(
        ax,
        "O sinal principal não se separa da seguidora aleatória",
        "Mesmos eventos; linha cinza: IA sem rede. H=3, líquido de custos",
    )
    return _salvar(fig, caminho)


def fig_intervalos(resumo: pd.DataFrame, caminho, rotulos: dict[str, str]):
    """Estimativa e intervalo por braço: mostra largura, não só o ponto."""
    base = resumo[
        resumo["h"].eq(3) & ~resumo["braco"].eq("rede_sem_ia")
    ].copy()
    base["rotulo"] = base["braco"].map(rotulos).fillna(base["braco"])
    base = base.iloc[::-1]
    fig, ax = _base(9.0, 4.6)

    escala = 1e5
    y = range(len(base))
    for indice, (_, linha) in enumerate(base.iterrows()):
        baixo = linha["ic_inferior"] * escala
        alto = linha["ic_superior"] * escala
        ax.plot([baixo, alto], [indice, indice], linewidth=2.0, color=AZUL,
                solid_capstyle="round", zorder=2)
        ax.plot([linha["media_diaria"] * escala], [indice], marker="o",
                markersize=9, color=AZUL, markeredgecolor=SUPERFICIE,
                markeredgewidth=2, zorder=3)

    ax.axvline(0, color=LARANJA, linewidth=1.4, zorder=1)
    ax.text(0.15, -0.55, "zero", color=LARANJA, fontsize=10,
            fontweight="600", va="center")

    ax.set_yticks(list(y), base["rotulo"].tolist(), fontsize=10.5)
    ax.set_xlabel("retorno médio diário (×10⁻⁵ do capital)", fontsize=10,
                  color=TINTA_2, labelpad=10)
    ax.xaxis.grid(True, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylim(-0.9, len(base) - 0.4)
    p_randomizacao = resumo.loc[
        resumo["braco"].eq("ia_mais_rede") & resumo["h"].eq(3),
        "p_randomizacao_seguidora",
    ].dropna()
    p_texto = (
        f"p={float(p_randomizacao.iloc[0]):.3f}".replace(".", ",")
        if not p_randomizacao.empty
        else "p não disponível"
    )
    _titulo(
        ax,
        "O efeito combinado de IA e rede não se separa de zero",
        f"99 operações; 500 redes aleatórias, {p_texto}. IA sem rede é negativa",
    )
    return _salvar(fig, caminho)
