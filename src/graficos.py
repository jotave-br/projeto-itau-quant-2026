"""
Figuras a partir das tabelas ja calculadas.

Nada de estatistica nova aqui: todo numero que aparece num grafico saiu de uma
tabela em outputs/runs/. Grafico que calcula a propria estatistica esconde
metodologia. Soma acumulada e mediana para exibicao, tudo bem.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_CORES_FAIXA = {"top20": "#1a5276", "top40": "#c0392b",
                "top60": "#7d8a2e", "top100": "#8e6c3a"}


def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def fig_efeito_por_faixa(rede: pd.DataFrame, faixas, caminho):
    """
    Beta mediano no lag principal, por faixa cumulativa e por janela. Efeito
    que cresce ao afrouxar a faixa (mais iliquidos) e o padrao do preco velho.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for f in sorted(faixas):
        sub = rede[rede["faixa_minima"] <= f]
        serie = sub.groupby("janela")["beta"].median()
        ax.plot(serie.index, serie.values, label=f"top{f}", linewidth=1.6,
                color=_CORES_FAIXA.get(f"top{f}"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Beta mediano da rede por faixa de liquidez (lag 1)")
    ax.set_ylabel("beta mediano da janela")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.legend(title="faixa")
    return _salvar(fig, caminho)


def fig_pnl_acumulado(pnl: pd.DataFrame, cenarios: pd.DataFrame, caminho,
                      titulo: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pnl.index, pnl["bruto"].cumsum(), label="bruto", linewidth=1.8,
            color="#1a5276")
    ax.plot(pnl.index, pnl["liquido"].cumsum(),
            label="liquido (aluguel base)", linewidth=1.8, color="#c0392b")
    for col in cenarios.columns:
        ax.plot(cenarios.index, cenarios[col].cumsum(), linewidth=0.8,
                alpha=0.5, label=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(titulo)
    ax.set_ylabel("P&L acumulado (fracao do capital)")
    ax.legend(fontsize=7)
    return _salvar(fig, caminho)


def fig_real_vs_placebo(rede_top: pd.DataFrame, placebo_dist: pd.DataFrame,
                        invertida: pd.DataFrame, caminho):
    """
    Comparacao exploratoria entre betas por par e medianas dos embaralhamentos.

    As unidades amostrais diferem, portanto a figura nao sustenta inferencia;
    o teste formal usa o resumo do placebo por janela.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rede_top["beta"].dropna(), bins=60, density=True, alpha=0.55,
            label="pares reais (lag 1)", color="#1a5276")
    ax.hist(invertida["beta"].dropna(), bins=60, density=True, alpha=0.45,
            label="direcao invertida", color="#7d8a2e")
    ax.hist(placebo_dist["beta_mediano"].dropna(), bins=60, density=True,
            alpha=0.45, label="placebo (mediana por embaralhamento)",
            color="#8e6c3a")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Betas: reais x direcao invertida x placebo")
    ax.set_xlabel("beta")
    ax.legend()
    return _salvar(fig, caminho)


def fig_robustez_por_janela(fdr_resumo: pd.DataFrame, caminho):
    fig, ax = plt.subplots(figsize=(10, 4))
    principal = fdr_resumo[fdr_resumo["faixa"] == "top40"]
    ax.bar(principal["janela"], principal["aprovados_fdr"],
           color="#1a5276", label="aprovados FDR (q=10%)")
    ax.bar(principal["janela"], principal["aprovados_fdr_beta_positivo"],
           color="#c0392b", label="dos quais beta positivo")
    ax.set_title("Aprovacoes FDR por janela (top40)")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.legend()
    return _salvar(fig, caminho)


def fig_histograma_pvalores(rede: pd.DataFrame, caminho):
    """
    P-valores do lag principal. Sob a nula global o histograma e uniforme;
    excesso concentrado perto de zero e o sinal de efeito.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rede["p_valor"].dropna(), bins=40, color="#1a5276", alpha=0.8)
    ax.set_title("Distribuicao dos p-valores (todas as janelas, lag 1)")
    ax.set_xlabel("p-valor")
    return _salvar(fig, caminho)


def fig_exposicao_turnover(exposicoes: pd.DataFrame, turnover: pd.Series,
                           caminho, titulo: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(exposicoes.index, exposicoes["bruta"], linewidth=0.9,
            color="#1a5276", label="bruta")
    a1.plot(exposicoes.index, exposicoes["liquida"], linewidth=0.9,
            color="#c0392b", label="liquida")
    a1.set_ylabel("exposicao")
    a1.legend(fontsize=8)
    a1.set_title(titulo)
    a2.plot(turnover.index, turnover.values, linewidth=0.7, color="#5d6d7e")
    a2.set_ylabel("turnover diario")
    return _salvar(fig, caminho)
