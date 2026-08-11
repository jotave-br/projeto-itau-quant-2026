"""
Metricas e inferencia sobre series ja apuradas.

CAR e alpha vem de literaturas diferentes e medem coisas diferentes; misturar
os dois e erro classico, entao os dois niveis ficam separados aqui:

  nivel do sinal      cada disparo da lider e um evento, e acumulamos o retorno
                      anormal da seguidora ao longo do holding -> CAR.
  nivel da estrategia regressao do P&L diario contra o benchmark -> alpha de
                      Jensen, beta, Sharpe. Para carteira rebalanceada e este o
                      instrumento certo, nao o CAR.

Vale principalmente para a long-only: carteira que ganha dinheiro mas nao passa
do benchmark depois de custo e de ajuste de beta esta entregando exposicao a
mercado, nao alpha de lead-lag.

A inferencia principal e bootstrap por blocos sobre a serie diaria, com bloco
maior que o holding para preservar a autocorrelacao das safras sobrepostas.
Newey-West entra como segunda opiniao. Bootstrap iid e teste t ingenuo ficam de
fora do resultado principal: tratam dias vizinhos como independentes, o que
aqui e falso, e inflam a significancia justamente onde a amostra e curta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import SEED, MetricasConfig


def sharpe(pnl: pd.Series, cfg: MetricasConfig | None = None) -> float:
    """
    Sharpe anualizado sobre o retorno em excesso simples (rf da config).

    Serie sem variacao devolve 0.0: sem risco medido nao ha retorno ajustado a
    risco.
    """
    cfg = cfg or MetricasConfig()
    excesso = pnl - cfg.taxa_livre_risco_anual / 252
    vol = float(excesso.std())
    # tolerancia em vez de zero exato: o std de uma serie constante sai como
    # residuo de ponto flutuante (~1e-19) e fabricaria um Sharpe astronomico.
    if not np.isfinite(vol) or vol < 1e-12:
        return 0.0
    return float(excesso.mean() / vol * np.sqrt(252))


def max_drawdown(pnl: pd.Series) -> float:
    """
    Maior queda do P&L acumulado (soma simples), como numero negativo.

    Soma e nao composicao porque a serie e fracao do capital de referencia por
    dia, o mesmo espaco em que custos e pesos foram definidos.
    """
    acumulado = pnl.cumsum()
    return float((acumulado - acumulado.cummax()).min())


def resumo_estrategia(
    pnl: pd.Series,
    cfg: MetricasConfig | None = None,
    turnover: pd.Series | None = None,
) -> dict:
    """Metricas descritivas de uma serie diaria de P&L."""
    cfg = cfg or MetricasConfig()
    return {
        "dias": int(pnl.notna().sum()),
        "retorno_total": float(pnl.sum()),
        "retorno_anual_medio": float(pnl.mean() * 252),
        "vol_anualizada": float(pnl.std() * np.sqrt(252)),
        "sharpe": sharpe(pnl, cfg),
        "max_drawdown": max_drawdown(pnl),
        "frac_dias_ativos": float((pnl != 0).mean()),
        "turnover_medio": (float(turnover.mean())
                           if turnover is not None else None),
    }


def alpha_beta(
    pnl: pd.Series,
    benchmark: pd.Series,
    cfg: MetricasConfig | None = None,
) -> dict:
    """
    Regressao do P&L diario contra o benchmark, com erros HAC.

    O alpha diario vira anual por multiplicacao simples (x252), coerente com a
    soma simples usada no P&L acumulado.
    """
    cfg = cfg or MetricasConfig()
    dados = pd.DataFrame({"pnl": pnl, "bench": benchmark}).dropna()
    if len(dados) < 30:
        return {"n": int(len(dados)), "alpha_diario": float("nan"),
                "alpha_anual": float("nan"), "beta": float("nan"),
                "alpha_t": float("nan"), "alpha_p": float("nan")}
    r = sm.OLS(dados["pnl"], sm.add_constant(dados["bench"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": cfg.hac_maxlags})
    return {
        "n": int(len(dados)),
        "alpha_diario": float(r.params["const"]),
        "alpha_anual": float(r.params["const"] * 252),
        "beta": float(r.params["bench"]),
        "alpha_t": float(r.tvalues["const"]),
        "alpha_p": float(r.pvalues["const"]),
    }


def car_por_evento(
    eventos: pd.DataFrame,
    retornos: pd.DataFrame,
    benchmark: pd.Series,
    holding_dias: int,
    defasagem_dias: int = 1,
) -> pd.Series:
    """
    Retorno anormal acumulado por evento (nivel do sinal).

    `eventos` tem colunas `data` e `ticker`, uma linha por disparo da lider. O
    retorno anormal da seguidora (retorno menos benchmark) e acumulado do
    pregao `defasagem_dias` apos o evento ate o fim do holding. Nao e metrica
    de carteira: aqui evento sobreposto conta duas vezes de proposito.
    """
    anormal = retornos.sub(benchmark, axis=0)
    cal = retornos.index
    cars = []
    for _, ev in eventos.iterrows():
        if ev["ticker"] not in anormal.columns or ev["data"] not in cal:
            cars.append(float("nan"))
            continue
        pos = cal.get_loc(ev["data"])
        ini = pos + defasagem_dias + 1          # primeiro retorno auferido
        fim = ini + holding_dias
        cars.append(float(anormal[ev["ticker"]].iloc[ini:fim].sum())
                    if ini < len(cal) else float("nan"))
    return pd.Series(cars, index=eventos.index, name="car")


def block_bootstrap(
    pnl: pd.Series,
    cfg: MetricasConfig | None = None,
    seed: int = SEED,
    holding_dias: int | None = None,
) -> dict:
    """
    Bootstrap por blocos moveis da media diaria do P&L.

    Blocos contiguos de `bootstrap_bloco_dias` preservam a autocorrelacao das
    safras sobrepostas. Com `holding_dias` informado, bloco menor ou igual ao
    holding levanta erro: quebraria justamente a dependencia que o metodo
    existe para preservar.

    Devolve IC percentil e a cauda inferior da distribuicao bootstrap (fracao
    de reamostragens com media <= 0), mantida no campo historico
    `p_unilateral`. Nao e um teste bootstrap centralizado sob H0.
    """
    cfg = cfg or MetricasConfig()
    bloco = cfg.bootstrap_bloco_dias
    if holding_dias is not None and bloco <= holding_dias:
        raise ValueError(
            f"bloco de {bloco} pregoes <= holding de {holding_dias}: o "
            "bootstrap quebraria a dependencia que deveria preservar")

    serie = pnl.dropna().to_numpy()
    n = len(serie)
    if n < bloco * 2:
        return {"n": n, "media_observada": float(np.mean(serie)) if n else float("nan"),
                "ic_inferior": float("nan"), "ic_superior": float("nan"),
                "p_unilateral": float("nan"), "reamostragens": 0}

    rng = np.random.default_rng(seed)
    n_blocos = int(np.ceil(n / bloco))
    medias = np.empty(cfg.bootstrap_n)
    lote = 1000
    offsets = np.arange(bloco)
    for i in range(0, cfg.bootstrap_n, lote):
        m = min(lote, cfg.bootstrap_n - i)
        inicios = rng.integers(0, n - bloco + 1, size=(m, n_blocos))
        idx = (inicios[:, :, None] + offsets).reshape(m, -1)[:, :n]
        medias[i:i + m] = serie[idx].mean(axis=1)

    alpha = cfg.bootstrap_alpha
    return {
        "n": n,
        "media_observada": float(np.mean(serie)),
        "ic_inferior": float(np.quantile(medias, alpha / 2)),
        "ic_superior": float(np.quantile(medias, 1 - alpha / 2)),
        "p_unilateral": float((medias <= 0).mean()),
        "reamostragens": int(cfg.bootstrap_n),
        "bloco_dias": bloco,
    }


def newey_west(pnl: pd.Series, cfg: MetricasConfig | None = None) -> dict:
    """
    t da media diaria com erro-padrao HAC (Newey-West).

    Segunda opiniao, com premissas diferentes das do bootstrap por blocos.
    """
    cfg = cfg or MetricasConfig()
    serie = pnl.dropna()
    if len(serie) < 30:
        return {"n": int(len(serie)), "t": float("nan"), "p": float("nan")}
    r = sm.OLS(serie.to_numpy(), np.ones(len(serie))).fit(
        cov_type="HAC", cov_kwds={"maxlags": cfg.hac_maxlags})
    return {"n": int(len(serie)), "media": float(r.params[0]),
            "t": float(r.tvalues[0]), "p": float(r.pvalues[0])}
