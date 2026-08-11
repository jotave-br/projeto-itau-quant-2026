"""
Estimacao das relacoes lead-lag:

    retorno_seguidora(t) = alpha + beta * retorno_lider(t - k) + erro

Beta positivo e significativo quer dizer que dia de alta da lider tende a ser
seguido por alta da seguidora. Nao comecamos por Granger, entropia de
transferencia ou rede neural: sao mais dificeis de fazer funcionar, mais
dificeis de defender quando funcionam, e escondem o mecanismo em vez de
mostrar. Erros-padrao HAC porque retorno diario tem heterocedasticidade e
alguma autocorrelacao.

Nada e interpolado. O painel chega com NaN onde nao houve dois pregoes
consecutivos observados (ver retornos.py) e a regressao usa so os dias em que a
lider defasada e a seguidora sao validas juntas - buraco reduz a amostra e o
`n` do par fica na tabela. A defasagem e aplicada depois do recorte de treino,
para o primeiro pregao da janela nao ir buscar retorno anterior a ela.

O recorte de treino e feito aqui dentro, a partir da janela, em vez de confiar
que quem chamou recortou antes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.backtest import Janela
from src.config import LeadLagConfig

COLUNAS_ESTIMATIVA = ("beta", "erro_padrao", "estat_t", "p_valor", "alpha",
                      "r2", "n")

# Schema fixo, mantido mesmo quando a janela nao tem par nenhum (setor pequeno
# numa faixa estreita e resultado legitimo). Assim o resto do pipeline segue
# sem caso especial.
COLUNAS_REDE = ("janela", "setor", "subsetor", "lider", "seguidora",
                "faixa_minima", "posicao_lider", "posicao_seguidora",
                "direcao", *COLUNAS_ESTIMATIVA, "lag",
                "data_inicio", "data_fim")


def estimar_par(
    ret_lider: pd.Series,
    ret_seguidora: pd.Series,
    lag: int,
    cfg: LeadLagConfig | None = None,
) -> dict:
    """
    Regressao de um par: seguidora(t) ~ alpha + beta * lider(t - lag).

    Par com menos de `cfg.min_observacoes_par` dias pareados devolve `n` e
    estatisticas NaN, para a reprovacao ficar visivel na tabela em vez de
    produzir uma estimativa instavel em amostra minuscula.
    """
    cfg = cfg or LeadLagConfig()
    x = ret_lider.shift(lag)
    dados = pd.DataFrame({"x": x, "y": ret_seguidora}).dropna()

    base = {c: float("nan") for c in COLUNAS_ESTIMATIVA}
    base.update({"lag": lag, "n": int(len(dados)),
                 "data_inicio": dados.index.min() if len(dados) else pd.NaT,
                 "data_fim": dados.index.max() if len(dados) else pd.NaT})
    if len(dados) < cfg.min_observacoes_par:
        return base

    if cfg.cov_type == "HAC":
        cov_kwds = {"maxlags": cfg.hac_maxlags}
    else:
        cov_kwds = None
    modelo = sm.OLS(dados["y"], sm.add_constant(dados["x"]))
    r = modelo.fit(cov_type=cfg.cov_type, cov_kwds=cov_kwds)

    base.update({
        "beta": float(r.params["x"]),
        "erro_padrao": float(r.bse["x"]),
        "estat_t": float(r.tvalues["x"]),
        "p_valor": float(r.pvalues["x"]),
        "alpha": float(r.params["const"]),
        "r2": float(r.rsquared),
    })
    return base


def _recorte_treino(retornos: pd.DataFrame, janela: Janela) -> pd.DataFrame:
    """Somente os pregoes do treino."""
    return retornos.loc[(retornos.index >= janela.treino_inicio)
                        & (retornos.index < janela.treino_fim)]


def estimar_rede(
    retornos: pd.DataFrame,
    pares: pd.DataFrame,
    janela: Janela,
    cfg: LeadLagConfig | None = None,
    lag: int | None = None,
    direcao_invertida: bool = False,
) -> pd.DataFrame:
    """
    A rede de uma janela: uma linha por par, com a estimativa e o contexto.

    Mexer em qualquer dado posterior a `janela.treino_fim` nao pode mudar um
    numero sequer da saida.

    Com `direcao_invertida=True` estima lider(t) ~ seguidora(t - lag), que e o
    placebo de direcionalidade: se a iliquida "preve" a liquida com a mesma
    forca, nao ha difusao, ha correlacao.

    Par cujo ticker nao existe no painel sai com n=0 e beta NaN, para a
    ausencia ficar registrada.
    """
    cfg = cfg or LeadLagConfig()
    lag = cfg.lag_principal if lag is None else lag
    treino = _recorte_treino(retornos, janela)

    linhas = []
    for _, par in pares.iterrows():
        meta = {
            "janela": par["janela"], "setor": par["setor"],
            "subsetor": par["subsetor"], "lider": par["lider"],
            "seguidora": par["seguidora"], "faixa_minima": par["faixa_minima"],
            "posicao_lider": par["posicao_lider"],
            "posicao_seguidora": par["posicao_seguidora"],
            "direcao": "invertida" if direcao_invertida else "lider_para_seguidora",
        }
        faltando = [t for t in (par["lider"], par["seguidora"])
                    if t not in treino.columns]
        if faltando:
            est = {c: float("nan") for c in COLUNAS_ESTIMATIVA}
            est.update({"lag": lag, "n": 0,
                        "data_inicio": pd.NaT, "data_fim": pd.NaT})
        elif direcao_invertida:
            est = estimar_par(treino[par["seguidora"]], treino[par["lider"]],
                              lag, cfg)
        else:
            est = estimar_par(treino[par["lider"]], treino[par["seguidora"]],
                              lag, cfg)
        linhas.append(meta | est)
    return pd.DataFrame(linhas, columns=list(COLUNAS_REDE))


def estimar_lags(
    retornos: pd.DataFrame,
    pares: pd.DataFrame,
    janela: Janela,
    cfg: LeadLagConfig | None = None,
) -> pd.DataFrame:
    """
    Rede no lag principal e nos alternativos, empilhadas com a coluna `lag`.

    O lag 0 entra como diagnostico, nao como candidato a sinal: se o efeito
    contemporaneo domina, o que a regressao defasada captura pode ser vazamento
    de correlacao comum.
    """
    cfg = cfg or LeadLagConfig()
    lags = (cfg.lag_principal, *cfg.lags_robustez)
    return pd.concat(
        [estimar_rede(retornos, pares, janela, cfg, lag=k) for k in lags],
        ignore_index=True)


def estimar_direcao_invertida(
    retornos: pd.DataFrame,
    pares: pd.DataFrame,
    janela: Janela,
    cfg: LeadLagConfig | None = None,
) -> pd.DataFrame:
    """Placebo de direcao: seguidora(t - lag) -> lider(t), no lag principal."""
    return estimar_rede(retornos, pares, janela, cfg, direcao_invertida=True)


def betas_em_lote(
    treino: pd.DataFrame,
    lideres: list[str],
    seguidoras: list[str],
    lag: int,
    min_observacoes: int = 100,
) -> pd.DataFrame:
    """
    Beta e t classico de muitos pares de uma vez, vetorizado.

    Existe por causa dos placebos: 500 embaralhamentos por janela dao centenas
    de milhares de regressoes, inviaveis uma a uma no statsmodels. O beta e o
    mesmo do OLS; o t usa erro-padrao classico em vez de HAC, o que nao vicia a
    comparacao real x placebo desde que os dois lados usem este mesmo
    estimador.

    Quem chama e que faz o recorte de treino, porque o placebo reusa o mesmo
    recorte centenas de vezes.
    """
    x = treino[lideres].to_numpy(dtype=float)
    x = np.vstack([np.full((lag, len(lideres)), np.nan), x[:len(x) - lag]])
    y = treino[seguidoras].to_numpy(dtype=float)

    valido = ~np.isnan(x) & ~np.isnan(y)
    n = valido.sum(axis=0)
    xv = np.where(valido, x, 0.0)
    yv = np.where(valido, y, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        n_seguro = np.where(n > 0, n, 1)
        mx = xv.sum(axis=0) / n_seguro
        my = yv.sum(axis=0) / n_seguro
        sxx = (np.where(valido, (x - mx) ** 2, 0.0)).sum(axis=0)
        sxy = (np.where(valido, (x - mx) * (y - my), 0.0)).sum(axis=0)
        syy = (np.where(valido, (y - my) ** 2, 0.0)).sum(axis=0)

        beta = sxy / sxx
        gl = n - 2
        var_residuo = (syy - beta * sxy) / np.where(gl > 0, gl, 1)
        erro_padrao = np.sqrt(var_residuo / sxx)
        t = beta / erro_padrao

    poucos = n < min_observacoes
    beta = np.where(poucos, np.nan, beta)
    t = np.where(poucos, np.nan, t)
    return pd.DataFrame({"lider": lideres, "seguidora": seguidoras,
                         "beta": beta, "estat_t": t, "n": n})
