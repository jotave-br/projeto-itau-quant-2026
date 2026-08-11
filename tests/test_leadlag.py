"""
Testes da estimacao lead-lag.

A tecnica principal e dado sintetico com resposta conhecida: se a seguidora
foi construida com beta 0,5 sobre a lider defasada, o estimador tem que achar
0,5 - e tem que achar nada nos lags errados e na direcao errada.
"""

import numpy as np
import pandas as pd

from src import backtest, leadlag
from src.config import LeadLagConfig, WalkForwardConfig

BETA_VERDADEIRO = 0.5


def _cenario(pregoes=700, seed=42):
    """
    Painel sintetico: SEGUE3 responde a LIDER3 com um pregao de atraso.
    TERCE3 e ruido independente, o controle negativo.
    """
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2015-01-01", periods=pregoes)
    lider = pd.Series(rng.normal(0, 0.02, pregoes), index=datas)
    ruido = pd.Series(rng.normal(0, 0.01, pregoes), index=datas)
    segue = BETA_VERDADEIRO * lider.shift(1) + ruido
    terce = pd.Series(rng.normal(0, 0.02, pregoes), index=datas)
    retornos = pd.DataFrame({"LIDER3": lider, "SEGUE3": segue, "TERCE3": terce})

    j = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                               pd.Timestamp("2018-12-31").date(),
                               WalkForwardConfig())[0]
    return retornos, j


def _pares(*duplas):
    linhas = [{
        "janela": "2017-01", "setor": "S", "subsetor": "X",
        "lider": a, "seguidora": b, "faixa_minima": 20,
        "posicao_lider": 1, "posicao_seguidora": 2,
    } for a, b in duplas]
    return pd.DataFrame(linhas)


def test_recupera_beta_construido():
    retornos, j = _cenario()
    rede = leadlag.estimar_rede(retornos, _pares(("LIDER3", "SEGUE3")), j)
    r = rede.iloc[0]

    assert abs(r["beta"] - BETA_VERDADEIRO) < 0.05
    assert r["p_valor"] < 1e-10
    assert r["n"] > 400
    assert r["direcao"] == "lider_para_seguidora"


def test_par_sem_relacao_nao_inventa_efeito():
    retornos, j = _cenario()
    rede = leadlag.estimar_rede(retornos, _pares(("LIDER3", "TERCE3")), j)
    assert abs(rede.iloc[0]["beta"]) < 0.05
    assert rede.iloc[0]["p_valor"] > 0.05


def test_lags_errados_nao_mostram_efeito():
    """O efeito foi construido no lag 1. Nos lags 0, 2 e 3 nao pode existir."""
    retornos, j = _cenario()
    lags = leadlag.estimar_lags(retornos, _pares(("LIDER3", "SEGUE3")), j)

    no_lag_certo = lags[lags["lag"] == 1].iloc[0]
    assert abs(no_lag_certo["beta"] - BETA_VERDADEIRO) < 0.05
    # Nos lags errados o criterio e em unidades de t, nao de beta: o ruido do
    # estimador escala com a razao das volatilidades, e um limiar absoluto de
    # beta reprovaria flutuacao amostral normal.
    for k in (0, 2, 3):
        fora = lags[lags["lag"] == k].iloc[0]
        assert abs(fora["estat_t"]) < 3, f"lag {k} nao deveria ter efeito"
        assert abs(fora["beta"]) < BETA_VERDADEIRO / 3


def test_direcao_invertida_nao_mostra_efeito():
    """
    A seguidora foi construida para reagir a lider. Na direcao invertida
    (seguidora defasada explicando a lider) nao pode haver nada - e se
    houvesse, a assimetria que fundamenta a hipotese nao existiria.
    """
    retornos, j = _cenario()
    inv = leadlag.estimar_direcao_invertida(
        retornos, _pares(("LIDER3", "SEGUE3")), j)
    assert inv.iloc[0]["direcao"] == "invertida"
    assert abs(inv.iloc[0]["estat_t"]) < 3
    assert abs(inv.iloc[0]["beta"]) < BETA_VERDADEIRO / 3
    assert inv.iloc[0]["p_valor"] > 0.05


def test_amostra_curta_reprova_sem_inventar_beta():
    """Par com poucos dias pareados sai com n registrado e beta NaN."""
    retornos, j = _cenario()
    curto = retornos.copy()
    curto.loc[curto.index[50:], "SEGUE3"] = np.nan

    rede = leadlag.estimar_rede(curto, _pares(("LIDER3", "SEGUE3")), j)
    assert rede.iloc[0]["n"] < LeadLagConfig().min_observacoes_par
    assert np.isnan(rede.iloc[0]["beta"])
    assert np.isnan(rede.iloc[0]["p_valor"])


def test_ticker_ausente_do_painel_sai_com_n_zero():
    retornos, j = _cenario()
    rede = leadlag.estimar_rede(retornos, _pares(("LIDER3", "SUMIU3")), j)
    assert rede.iloc[0]["n"] == 0
    assert np.isnan(rede.iloc[0]["beta"])


def test_betas_em_lote_batem_com_o_estimador_par_a_par():
    """
    O beta vetorizado e a mesma formula do OLS; a diferenca permitida e so
    numerica. Se divergirem, o placebo estaria comparando estimadores
    diferentes e a distribuicao nula nao serviria de referencia.
    """
    retornos, j = _cenario()
    pares = _pares(("LIDER3", "SEGUE3"), ("LIDER3", "TERCE3"),
                   ("TERCE3", "SEGUE3"))
    rede = leadlag.estimar_rede(retornos, pares, j)

    treino = retornos.loc[(retornos.index >= j.treino_inicio)
                          & (retornos.index < j.treino_fim)]
    lote = leadlag.betas_em_lote(treino, list(pares["lider"]),
                                 list(pares["seguidora"]), lag=1)

    np.testing.assert_allclose(lote["beta"], rede["beta"], atol=1e-12)
    np.testing.assert_array_equal(lote["n"], rede["n"])
