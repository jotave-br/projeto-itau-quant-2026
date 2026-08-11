"""
Testes de metricas e inferencia estatistica.
"""

import numpy as np
import pandas as pd
import pytest

from src import metricas
from src.config import EstrategiaConfig, MetricasConfig


def _pnl(media=0.0005, vol=0.01, dias=500, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-06", periods=dias)
    return pd.Series(rng.normal(media, vol, dias), index=idx)


def _benchmark(dias=500, seed=8):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-06", periods=dias)
    return pd.Series(rng.normal(0.0004, 0.015, dias), index=idx)


def test_bloco_do_bootstrap_maior_que_holding():
    """
    Se o bloco for menor que o holding, o bootstrap quebra exatamente a
    dependencia que ele deveria preservar, e volta a inflar a significancia.
    A garantia e dupla: a config respeita a regra, e a funcao recusa violacao.
    """
    assert MetricasConfig().bootstrap_bloco_dias > EstrategiaConfig().holding_dias

    with pytest.raises(ValueError, match="holding"):
        metricas.block_bootstrap(_pnl(), holding_dias=99)


def test_bootstrap_reproduz_com_seed_fixa():
    """Reprodutibilidade da inferencia."""
    pnl = _pnl()
    a = metricas.block_bootstrap(pnl, seed=42)
    b = metricas.block_bootstrap(pnl, seed=42)
    outro = metricas.block_bootstrap(pnl, seed=7)

    assert a == b
    assert a["ic_inferior"] != outro["ic_inferior"]
    # sanidade: o IC percentil cobre a media observada
    assert a["ic_inferior"] <= a["media_observada"] <= a["ic_superior"]


def test_long_only_calcula_retorno_anormal():
    """
    Avaliar long-only so pelo retorno bruto confundiria beta de mercado com
    alpha de lead-lag. Uma carteira que e metade do benchmark ganha dinheiro,
    mas o alpha dela e zero - e a leitura tem que ser essa.
    """
    bench = _benchmark()
    pnl = 0.5 * bench                       # so exposicao, nenhuma habilidade
    r = metricas.alpha_beta(pnl, bench)

    assert pnl.sum() > 0                    # "ganha dinheiro"
    assert r["beta"] == pytest.approx(0.5, abs=1e-9)
    assert abs(r["alpha_anual"]) < 1e-9     # mas nao ha alpha nenhum


def test_alpha_e_beta_no_nivel_da_estrategia():
    """
    Regressao da serie de P&L contra o benchmark. E o instrumento correto
    para carteira rebalanceada.
    """
    rng = np.random.default_rng(11)
    bench = _benchmark()
    alpha_verdadeiro, beta_verdadeiro = 0.0004, 0.8
    pnl = (alpha_verdadeiro + beta_verdadeiro * bench
           + pd.Series(rng.normal(0, 0.002, len(bench)), index=bench.index))

    r = metricas.alpha_beta(pnl, bench)
    assert r["beta"] == pytest.approx(beta_verdadeiro, abs=0.05)
    assert r["alpha_diario"] == pytest.approx(alpha_verdadeiro, abs=0.0003)
    assert r["alpha_p"] < 0.05


def test_car_no_nivel_do_sinal():
    """
    CAR e metrica de estudo de evento: cada disparo da lider e um evento, e o
    retorno anormal da seguidora e acumulado ao longo do holding. Nao deve ser
    usado como metrica da carteira.
    """
    idx = pd.bdate_range("2020-01-06", periods=20)
    bench = pd.Series(0.001, index=idx)
    # a seguidora rende benchmark + 1% ao dia: anormal de 1% por pregao
    ret = pd.DataFrame({"SG00": 0.011}, index=idx)
    eventos = pd.DataFrame({"data": [idx[5]], "ticker": ["SG00"]})

    car = metricas.car_por_evento(eventos, ret, bench, holding_dias=3)
    assert car.iloc[0] == pytest.approx(0.03)


def test_sharpe_zero_para_serie_constante():
    """Sanidade basica: serie sem variacao nao tem retorno ajustado a risco."""
    idx = pd.bdate_range("2020-01-06", periods=100)
    assert metricas.sharpe(pd.Series(0.001, index=idx)) == 0.0
    assert metricas.sharpe(pd.Series(0.0, index=idx)) == 0.0


def test_resumo_e_newey_west_coerentes():
    pnl = _pnl(media=0.001, vol=0.01)
    resumo = metricas.resumo_estrategia(pnl)
    nw = metricas.newey_west(pnl)

    assert resumo["dias"] == len(pnl)
    assert resumo["retorno_total"] == pytest.approx(pnl.sum())
    assert nw["media"] == pytest.approx(pnl.mean(), rel=1e-6)
    assert nw["t"] > 2                      # media construida bem positiva
    assert metricas.max_drawdown(pnl) <= 0
