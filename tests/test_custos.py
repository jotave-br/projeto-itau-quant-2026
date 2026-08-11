"""
Testes de custos de transacao.
"""

import numpy as np
import pandas as pd
import pytest

from src import custos, estrategia
from src.config import CustosConfig


def _carteira(pregoes=30, seed=2):
    """Posicoes com pernas compradas e vendidas e P&L bruto nao trivial."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-06", periods=pregoes)
    posicoes = pd.DataFrame({
        "COMP3": rng.uniform(0.0, 0.1, pregoes),
        "VEND3": -rng.uniform(0.0, 0.1, pregoes),
    }, index=idx)
    ret = pd.DataFrame(rng.normal(0, 0.02, posicoes.shape),
                       index=idx, columns=posicoes.columns)
    pnl, _ = estrategia.pnl_bruto(posicoes, ret)
    turnover = estrategia.turnover_diario(posicoes)
    return posicoes, pnl, turnover


def test_liquido_nunca_maior_que_bruto():
    """
    Invariante fundamental: custo nunca aumenta retorno. Se este teste falhar,
    ha erro de sinal em algum lugar - provavelmente na perna vendida.
    """
    posicoes, pnl, turnover = _carteira()
    for taxa in CustosConfig().aluguel_cenarios_anual:
        liquido = custos.aplicar_custos(pnl, posicoes, turnover,
                                        taxa_aluguel_anual=taxa)
        assert (liquido <= pnl + 1e-12).all()


def test_aluguel_incide_somente_na_perna_vendida():
    """
    Aluguel (BTC) e o custo de tomar a acao emprestada para vender descoberto.
    Posicao comprada nao paga aluguel.
    """
    idx = pd.bdate_range("2020-01-06", periods=5)
    so_comprada = pd.DataFrame({"COMP3": [0.5] * 5}, index=idx)
    so_vendida = pd.DataFrame({"VEND3": [-0.5] * 5}, index=idx)

    assert custos.custo_aluguel(so_comprada, 0.10).sum() == pytest.approx(0.0)
    esperado = 0.5 * 0.10 / CustosConfig().dias_uteis_ano * 5
    assert custos.custo_aluguel(so_vendida, 0.10).sum() == pytest.approx(esperado)


def test_long_only_nao_paga_aluguel():
    """Consequencia direta do teste anterior, verificada no nivel da estrategia."""
    posicoes, pnl, turnover = _carteira()
    long_only = posicoes.clip(lower=0.0)
    pnl_lo = pnl * 0 + 0.001

    com_aluguel = custos.aplicar_custos(pnl_lo, long_only, turnover,
                                        taxa_aluguel_anual=0.20)
    sem_aluguel = custos.aplicar_custos(pnl_lo, long_only, turnover,
                                        taxa_aluguel_anual=0.0)
    pd.testing.assert_series_equal(com_aluguel, sem_aluguel)


def test_taxa_anual_convertida_pro_rata_por_dia_util():
    """Aluguel de 5% a.a. sobre 3 pregoes nao custa 5%."""
    idx = pd.bdate_range("2020-01-06", periods=3)
    vendida = pd.DataFrame({"VEND3": [-1.0] * 3}, index=idx)
    total = custos.custo_aluguel(vendida, 0.05).sum()
    assert total == pytest.approx(0.05 * 3 / 252)
    assert total < 0.05 / 10


def test_cenarios_de_aluguel_sao_monotonicos():
    """Cenario de aluguel maior nunca produz P&L liquido maior."""
    posicoes, pnl, turnover = _carteira()
    series = custos.cenarios_aluguel(pnl, posicoes, turnover)

    taxas = CustosConfig().aluguel_cenarios_anual
    acumulados = [series[f"aluguel_{t:.0%}"].sum() for t in taxas]
    assert all(a >= b - 1e-12 for a, b in zip(acumulados, acumulados[1:]))


def test_ir_ilustrativo_so_tributa_ano_positivo():
    pnl_anual = pd.Series({2020: 0.10, 2021: -0.05, 2022: 0.20})
    liquido = custos.ir_ilustrativo(pnl_anual)
    assert liquido[2020] == pytest.approx(0.10 * 0.85)
    assert liquido[2021] == pytest.approx(-0.05)      # prejuizo nao vira credito
    assert liquido[2022] == pytest.approx(0.20 * 0.85)
