"""
Testes da carteira de safras sobrepostas.

Com holding de k pregoes e sinal diario, a operacao aberta na segunda se
sobrepoe as de terca e quarta. Se isso for tratado errado, duas coisas quebram:
o capital e multiplicado artificialmente, e a inferencia trata dias
dependentes como independentes, inflando a significancia.
"""

import numpy as np
import pandas as pd
import pytest

from src import backtest, estrategia
from src.config import EstrategiaConfig, WalkForwardConfig

K = EstrategiaConfig().holding_dias


def _cenario(n_pares=1, sinal_em=None, pregoes=60):
    """
    Janela de teste com calendario curto e sinais controlados.

    `sinal_em`: indices (no calendario) dos dias com retorno positivo da
    lider. None = todos os dias do teste.
    """
    cal = pd.bdate_range("2020-01-01", periods=pregoes)
    j = backtest.Janela(0, cal[0], cal[20], cal[20], cal[40])

    lideres = [f"LD{i:02d}" for i in range(n_pares)]
    seguidoras = [f"SG{i:02d}" for i in range(n_pares)]
    ret_sinal = pd.DataFrame(0.0, index=cal, columns=lideres + seguidoras)
    dias = cal if sinal_em is None else cal[sinal_em]
    ret_sinal.loc[ret_sinal.index.isin(dias), lideres] = 0.01

    vol = pd.DataFrame(0.20, index=cal, columns=seguidoras)
    pares = pd.DataFrame({"lider": lideres, "seguidora": seguidoras})
    return cal, j, pares, ret_sinal, vol


def test_holding_k_produz_pesos_um_sobre_k():
    """Cada safra diaria recebe peso 1/k."""
    cal, j, pares, ret_sinal, vol = _cenario(sinal_em=[25])
    pos, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal)

    cap = EstrategiaConfig().peso_maximo_por_posicao
    esperado = cap / K                     # 1 par: peso normalizado 1, teto cap
    ativos = pos[pos["SG00"] != 0]
    assert len(ativos) == K
    for v in ativos["SG00"]:
        assert v == pytest.approx(esperado)


def test_posicao_encerra_exatamente_apos_k_pregoes():
    """
    O holding e fixo e definido de antemao. Fechar no feeling bagunçaria a
    comparacao entre operacoes.
    """
    cal, j, pares, ret_sinal, vol = _cenario(sinal_em=[25])
    pos, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal)

    ativos = list(pos.index[pos["SG00"] != 0])
    # abre no fechamento de t+1 e mantem k snapshots consecutivos
    assert ativos == list(cal[26:26 + K])


def test_exposicao_bruta_nao_multiplica_capital():
    """
    Com k safras simultaneas de peso 1/k cada, a exposicao total permanece
    dentro do limite, nao vira k vezes o capital.
    """
    cal, j, pares, ret_sinal, vol = _cenario(n_pares=12, sinal_em=None)
    pos, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal)

    bruta = estrategia.exposicoes(pos)["bruta"]
    assert bruta.max() <= 1.0 + 1e-9
    # e as safras realmente se sobrepoem: em regime cheio ha k safras vivas
    assert bruta.max() == pytest.approx(1.0)


def test_saida_e_serie_diaria_unica_sem_contagem_dupla():
    """Todas as posicoes simultaneas agregadas em uma unica serie de P&L."""
    cal, j, pares, ret_sinal, vol = _cenario(n_pares=3, sinal_em=None)
    pos, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal)

    ret_pnl = pd.DataFrame(0.01, index=cal, columns=pos.columns)
    pnl, diag = estrategia.pnl_bruto(pos, ret_pnl)

    assert isinstance(pnl, pd.Series)
    assert pnl.index.is_unique and pnl.index.is_monotonic_increasing
    assert diag["posicao_dias_sem_retorno"] == 0
    # conferencia manual: pnl(d) = exposicao do fechamento anterior x 1%
    exposta = pos.shift(1).fillna(0.0).abs().sum(axis=1)
    assert pnl.sum() == pytest.approx((exposta * 0.01).sum())


def test_long_only_descarta_sinal_de_venda():
    cal, j, pares, ret_sinal, vol = _cenario(sinal_em=[25])
    ret_sinal.loc[cal[25], "LD00"] = -0.01          # sinal de venda

    ls, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal, modo="long_short")
    lo, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal, modo="long_only")

    assert (ls["SG00"] < 0).any()                   # long-short vende
    assert lo.empty or not (lo.get("SG00", pd.Series()) != 0).any()


def test_pernas_separadas_somam_o_total():
    cal, j, pares, ret_sinal, vol = _cenario(n_pares=4, sinal_em=None)
    ret_sinal.loc[:, ["LD01", "LD03"]] = -0.01      # metade vende
    pos, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal)

    rng = np.random.default_rng(1)
    ret_pnl = pd.DataFrame(rng.normal(0, 0.02, pos.shape),
                           index=pos.index, columns=pos.columns)
    pnl, _ = estrategia.pnl_bruto(pos, ret_pnl)
    pernas = estrategia.separar_pernas(pos, ret_pnl)

    assert (pernas["perna_comprada"] + pernas["perna_vendida"]).sum() == \
        pytest.approx(pnl.sum())
    assert (pernas["perna_vendida"] != 0).any()
