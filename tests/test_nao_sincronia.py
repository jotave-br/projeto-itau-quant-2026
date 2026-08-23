"""
Testes do controle de nao-sincronia e dos placebos.

O placebo e verificado com um cenario onde a resposta e conhecida por
construcao: cada seguidora responde so ao seu proprio lider. Embaralhar as
seguidoras tem que destruir o efeito - se nao destruir, o placebo nao mede
nada.
"""

import dataclasses

import numpy as np
import pandas as pd

from src import backtest, nao_sincronia, pares as mod_pares
from src.config import NaoSincroniaConfig, PlaceboConfig, WalkForwardConfig


def _janela():
    return backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                  pd.Timestamp("2018-12-31").date(),
                                  WalkForwardConfig())[0]


def _paineis(pregoes=700):
    datas = pd.bdate_range("2015-01-01", periods=pregoes)
    rng = np.random.default_rng(3)
    volume = pd.DataFrame({
        "SAUDA3": np.full(pregoes, 1e8),
        "RARO3": rng.uniform(1e5, 1e7, pregoes),
    }, index=datas)
    negocios = pd.DataFrame(1000, index=datas, columns=volume.columns)
    precos = pd.DataFrame(
        rng.uniform(9, 11, (pregoes, 2)), index=datas, columns=volume.columns)
    return datas, volume, negocios, precos


def test_marca_poucos_negocios_e_preco_travado():
    datas, volume, negocios, precos = _paineis()
    j = _janela()
    negocios.loc[datas[100], "SAUDA3"] = 3          # quase nao negociou
    precos.loc[datas[201], "SAUDA3"] = precos.loc[datas[200], "SAUDA3"]

    m = nao_sincronia.filtrar_dias_preco_velho(
        volume, negocios, precos, j, NaoSincroniaConfig())

    assert bool(m.loc[datas[100], "SAUDA3"])        # poucos negocios
    assert bool(m.loc[datas[201], "SAUDA3"])        # fechamento travado
    assert not bool(m.loc[datas[200], "SAUDA3"])    # o 1o dia da sequencia nao
    assert not bool(m.loc[datas[102], "SAUDA3"])    # dia saudavel


def test_volume_constante_nao_marca_a_serie_inteira():
    """
    O limiar de volume raro e o percentil do proprio ticker. Num papel de
    volume constante o percentil coincide com o volume - e dia igual ao normal
    do papel nao e rarefeito.
    """
    _, volume, negocios, precos = _paineis()
    m = nao_sincronia.filtrar_dias_preco_velho(
        volume, negocios, precos, _janela(), NaoSincroniaConfig())
    assert not m["SAUDA3"].any()
    frac = m["RARO3"].mean()
    assert 0 < frac < 0.25


def test_percentil_de_volume_usa_somente_a_janela():
    """Anti-lookahead: volume gigante depois do treino nao muda a mascara."""
    datas, volume, negocios, precos = _paineis()
    j = _janela()
    antes = nao_sincronia.filtrar_dias_preco_velho(
        volume, negocios, precos, j, NaoSincroniaConfig())

    adulterado = volume.copy()
    adulterado.loc[adulterado.index >= j.treino_fim] = 1e15
    depois = nao_sincronia.filtrar_dias_preco_velho(
        adulterado, negocios, precos, j, NaoSincroniaConfig())
    pd.testing.assert_frame_equal(antes, depois)


def test_subconjunto_sempre_negociado_e_por_janela():
    datas, volume, negocios, precos = _paineis()
    j = _janela()
    volume.loc[datas[50], "RARO3"] = 0.0             # buraco dentro do treino
    # zerar depois do treino nao pode tirar ninguem do subconjunto
    volume.loc[volume.index >= j.treino_fim, "SAUDA3"] = 0.0

    assert nao_sincronia.subconjunto_sempre_negociado(volume, j) == ["SAUDA3"]


def _cenario_placebo(n_duplas=4, pregoes=700, seed=11):
    """
    n_duplas lideres e seguidoras num unico (setor, subsetor). Cada seguidora
    responde apenas ao seu lider: o alinhamento e a unica fonte do efeito.
    """
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2015-01-01", periods=pregoes)
    series = {}
    linhas = []
    for i in range(n_duplas):
        lider = pd.Series(rng.normal(0, 0.02, pregoes), index=datas)
        segue = 0.5 * lider.shift(1) + pd.Series(
            rng.normal(0, 0.005, pregoes), index=datas)
        series[f"LD{i:02d}"] = lider
        series[f"SG{i:02d}"] = segue
        linhas.append({
            "janela": "2017-01", "setor": "S", "subsetor": "X",
            "lider": f"LD{i:02d}", "seguidora": f"SG{i:02d}",
            "emissor_lider": f"LD{i:02d}", "emissor_seguidora": f"SG{i:02d}",
            "posicao_lider": i + 1, "posicao_seguidora": n_duplas + i + 1,
            "liquidez_lider": 1e9, "liquidez_seguidora": 1e8,
            "faixa_minima": 20,
        })
    return pd.DataFrame(series), pd.DataFrame(linhas)


def test_embaralhamento_preserva_grupo_e_regras_duras():
    _, pares = _cenario_placebo()
    rng = np.random.default_rng(0)
    emb = mod_pares.pares_placebo_embaralhados(pares, rng)

    assert list(emb["lider"]) == list(pares["lider"])
    assert sorted(emb["seguidora"]) == sorted(pares["seguidora"])
    assert (emb["setor"] == pares["setor"]).all()
    assert (emb["lider"] != emb["seguidora"]).all()
    assert (emb["emissor_lider"] != emb["emissor_seguidora"]).all()


def test_embaralhamento_e_deterministico_com_a_mesma_seed():
    _, pares = _cenario_placebo()
    a = mod_pares.pares_placebo_embaralhados(pares, np.random.default_rng(7))
    b = mod_pares.pares_placebo_embaralhados(pares, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_grupo_de_um_par_so_mantem_a_atribuicao():
    _, pares = _cenario_placebo(n_duplas=1)
    emb = mod_pares.pares_placebo_embaralhados(pares, np.random.default_rng(0))
    pd.testing.assert_frame_equal(emb, pares)


def test_placebo_destroi_efeito_construido_por_alinhamento():
    """
    O teste central do placebo. O efeito real existe SO no alinhamento
    lider_i -> seguidora_i. Embaralhado, o beta mediano tem que despencar, e o
    p-valor empirico da mediana real tem que ficar pequeno.
    """
    retornos, pares = _cenario_placebo()
    j = _janela()
    cfg = dataclasses.replace(PlaceboConfig(), n_embaralhamentos=60)

    dist, resumo = nao_sincronia.rodar_placebos(
        retornos, pares, j, cfg, seed=5)

    assert len(dist) == 60
    assert resumo["beta_real_mediano"] > 0.4
    assert resumo["placebo_mediana_p50"] < 0.1
    # a identidade pode ser sorteada de vez em quando (grupo pequeno), entao o
    # p empirico nao precisa ser zero - so claramente pequeno
    assert resumo["p_empirico_mediana"] < 0.15


def test_reestimativa_sem_preco_velho_perde_amostra_mas_nao_o_efeito_real():
    """
    Efeito construido em todos os dias sobrevive ao filtro - so encolhe o n.
    Se ate um efeito de verdade morresse aqui, o filtro estaria descartando
    informacao demais.
    """
    retornos, pares = _cenario_placebo(n_duplas=1)
    j = _janela()
    completo = nao_sincronia.leadlag.estimar_rede(retornos, pares, j)

    mascara = pd.DataFrame(False, index=retornos.index,
                           columns=retornos.columns)
    dentro = retornos.index[(retornos.index >= j.treino_inicio)
                            & (retornos.index < j.treino_fim)]
    mascara.loc[dentro[100:150], "SG00"] = True

    rede = nao_sincronia.reestimar_sem_preco_velho(
        retornos, mascara, pares, j)

    assert rede.iloc[0]["reestimativa"] == "sem_preco_velho"
    assert rede.iloc[0]["n"] <= completo.iloc[0]["n"] - 50
    assert abs(rede.iloc[0]["beta"] - 0.5) < 0.05
