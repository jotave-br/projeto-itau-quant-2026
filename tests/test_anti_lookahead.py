"""
Testes anti-lookahead: a prova de que o backtest nao espiou o futuro.

Backtest com lookahead produz numero espetacular e completamente falso, e a
unica forma de mostrar que o nosso nao tem isso e ter a verificacao rodando.

A tecnica e sempre a mesma: calcular, adulterar os dados posteriores ao fim do
treino, recalcular e exigir resultado identico. Um teste que so conferisse o
recorte passaria mesmo com o codigo lendo o futuro por outro caminho.

Teste de garantia cujo modulo ainda nao existe fica em skip, e a lista serve de
checklist do que falta provar.
"""

import numpy as np
import pandas as pd
import pytest

from src import backtest, leadlag, universo
from src.config import LeadLagConfig, LiquidezConfig, WalkForwardConfig

FASE_2 = "pendente: implementacao na fase 2"
FASE_3 = "pendente: implementacao na fase 3"


def _painel(inicio="2015-01-01", pregoes=900, tickers=6):
    """
    Painel COTAHIST sintetico com liquidez decrescente por ticker.

    A ordem e conhecida por construcao, entao qualquer mudanca no ranking
    depois de adulterar o futuro e sinal de vazamento.
    """
    datas = pd.bdate_range(inicio, periods=pregoes)
    partes = []
    for i in range(tickers):
        partes.append(pd.DataFrame({
            "DATA": datas,
            "CODNEG": f"T{i:02d}",
            "VOLTOT": 1e9 / (i + 1),
            "TOTNEG": 1000,
            "PREULT": 10.0,
            "CODISI": f"BRT{i:02d}ACNOR0",
            "TIPO_PAPEL": "ON",
            "NOMRES": f"EMPRESA {i}",
        }))
    return pd.concat(partes, ignore_index=True)


def _janelas(fim="2018-12-31"):
    return backtest.gerar_janelas(
        pd.Timestamp("2015-01-01").date(), pd.Timestamp(fim).date(),
        WalkForwardConfig())


def test_janela_treino_nao_contem_datas_do_teste():
    """Nenhuma janela de treino pode conter data posterior ao seu proprio fim."""
    janelas = _janelas("2020-12-31")
    assert janelas
    for j in janelas:
        assert j.treino_inicio < j.treino_fim
        assert j.treino_fim <= j.teste_inicio < j.teste_fim


def test_ranking_liquidez_usa_somente_dados_de_treino():
    """
    Pega o lookahead disfarcado de "definicao de universo".

    Inverte a liquidez depois do fim do treino, fazendo o papel menos liquido
    virar o mais liquido no futuro. Se o ranking olhasse a amostra inteira, a
    ordem mudaria.
    """
    cfg = LiquidezConfig()
    j = _janelas()[0]
    painel = _painel()
    antes = universo.ranking_liquidez_pit(painel, j, cfg)

    adulterado = painel.copy()
    futuro = adulterado["DATA"] >= j.treino_fim
    adulterado.loc[futuro, "VOLTOT"] = adulterado.loc[futuro, "CODNEG"].map(
        lambda t: 1e12 if t == "T05" else 1.0)

    depois = universo.ranking_liquidez_pit(adulterado, j, cfg)
    assert list(antes.index) == list(depois.index)
    pd.testing.assert_series_equal(antes["posicao"], depois["posicao"])


def test_universo_congelado_durante_a_janela_de_teste():
    """
    O universo vale para a janela de teste inteira. Adulterar dados dentro do
    teste nao pode mexer em quem foi selecionado - reagir a liquidez que ainda
    nao aconteceu e lookahead com outro nome.
    """
    cfg = LiquidezConfig()
    j = _janelas()[0]
    painel = _painel()
    r = universo.ranking_liquidez_pit(painel, j, cfg)
    selecionado = list(r[r["elegivel"]].index)

    adulterado = painel.copy()
    no_teste = ((adulterado["DATA"] >= j.teste_inicio)
                & (adulterado["DATA"] < j.teste_fim))
    adulterado.loc[no_teste, "VOLTOT"] = 0.0

    depois = universo.ranking_liquidez_pit(adulterado, j, cfg)
    assert list(depois[depois["elegivel"]].index) == selecionado


def test_estimacao_nao_recebe_dataframe_com_datas_futuras():
    """
    Verificacao estrutural: o recorte acontece dentro da funcao, a partir da
    janela, e nao depende de quem chama ter recortado antes.
    """
    cfg = LiquidezConfig()
    j = _janelas()[0]
    completo = universo.ranking_liquidez_pit(_painel(), j, cfg)

    painel = _painel()
    so_treino = painel[(painel["DATA"] >= j.treino_inicio)
                       & (painel["DATA"] < j.treino_fim)]
    recortado = universo.ranking_liquidez_pit(so_treino, j, cfg)
    pd.testing.assert_frame_equal(completo, recortado)


def _inverter_liquidez_no_futuro(painel, corte, tickers=6):
    """
    Inverte a ordem de liquidez SO depois do corte: o ultimo vira o primeiro.

    Multiplicar todo mundo pelo mesmo fator nao serve de teste - preserva o
    ranking, e uma implementacao com vazamento passaria. A ordem tem que virar
    do avesso para a diferenca aparecer.
    """
    adulterado = painel.copy()
    futuro = adulterado["DATA"] >= corte
    novo = {f"T{i:02d}": 1e9 / (tickers - i) for i in range(tickers)}
    adulterado.loc[futuro, "VOLTOT"] = adulterado.loc[futuro, "CODNEG"].map(novo)
    return adulterado


def test_melhor_posicao_por_ticker_nao_vaza_entre_janelas():
    """
    Cada janela e rankeada com os seus proprios dados. Saber que um papel ficou
    liquido em 2024 nao pode promove-lo no universo de 2017.
    """
    cfg = LiquidezConfig()
    janelas = _janelas()[:1]
    painel = _painel(pregoes=1000)
    antes = universo.melhor_posicao_por_ticker(painel, janelas, cfg)

    adulterado = _inverter_liquidez_no_futuro(painel, janelas[0].treino_fim)
    depois = universo.melhor_posicao_por_ticker(adulterado, janelas, cfg)

    # A adulteracao tem que ser capaz de mudar o resultado, senao o teste nao
    # discrimina nada: rankeando o periodo adulterado a ordem realmente vira.
    so_futuro = adulterado[adulterado["DATA"] >= janelas[0].treino_fim]
    invertido = universo.ranking_liquidez_pit(
        so_futuro, backtest.Janela(0, janelas[0].treino_fim,
                                   so_futuro["DATA"].max(),
                                   so_futuro["DATA"].max(),
                                   so_futuro["DATA"].max()), cfg)
    assert list(invertido.index)[0] == "T05"

    pd.testing.assert_series_equal(antes["melhor_posicao"], depois["melhor_posicao"])
    assert antes["melhor_posicao"].idxmin() == "T00"


def test_alcance_pit_nao_usa_liquidez_futura():
    """
    A triagem dos retornos extremos ordena a fila por posicao NA DATA. Se essa
    posicao viesse da amostra inteira, um papel que ficou liquido depois
    subiria na fila de um evento antigo.
    """
    cfg = LiquidezConfig()
    janelas = _janelas()[:1]
    painel = _painel(pregoes=1000)
    ocorrencias = pd.DataFrame({"ticker": ["T05"],
                                "data": [pd.Timestamp("2015-06-01")]})
    antes = universo.alcance_pit(ocorrencias, painel, janelas, cfg)

    adulterado = _inverter_liquidez_no_futuro(painel, janelas[0].treino_fim)
    depois = universo.alcance_pit(ocorrencias, adulterado, janelas, cfg)

    for coluna in ("melhor_posicao_treino_na_data", "status_treino_na_data",
                   "melhor_posicao_teste_na_data", "status_teste_na_data",
                   "pode_afetar_estimacao", "pode_afetar_sinal"):
        pd.testing.assert_series_equal(antes[coluna], depois[coluna])


def _retornos_e_pares_sinteticos(pregoes=900, seed=7):
    """Painel de retornos e pares para os testes de estimacao."""
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2015-01-01", periods=pregoes)
    lider = pd.Series(rng.normal(0, 0.02, pregoes), index=datas)
    segue = 0.4 * lider.shift(1) + pd.Series(rng.normal(0, 0.01, pregoes),
                                             index=datas)
    retornos = pd.DataFrame({"LIDER3": lider, "SEGUE3": segue})
    pares = pd.DataFrame([{
        "janela": "2017-01", "setor": "S", "subsetor": "X",
        "lider": "LIDER3", "seguidora": "SEGUE3", "faixa_minima": 20,
        "posicao_lider": 1, "posicao_seguidora": 2,
    }])
    return retornos, pares


def test_rede_estimada_ignora_dados_pos_treino():
    """
    A rede de uma janela so pode depender do treino dela. A adulteracao troca
    o futuro por uma relacao perfeita (seguidora identica a lider defasada):
    se um unico ponto do futuro entrasse na regressao, beta, t e n mudariam.
    """
    retornos, pares = _retornos_e_pares_sinteticos()
    j = _janelas()[0]
    antes = leadlag.estimar_rede(retornos, pares, j, LeadLagConfig())

    adulterado = retornos.copy()
    futuro = adulterado.index >= j.treino_fim
    adulterado.loc[futuro, "SEGUE3"] = adulterado["LIDER3"].shift(1)[futuro]

    depois = leadlag.estimar_rede(adulterado, pares, j, LeadLagConfig())
    pd.testing.assert_frame_equal(antes, depois)


def test_fdr_usa_somente_testes_da_propria_janela():
    """
    Benjamini-Hochberg aplicado por janela. Aplicar sobre a amostra inteira e
    depois operar seria selecionar pares com informacao do futuro.

    Duas garantias: a funcao recusa tabela com mais de uma janela, e o
    resultado de uma janela nao muda quando os testes de outra janela mudam -
    porque eles nem podem entrar juntos.
    """
    from src import multiplos_testes as mt

    def _rede(pvalores, janela):
        return pd.DataFrame({
            "janela": janela,
            "lider": [f"L{i}" for i in range(len(pvalores))],
            "seguidora": [f"S{i}" for i in range(len(pvalores))],
            "p_valor": pvalores, "beta": 0.1,
            "estat_t": list(range(len(pvalores), 0, -1)),
        })

    janela_a = _rede([0.001, 0.02, 0.4], "2017-01")
    com_futuro_bom = _rede([0.0001] * 30, "2017-04")
    com_futuro_ruim = _rede([0.99] * 30, "2017-04")

    with pytest.raises(ValueError, match="janelas"):
        mt.aplicar_fdr_janela(pd.concat([janela_a, com_futuro_bom],
                                        ignore_index=True))

    r1, _ = mt.aplicar_fdr_janela(janela_a)
    r2, _ = mt.aplicar_fdr_janela(janela_a.copy())
    for outra in (com_futuro_bom, com_futuro_ruim):
        mt.aplicar_fdr_janela(outra)
    pd.testing.assert_frame_equal(r1, r2)


def test_sinal_em_t_so_opera_em_t_mais_1():
    """
    Defasagem de execucao. Operar no proprio fechamento de t seria negociar
    com um preco que ainda nao existia quando o sinal foi gerado.
    """
    from src import estrategia

    cal = pd.bdate_range("2020-01-01", periods=60)
    j = backtest.Janela(0, cal[0], cal[20], cal[20], cal[40])
    t = cal[25]

    ret_sinal = pd.DataFrame(0.0, index=cal, columns=["LD00", "SG00"])
    ret_sinal.loc[t, "LD00"] = 0.01
    vol = pd.DataFrame(0.20, index=cal, columns=["SG00"])
    pares = pd.DataFrame({"lider": ["LD00"], "seguidora": ["SG00"]})

    pos, _ = estrategia.construir_posicoes_janela(
        pares, ret_sinal, vol, j, cal)

    assert t not in pos.index or pos.loc[t, "SG00"] == 0
    assert pos.loc[cal[26], "SG00"] != 0

    # e o P&L so comeca no retorno de t+2: o retorno da seguidora em t+1 e
    # gigante de proposito - se um centavo dele entrasse, a serie mudaria
    ret_pnl = pd.DataFrame(0.0, index=cal, columns=["LD00", "SG00"])
    ret_pnl.loc[cal[26], "SG00"] = 10.0            # t+1: nao pode entrar
    ret_pnl.loc[cal[27], "SG00"] = 0.01            # t+2: primeiro P&L real
    pnl, _ = estrategia.pnl_bruto(pos, ret_pnl)

    assert pnl.get(cal[26], 0.0) == pytest.approx(0.0)
    assert pnl.loc[cal[27]] == pytest.approx(pos.loc[cal[26], "SG00"] * 0.01)


def test_walk_forward_deterministico():
    """
    Duas execucoes identicas produzem exatamente os mesmos numeros, do beta
    ao P&L liquido. O walk-forward nao tem fonte de aleatoriedade - se este
    teste falhar, algo nao deterministico entrou no caminho do dinheiro.
    """
    from src import retornos as mod_ret, setores
    from src.config import CONFIG_PADRAO

    rng = np.random.default_rng(9)
    datas = pd.bdate_range("2015-01-01", periods=700)
    n = 8
    tickers = [f"T{i:02d}3" for i in range(n)]

    precos = {}
    base = rng.normal(0, 0.02, (len(datas), n // 2))
    for i in range(n // 2):
        precos[tickers[i]] = 50 * np.exp(np.cumsum(base[:, i]))
    for i in range(n // 2, n):
        eco = 0.4 * np.roll(base[:, i - n // 2], 1)
        eco[0] = 0.0
        ruido = rng.normal(0, 0.01, len(datas))
        precos[tickers[i]] = 30 * np.exp(np.cumsum(eco + ruido))

    cot = pd.concat([pd.DataFrame({
        "DATA": datas, "CODNEG": tk, "PREULT": precos[tk],
        "ESPECI": "ON      NM", "VOLTOT": 1e9 / (i + 1), "TOTNEG": 1000,
        "CODISI": f"BRT{i:02d}XACNOR0"[:12], "TIPO_PAPEL": "ON", "NOMRES": tk,
    }) for i, tk in enumerate(tickers)], ignore_index=True)

    tabela = pd.DataFrame([{c: "" for c in setores.COLUNAS} | {
        "ticker_observado": tk, "emissor_id": f"T{i:02d}X",
        "setor": "S", "subsetor": "X", "fonte": "curadoria",
        "data_fonte": "2026-08-01", "evidencia": "teste",
        "estabilidade_documentada": "sim", "confianca": "alta",
        "status_revisao": "confirmado",
    } for i, tk in enumerate(tickers)], columns=list(setores.COLUNAS))

    cal = pd.DatetimeIndex(sorted(cot["DATA"].unique()))
    vol = mod_ret.painel_volume_financeiro(cot).reindex(cal)
    ret_est = mod_ret.retornos_preco_bruto_cotahist(cot, cal, vol)
    ret_pnl = mod_ret.retornos_pnl_marcacao(cot, cal)
    janelas = backtest.gerar_janelas(datas[0].date(), datas[-1].date(),
                                     WalkForwardConfig())

    def _roda():
        return backtest.rodar_walk_forward(
            cot, cal, tabela, ret_est, ret_pnl, janelas, CONFIG_PADRAO,
            combos=[("top_k", "long_short")])

    a, b = _roda(), _roda()
    pd.testing.assert_frame_equal(a["rede"], b["rede"])
    resultado_a = a["combos"]["top_k_long_short"]
    resultado_b = b["combos"]["top_k_long_short"]
    assert "vazio" not in resultado_a          # o cenario gera trades de verdade
    pd.testing.assert_frame_equal(resultado_a["pnl"], resultado_b["pnl"])
    assert resultado_a["resumo"] == resultado_b["resumo"]
