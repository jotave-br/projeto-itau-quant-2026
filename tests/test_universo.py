"""
Testes de selecao de universo e formacao de pares.
"""

import pandas as pd

from src import backtest, pares, setores, universo
from src.config import LiquidezConfig, WalkForwardConfig


# construtores dos cenarios
def _papel(ticker, datas, voltot, emissor=None, tipo="ON", codisi=None):
    if codisi is None:
        codisi = f"BR{(emissor or ticker[:4]).ljust(4, 'X')[:4]}ACNOR0"
    return pd.DataFrame({
        "DATA": datas, "CODNEG": ticker, "VOLTOT": float(voltot),
        "TOTNEG": 1000, "PREULT": 10.0, "CODISI": codisi,
        "TIPO_PAPEL": tipo, "NOMRES": ticker})


def _tabela_setores(classificacoes):
    """
    Tabela curada minima e valida para os testes.

    `classificacoes`: dicts com ticker, emissor, setor, subsetor e,
    opcionalmente, inicio/fim de validade. Sem validade explicita a linha
    declara estabilidade documentada, que e o que a validacao exige de uma
    linha confirmada.
    """
    linhas = []
    for c in classificacoes:
        linha = {col: "" for col in setores.COLUNAS}
        linha.update({
            "ticker_observado": c["ticker"],
            "emissor_id": c["emissor"],
            "setor": c["setor"],
            "subsetor": c["subsetor"],
            "fonte": "curadoria",
            "data_fonte": "2026-08-01",
            "evidencia": "cenario de teste",
            "estabilidade_documentada": "sim" if "inicio" not in c else "",
            "validade_inicio": c.get("inicio", ""),
            "validade_fim": c.get("fim", ""),
            "confianca": "alta",
            "status_revisao": "confirmado",
        })
        linhas.append(linha)
    tabela = pd.DataFrame(linhas, columns=list(setores.COLUNAS))
    assert setores.validar(tabela) == []
    return tabela


def _janela_e_formacao(datas):
    j = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                               pd.Timestamp("2018-12-31").date(),
                               WalkForwardConfig())[0]
    return j, datas[datas < j.treino_fim].max()


def test_um_ticker_por_emissor():
    """
    PETR3 e PETR4 sao a mesma empresa. Se ambas entram, o par lider-seguidora
    entre elas mostraria um beta enorme que e puro artefato de preco velho -
    exatamente o que o projeto existe para detectar, e nao para celebrar.
    """
    datas = pd.bdate_range("2015-01-01", periods=900)
    j, _ = _janela_e_formacao(datas)
    painel = pd.concat([
        _papel("PETR4", datas, 2e9, emissor="PETR"),
        _papel("PETR3", datas, 1e9, emissor="PETR"),
        _papel("VALE3", datas, 5e8),
        _papel("RUIM3", datas, 9e9, codisi=""),
    ], ignore_index=True)

    sel = universo.selecionar_universo(painel, j, LiquidezConfig())

    # o emissor vem do parser validado, nao de fatia cega do ISIN
    assert sel.loc["PETR4", "emissor_id"] == "PETR"
    assert sel.loc["PETR4", "motivo_exclusao"] == ""
    assert sel.loc["PETR4", "posicao_final"] == 1
    # a classe menos liquida do emissor sai, com motivo contabilizado
    assert sel.loc["PETR3", "motivo_exclusao"] == universo.MOTIVO_CLASSE_MENOS_LIQUIDA
    assert pd.isna(sel.loc["PETR3", "posicao_final"])
    # ISIN invalido nao vira emissor inventado: fica fora, contado
    assert sel.loc["RUIM3", "motivo_exclusao"] == universo.MOTIVO_ISIN_INVALIDO
    # a posicao final e recontada entre os selecionados, sem buracos
    assert sel.loc["VALE3", "posicao_final"] == 2


def test_ticker_extinto_no_treino_nao_entra_e_a_classe_viva_herda():
    """
    O caso VALE5 em 2018: convertida em VALE3 semanas antes do fim do treino,
    a mediana de 24 meses ainda a rankeava em 2o lugar - numa janela em que
    ela ja nao existia. Sem negocio no pregao de formacao o papel nao e
    operavel; e a exclusao vem antes da deduplicacao, senao a classe morta
    eliminaria a classe viva do mesmo emissor.
    """
    datas = pd.bdate_range("2015-01-01", periods=900)
    j, _ = _janela_e_formacao(datas)
    pregoes_treino = datas[datas < j.treino_fim]
    # morre 10 pregoes antes do fim do treino: cobertura ainda passa de 95%
    vivas_ate = pregoes_treino[:-10]

    painel = pd.concat([
        _papel("VALE5", vivas_ate, 3e9, emissor="VALE"),
        _papel("VALE3", datas, 1e9, emissor="VALE"),
        _papel("PETR4", datas, 2e9),
    ], ignore_index=True)

    sel = universo.selecionar_universo(painel, j, LiquidezConfig())

    assert sel.loc["VALE5", "motivo_exclusao"] == \
        universo.MOTIVO_SEM_NEGOCIACAO_NA_FORMACAO
    assert pd.isna(sel.loc["VALE5", "posicao_final"])
    # a classe viva representa o emissor, apesar de menos liquida
    assert sel.loc["VALE3", "motivo_exclusao"] == ""
    assert sel.loc["PETR4", "posicao_final"] == 1
    assert sel.loc["VALE3", "posicao_final"] == 2


def test_units_nao_pareiam_com_acao_do_mesmo_emissor():
    """Mesma logica do teste anterior, para units (SANB11, TAEE11, ALUP11)."""
    datas = pd.bdate_range("2015-01-01", periods=900)
    j, formacao = _janela_e_formacao(datas)
    painel = pd.concat([
        _papel("SANB11", datas, 3e9, emissor="SANB", tipo="UNT"),
        _papel("SANB3", datas, 1e9, emissor="SANB"),
        _papel("ITUB4", datas, 2e9),
    ], ignore_index=True)
    tabela = _tabela_setores([
        {"ticker": t, "emissor": e, "setor": "Financeiro", "subsetor": "Bancos"}
        for t, e in (("SANB11", "SANB"), ("SANB3", "SANB"), ("ITUB4", "ITUB"))])

    cfg = LiquidezConfig()
    sel = universo.selecionar_universo(painel, j, cfg)
    sel = pares.com_setor_vigente(sel, tabela, formacao)
    p = pares.gerar_pares(sel, cfg.faixas)

    assert sel.loc["SANB3", "motivo_exclusao"] == universo.MOTIVO_CLASSE_MENOS_LIQUIDA
    assert "SANB3" not in set(p["lider"]) | set(p["seguidora"])
    assert (p["emissor_lider"] != p["emissor_seguidora"]).all()
    # o unico par possivel e entre emissores distintos, com a unit na frente
    assert p[["lider", "seguidora"]].values.tolist() == [["SANB11", "ITUB4"]]


def test_pares_sao_do_mesmo_setor():
    """Restricao economica e estatistica: reduz hipoteses e faz sentido."""
    datas = pd.bdate_range("2015-01-01", periods=900)
    j, formacao = _janela_e_formacao(datas)
    painel = pd.concat([
        _papel("PETR4", datas, 4e9),
        _papel("ITUB4", datas, 3e9),
        _papel("BBDC4", datas, 2e9),
        _papel("PRIO3", datas, 1e9),
        _papel("QUIM3", datas, 5e8),
    ], ignore_index=True)
    tabela = _tabela_setores([
        {"ticker": "PETR4", "emissor": "PETR",
         "setor": "Petroleo", "subsetor": "Exploracao"},
        {"ticker": "PRIO3", "emissor": "PRIO",
         "setor": "Petroleo", "subsetor": "Exploracao"},
        # mesmo setor, subsetor diferente: nao pareia - a chave e o par
        {"ticker": "QUIM3", "emissor": "QUIM",
         "setor": "Petroleo", "subsetor": "Quimicos"},
        {"ticker": "ITUB4", "emissor": "ITUB",
         "setor": "Financeiro", "subsetor": "Bancos"},
        {"ticker": "BBDC4", "emissor": "BBDC",
         "setor": "Financeiro", "subsetor": "Bancos"},
    ])

    cfg = LiquidezConfig()
    sel = universo.selecionar_universo(painel, j, cfg)
    p = pares.gerar_pares(pares.com_setor_vigente(sel, tabela, formacao),
                          cfg.faixas)

    assert set(map(tuple, p[["lider", "seguidora"]].values)) == {
        ("PETR4", "PRIO3"), ("ITUB4", "BBDC4")}


def test_lider_e_mais_liquida_que_seguidora():
    """A direcao do par vem da hipotese, nao dos dados."""
    datas = pd.bdate_range("2015-01-01", periods=900)
    j, formacao = _janela_e_formacao(datas)
    painel = pd.concat([
        _papel("CCCC3", datas, 3e9),
        # empate exato de liquidez: o desempate e deterministico, alfabetico
        _papel("BBBB3", datas, 1e9),
        _papel("AAAA3", datas, 1e9),
    ], ignore_index=True)
    tabela = _tabela_setores([
        {"ticker": t, "emissor": t[:4], "setor": "Setor", "subsetor": "Sub"}
        for t in ("AAAA3", "BBBB3", "CCCC3")])

    cfg = LiquidezConfig()
    sel = universo.selecionar_universo(painel, j, cfg)
    p = pares.gerar_pares(pares.com_setor_vigente(sel, tabela, formacao),
                          cfg.faixas)

    assert (p["liquidez_lider"] >= p["liquidez_seguidora"]).all()
    assert (p["posicao_lider"] < p["posicao_seguidora"]).all()
    empate = p[(p["lider"] == "AAAA3") | (p["seguidora"] == "AAAA3")]
    assert ("AAAA3", "BBBB3") in set(map(tuple, empate[["lider", "seguidora"]].values))


def test_cobertura_minima_exclui_ticker_com_buraco():
    """
    Papel com serie incompleta na janela nao entra no universo daquela janela.

    Note que a elegibilidade e por janela. Um papel pode ser elegivel em 2019 e
    nao em 2017 - exigir cobertura sobre a amostra inteira reprovaria qualquer
    empresa que abriu ou fechou capital no meio, o que nao e defeito de dado.
    """
    j = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                               pd.Timestamp("2018-12-31").date(),
                               WalkForwardConfig())[0]
    datas = pd.bdate_range("2015-01-01", periods=900)
    metade = datas[: len(datas) // 2]

    def _linhas(ticker, quando):
        return pd.DataFrame({
            "DATA": quando, "CODNEG": ticker, "VOLTOT": 1e9, "TOTNEG": 1000,
            "PREULT": 10.0, "CODISI": f"BR{ticker}ACNOR0",
            "TIPO_PAPEL": "ON", "NOMRES": ticker})

    painel = pd.concat([_linhas("INTEIRO3", datas), _linhas("BURACO3", metade)],
                       ignore_index=True)
    r = universo.ranking_liquidez_pit(painel, j, LiquidezConfig())

    assert bool(r.loc["INTEIRO3", "elegivel"]) is True
    assert bool(r.loc["BURACO3", "elegivel"]) is False
    assert pd.isna(r.loc["BURACO3", "posicao"])


def test_pares_nao_sao_escolhidos_por_desempenho():
    """
    Selecionar par olhando retorno e o lookahead classico. Pares saem de
    setor, emissor e liquidez - nunca de resultado. A prova: mudar todos os
    precos do painel nao muda um unico par.
    """
    datas = pd.bdate_range("2015-01-01", periods=900)
    j, formacao = _janela_e_formacao(datas)
    tickers = [("AAAA3", 4e9), ("BBBB3", 3e9), ("CCCC3", 2e9), ("DDDD3", 1e9)]
    tabela = _tabela_setores([
        {"ticker": t, "emissor": t[:4], "setor": "Setor", "subsetor": "Sub"}
        for t, _ in tickers])
    cfg = LiquidezConfig()

    def _pares_com_precos(precos_por_ticker):
        painel = pd.concat([
            _papel(t, datas, v).assign(PREULT=precos_por_ticker[t])
            for t, v in tickers], ignore_index=True)
        sel = universo.selecionar_universo(painel, j, cfg)
        return pares.gerar_pares(pares.com_setor_vigente(sel, tabela, formacao),
                                 cfg.faixas)

    constantes = _pares_com_precos({t: 10.0 for t, _ in tickers})
    # retornos radicalmente diferentes por papel: tendencias opostas
    n = len(datas)
    tendencias = _pares_com_precos({
        "AAAA3": [10 * 1.001 ** i for i in range(n)],
        "BBBB3": [10 * 0.999 ** i for i in range(n)],
        "CCCC3": [10 + (i % 7) for i in range(n)],
        "DDDD3": [200 - 0.1 * i for i in range(n)],
    })
    pd.testing.assert_frame_equal(constantes, tendencias)


def test_sem_classificacao_confirmada_na_formacao_fica_fora():
    """
    O caso ITSA4: linha existe, mas nenhuma classificacao confirmada vigente
    na data de formacao. O ticker sai dos pares daquela janela, sem herdar
    setor e sem cair para a linha mais proxima.
    """
    datas = pd.bdate_range("2015-01-01", periods=900)
    j, formacao = _janela_e_formacao(datas)
    painel = pd.concat([
        _papel("AAAA3", datas, 3e9),
        _papel("BBBB3", datas, 2e9),
        _papel("LACU3", datas, 1e9),
    ], ignore_index=True)
    tabela = _tabela_setores([
        {"ticker": "AAAA3", "emissor": "AAAA", "setor": "S", "subsetor": "X"},
        {"ticker": "BBBB3", "emissor": "BBBB", "setor": "S", "subsetor": "X"},
        # confirmada, porem valida so depois da formacao desta janela
        {"ticker": "LACU3", "emissor": "LACU", "setor": "S", "subsetor": "X",
         "inicio": "2020-01-01"},
    ])

    cfg = LiquidezConfig()
    sel = universo.selecionar_universo(painel, j, cfg)
    sel = pares.com_setor_vigente(sel, tabela, formacao)
    p = pares.gerar_pares(sel, cfg.faixas)

    assert sel.loc["LACU3", "motivo_exclusao"] == pares.MOTIVO_SEM_SETOR
    assert "LACU3" not in set(p["lider"]) | set(p["seguidora"])
    assert p[["lider", "seguidora"]].values.tolist() == [["AAAA3", "BBBB3"]]


# alcance PIT
#
# A distincao que estes testes protegem: a melhor posicao que o papel ja teve e
# descritiva; a posicao na janela que contem a data e o que decide se o evento
# importa. Confundir as duas promove evento antigo por liquidez posterior.
def _painel_liquidez(niveis_por_periodo, pregoes=1000, inicio="2015-01-01"):
    """
    `niveis_por_periodo` mapeia ticker -> (voltot antes do corte, depois).
    O corte fica na metade da amostra.
    """
    datas = pd.bdate_range(inicio, periods=pregoes)
    corte = datas[len(datas) // 2]
    partes = []
    for tk, (antes, depois) in niveis_por_periodo.items():
        partes.append(pd.DataFrame({
            "DATA": datas, "CODNEG": tk,
            "VOLTOT": [antes if d < corte else depois for d in datas],
            "TOTNEG": 1000, "PREULT": 10.0,
            "CODISI": f"BR{tk[:4]}ACNOR0", "TIPO_PAPEL": "ON", "NOMRES": tk,
        }))
    return pd.concat(partes, ignore_index=True), corte


def test_faixa_na_data_difere_da_melhor_faixa_global():
    """
    O caso GOLL4. O papel ja esteve bem colocado em alguma janela, mas na
    janela que contem o evento estava pior. A fila tem que usar a segunda.
    """
    cfg = LiquidezConfig()
    niveis = {f"P{i:02d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(30)}
    # O alvo comeca mal colocado e melhora muito depois do corte.
    niveis["ALVO3"] = (1.0, 1e12)
    painel, corte = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["ALVO3"], "data": [pd.Timestamp("2015-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]

    assert r["melhor_posicao_qualquer_janela"] == 1          # depois ficou o mais liquido
    assert r["faixa_melhor_qualquer_janela"] == "top20"
    assert r["melhor_posicao_treino_na_data"] > 20           # na data, nao estava la
    assert r["status_treino_na_data"] != "top20"


def test_evento_que_alcanca_apenas_top100_na_data():
    cfg = LiquidezConfig()
    niveis = {f"P{i:03d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(120)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P080"], "data": [pd.Timestamp("2015-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["status_treino_na_data"] == "top100"
    assert bool(r["pode_afetar_estimacao"]) is True


def test_evento_fora_de_qualquer_universo_nao_afeta_nada():
    cfg = LiquidezConfig()
    niveis = {f"P{i:03d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(150)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P140"], "data": [pd.Timestamp("2015-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["status_treino_na_data"] == universo.STATUS_ELEGIVEL_FORA
    assert bool(r["pode_afetar_estimacao"]) is False
    assert bool(r["pode_afetar_sinal"]) is False


def test_data_fora_de_qualquer_janela_nao_afeta_estimacao():
    """Evento anterior ao primeiro treino nao contamina rede nenhuma."""
    cfg = LiquidezConfig()
    niveis = {f"P{i:02d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(30)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P00"], "data": [pd.Timestamp("2010-01-04")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["janelas_treino_contendo_a_data"] == 0
    assert bool(r["pode_afetar_estimacao"]) is False
    assert r["status_treino_na_data"] == universo.STATUS_SEM_JANELA
    # a informacao descritiva continua disponivel
    assert r["faixa_melhor_qualquer_janela"] == "top20"


# estados exclusivos x alcance cumulativo
#
# Sao leituras diferentes do mesmo dado e nao podem ser confundidas: um caso na
# 58a posicao tem estado exclusivo top60, e no cumulativo aparece em top60 e em
# top100.
def test_data_sem_nenhuma_janela():
    cfg = LiquidezConfig()
    niveis = {f"P{i:02d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(30)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P00"], "data": [pd.Timestamp("2010-01-04")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["status_treino_na_data"] == universo.STATUS_SEM_JANELA
    assert r["status_teste_na_data"] == universo.STATUS_SEM_JANELA
    assert r["janelas_treino_contendo_a_data"] == 0


def test_ticker_ausente_da_janela_nao_e_o_mesmo_que_nao_elegivel():
    """
    A distincao que a auditoria de dados faz entre "nao listado" e "sem
    negociacao" vale aqui: nao existir na janela nao e existir e nao qualificar.
    """
    cfg = LiquidezConfig()
    datas = pd.bdate_range("2015-01-01", periods=1000)
    partes = [pd.DataFrame({
        "DATA": datas, "CODNEG": f"P{i:02d}", "VOLTOT": 1e9 / (i + 1),
        "TOTNEG": 1000, "PREULT": 10.0, "CODISI": f"BRP{i:02d}ACNOR0",
        "TIPO_PAPEL": "ON", "NOMRES": f"P{i}"}) for i in range(30)]
    # SOZINHO3 so existe no fim da amostra: ausente das janelas iniciais.
    partes.append(pd.DataFrame({
        "DATA": datas[-50:], "CODNEG": "SOZINHO3", "VOLTOT": 1e9,
        "TOTNEG": 1000, "PREULT": 10.0, "CODISI": "BRSOZIACNOR0",
        "TIPO_PAPEL": "ON", "NOMRES": "SOZINHO"}))
    painel = pd.concat(partes, ignore_index=True)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["SOZINHO3"], "data": [pd.Timestamp("2015-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["status_treino_na_data"] == universo.STATUS_AUSENTE


def test_ticker_presente_mas_nao_elegivel_na_janela():
    """Existe na janela e falha nos criterios - estado proprio, nao ausencia."""
    cfg = LiquidezConfig()
    datas = pd.bdate_range("2015-01-01", periods=1000)
    partes = [pd.DataFrame({
        "DATA": datas, "CODNEG": f"P{i:02d}", "VOLTOT": 1e9 / (i + 1),
        "TOTNEG": 1000, "PREULT": 10.0, "CODISI": f"BRP{i:02d}ACNOR0",
        "TIPO_PAPEL": "ON", "NOMRES": f"P{i}"}) for i in range(30)]
    # FURADO3 negocia so um terco dos pregoes: reprova por cobertura.
    partes.append(pd.DataFrame({
        "DATA": datas[::3], "CODNEG": "FURADO3", "VOLTOT": 1e9,
        "TOTNEG": 1000, "PREULT": 10.0, "CODISI": "BRFURAACNOR0",
        "TIPO_PAPEL": "ON", "NOMRES": "FURADO"}))
    painel = pd.concat(partes, ignore_index=True)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["FURADO3"], "data": [pd.Timestamp("2015-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["status_treino_na_data"] == universo.STATUS_NAO_ELEGIVEL


def test_ticker_elegivel_abaixo_do_top100():
    cfg = LiquidezConfig()
    niveis = {f"P{i:03d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(150)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P130"], "data": [pd.Timestamp("2015-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["status_treino_na_data"] == universo.STATUS_ELEGIVEL_FORA
    assert bool(r["pode_afetar_estimacao"]) is False


def test_classificacao_exclusiva_top60():
    """Posicao 58 e top60, nao top100: a menor faixa que ela alcanca."""
    cfg = LiquidezConfig()
    niveis = {f"P{i:03d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(120)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P050"], "data": [pd.Timestamp("2015-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["melhor_posicao_treino_na_data"] == 51
    assert r["status_treino_na_data"] == "top60"


def test_alcance_cumulativo_conta_faixas_maiores():
    """
    top100 contem top60. Confundir cumulativo com exclusivo faz "dois casos
    top100" parecer dois casos distintos quando um deles ja e top60.
    """
    cfg = LiquidezConfig()
    niveis = {f"P{i:03d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(120)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P050", "P080"],
                       "data": [pd.Timestamp("2015-06-01")] * 2})
    alc = universo.alcance_pit(oc, painel, janelas, cfg)

    exclusivo = alc["status_treino_na_data"].value_counts().to_dict()
    assert exclusivo == {"top60": 1, "top100": 1}

    cum = universo.alcance_cumulativo_por_faixa(
        alc, "melhor_posicao_treino_na_data", cfg.faixas)
    assert cum["top20"] == 0
    assert cum["top40"] == 0
    assert cum["top60"] == 1
    assert cum["top100"] == 2          # o top60 tambem conta aqui


def test_multiplas_janelas_de_treino_usam_a_melhor_posicao():
    """
    Os treinos se sobrepoem, entao uma data cai em varias janelas. Se o papel
    entrou em faixa em alguma, e isso que vale.
    """
    cfg = LiquidezConfig()
    niveis = {f"P{i:03d}": (1e9 / (i + 1), 1e9 / (i + 1)) for i in range(120)}
    painel, _ = _painel_liquidez(niveis)
    janelas = backtest.gerar_janelas(pd.Timestamp("2015-01-01").date(),
                                     pd.Timestamp("2018-12-31").date(),
                                     WalkForwardConfig())

    oc = pd.DataFrame({"ticker": ["P010"], "data": [pd.Timestamp("2016-06-01")]})
    r = universo.alcance_pit(oc, painel, janelas, cfg).iloc[0]
    assert r["janelas_treino_contendo_a_data"] > 1
    assert r["status_treino_na_data"] == "top20"
