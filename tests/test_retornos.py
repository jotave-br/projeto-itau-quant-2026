"""
Testes do calculo de retornos.

O tema central e sempre o mesmo: um retorno diario tem que ser de um pregao so.
Emendar por cima de buracos produz retornos de varios dias disfarcados de
retorno diario, o que inflaria a volatilidade e criaria correlacao espuria
justamente nos papeis iliquidos - os que o projeto precisa examinar com mais
cuidado.
"""

import numpy as np
import pandas as pd
import pytest

from src import retornos


def cal(n: int = 6) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-06", periods=n)


def test_retorno_simples_basico():
    p = pd.Series([100.0, 110.0, 99.0], index=cal(3))
    r = retornos.retornos_simples(p)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(-0.10)


@pytest.mark.filterwarnings("ignore:The default fill_method:FutureWarning")
def test_buraco_no_meio_nao_vira_retorno_diario():
    """
    O ponto central do modulo. O papel negocia a 100, some por um pregao e
    volta a 121. `pct_change` do pandas devolveria 21% como se fosse retorno de
    um dia. Aqui os dois retornos que tocam o buraco sao NaN.
    """
    p = pd.Series([100.0, np.nan, 121.0, 121.0], index=cal(4))
    r = retornos.retornos_simples(p)
    assert np.isnan(r.iloc[1])          # dia ausente
    assert np.isnan(r.iloc[2])          # dia seguinte ao ausente
    assert r.iloc[3] == pytest.approx(0.0)

    # Contraste explicito com o comportamento padrao do pandas, que preenche
    # o buraco e devolve 21% como se fosse retorno de um dia. O proprio pandas
    # esta depreciando esse preenchimento automatico justamente por ser uma
    # armadilha.
    assert p.pct_change().iloc[2] == pytest.approx(0.21)


def test_retorno_log_tem_a_mesma_disciplina_de_buraco():
    p = pd.Series([100.0, np.nan, 110.0], index=cal(3))
    r = retornos.retornos_log(p)
    assert r.isna().all()


def test_retorno_log_soma_ao_longo_do_tempo():
    p = pd.Series([100.0, 110.0, 121.0], index=cal(3))
    total = retornos.retornos_log(p).sum()
    assert total == pytest.approx(np.log(1.21))


def test_alinhar_ao_calendario_nao_preenche():
    """
    Alinhamento cria a data, nunca o dado. Preencher antes das flags fabricaria
    preco travado e retorno zero que nao existiram.
    """
    df = pd.DataFrame({"Date": [cal(3)[0], cal(3)[2]], "Adj Close": [10.0, 11.0]})
    painel = retornos.painel_precos_ajustados({"XPTO3": df}, cal(3))
    assert len(painel) == 3
    assert np.isnan(painel["XPTO3"].iloc[1])


def test_mascara_exige_negociacao_nos_dois_pregoes():
    """
    Preco herdado nao e preco negociado. Um dia sem negocio tem variacao zero
    por falta de negocio, nao por decisao de mercado - e usar esse ponto e o
    caminho mais curto para o preco velho entrar na regressao.
    """
    idx = cal(4)
    r = pd.DataFrame({"A": [np.nan, 0.01, 0.0, 0.02]}, index=idx)
    vol = pd.DataFrame({"A": [1e6, 1e6, 0.0, 1e6]}, index=idx)
    m = retornos.mascara_retorno_valido(r, vol)
    assert bool(m["A"].iloc[1]) is True     # negociou nos dois dias
    assert bool(m["A"].iloc[2]) is False    # nao negociou hoje
    assert bool(m["A"].iloc[3]) is False    # nao negociou ontem


def test_liquidez_usa_mediana_e_nao_media():
    """Um leilao gigante isolado nao deve promover o papel no ranking."""
    idx = cal(5)
    vol = pd.DataFrame({"A": [1e5, 1e5, 1e9, 1e5, 1e5]}, index=idx)
    med = retornos.liquidez_mediana(vol, janela=5).iloc[-1, 0]
    assert med == pytest.approx(1e5)
    assert vol["A"].mean() > 1e8            # a media seria enganada


def test_volatilidade_anualiza_com_252():
    idx = pd.bdate_range("2020-01-06", periods=40)
    rng = np.random.default_rng(42)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, 40)}, index=idx)
    diaria = retornos.volatilidade_realizada(r, 20, anualizar=False).iloc[-1, 0]
    anual = retornos.volatilidade_realizada(r, 20, anualizar=True).iloc[-1, 0]
    assert anual == pytest.approx(diaria * np.sqrt(252))


def test_cobertura_conta_negociou_sem_preco():
    """
    Mede cobertura da fonte: dias em que a bolsa registrou negocio mas o
    yfinance nao tem preco. Nao e o tamanho do vies de sobrevivencia - ausencia
    no Yahoo tambem vem de migracao de ticker ou falha de download.
    """
    idx = cal(3)
    precos = pd.DataFrame({"A": [10.0, np.nan, 12.0]}, index=idx)
    vol = pd.DataFrame({"A": [1e6, 1e6, 1e6]}, index=idx)
    c = retornos.cobertura_yahoo_vs_cotahist(precos, vol)
    assert int(c.loc["A", "pregoes_com_negociacao"]) == 3
    assert int(c.loc["A", "pregoes_com_preco_yf"]) == 2
    assert int(c.loc["A", "negociou_sem_preco"]) == 1
    assert c.loc["A", "cobertura_yahoo_vs_cotahist"] == pytest.approx(2 / 3)
    assert c.loc["A", "gap_cobertura_yahoo"] == pytest.approx(1 / 3)


def test_situacao_cobertura_distingue_as_causas():
    """
    Ausencia no Yahoo tem causas diferentes e elas nao podem virar a mesma
    coisa: falha de download nao e empresa extinta, e migracao de ticker nao e
    dado faltante.
    """
    idx = cal(4)
    precos = pd.DataFrame({"COMPLETO": [1.0, 2.0, 3.0, 4.0],
                           "PARCIAL": [1.0, np.nan, np.nan, 4.0],
                           "AUSENTE": [np.nan] * 4,
                           "FALHOU": [np.nan] * 4,
                           "MIGROU": [np.nan] * 4,
                           "BLOQUEADO": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    vol = pd.DataFrame({c: [1e6] * 4 for c in precos.columns}, index=idx)
    status = pd.DataFrame({"ticker": ["FALHOU"], "status": ["erro_persistente"]})

    c = retornos.cobertura_yahoo_vs_cotahist(
        precos, vol, status_download=status,
        tickers_continuidade_bloqueada={"BLOQUEADO"},
        tickers_cobertos_por_terminal={"MIGROU": "NOVO3"})

    assert c.loc["COMPLETO", "situacao_cobertura"] == "cobertura_completa"
    assert c.loc["PARCIAL", "situacao_cobertura"] == "cobertura_parcial"
    assert c.loc["AUSENTE", "situacao_cobertura"] == "ausente"
    assert c.loc["FALHOU", "situacao_cobertura"] == "erro_temporario"
    assert c.loc["MIGROU", "situacao_cobertura"] == "historico_sob_codigo_posterior"
    assert c.loc["MIGROU", "ticker_terminal"] == "NOVO3"
    assert c.loc["BLOQUEADO", "situacao_cobertura"] == "continuidade_bloqueada"


def test_detecta_series_identicas_sob_codigos_diferentes():
    """
    O Yahoo migra o historico de tickers renomeados, e a mesma serie pode voltar
    sob dois codigos. Duas series identicas nao podem entrar na rede como
    instrumentos diferentes: um par lead-lag entre elas teria correlacao
    perfeita por construcao.
    """
    idx = pd.bdate_range("2020-01-06", periods=80)
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.01, 80)
    p = pd.DataFrame({"ANT3": base, "NOV3": base,
                      "OUTRO3": rng.normal(0, 0.01, 80)}, index=idx)
    d = retornos.detectar_series_duplicadas(p)
    pares = {tuple(sorted((a, b))) for a, b in zip(d.ticker_a, d.ticker_b)}
    assert ("ANT3", "NOV3") in pares
    assert not any("OUTRO3" in par for par in pares)
    assert d.iloc[0]["tipo"] == "fingerprint_identico"


def test_duplicidade_nao_e_resolvida_automaticamente():
    """
    A funcao audita, nao decide. Escolher um ticker seria resolver questao
    societaria com estatistica - a identidade canonica sai do mapeamento
    documental.
    """
    idx = pd.bdate_range("2020-01-06", periods=80)
    base = np.random.default_rng(1).normal(0, 0.01, 80)
    p = pd.DataFrame({"ANT3": base, "NOV3": base}, index=idx)
    d = retornos.detectar_series_duplicadas(p)
    assert "canonical_instrument_id" not in d.columns
    assert "ticker_escolhido" not in d.columns
    assert {"ticker_a", "ticker_b"} <= set(d.columns)


def test_series_de_tickers_diferentes_nao_sao_emendadas():
    """
    Continuidade canonica nao mora neste modulo. Cada ticker e uma serie
    independente ate a revisao documental autorizar o contrario.
    """
    a = pd.DataFrame({"Date": cal(3), "Adj Close": [10.0, 11.0, 12.0]})
    b = pd.DataFrame({"Date": cal(6)[3:], "Adj Close": [20.0, 21.0, 22.0]})
    painel = retornos.painel_precos_ajustados({"ANT3": a, "NOV3": b}, cal(6))
    assert list(painel.columns) == ["ANT3", "NOV3"]
    assert painel["ANT3"].iloc[3:].isna().all()
    assert painel["NOV3"].iloc[:3].isna().all()


def _cotahist(precos, especi, volume=None, ticker="XPTO3"):
    """Painel COTAHIST minimo, no formato que o modulo espera."""
    idx = cal(len(precos))
    return pd.DataFrame({
        "DATA": idx,
        "CODNEG": ticker,
        "PREULT": precos,
        "ESPECI": especi,
        "VOLTOT": volume if volume is not None else [1e6] * len(precos),
        "TOTNEG": [500] * len(precos),
    })


def test_marcador_de_evento_separa_evento_de_segmento():
    """Confundir os dois descartaria o Novo Mercado inteiro."""
    e = pd.Series(["ON      NM", "ON  ED  NM", "PN  EJ  N1", "PN", "ON      N2",
                   "ON  EDJ NM", "UNT     N2"])
    m = retornos.marcar_eventos_especi(e)
    assert list(m) == [False, True, True, False, False, True, False]


def test_evento_desconhecido_e_tratado_como_evento():
    """Marcador inedito tem que errar descartando o dia, nao aceitando-o."""
    m = retornos.marcar_eventos_especi(pd.Series(["ON  EZZ NM"]))
    assert bool(m.iloc[0]) is True


def test_retorno_bruto_igual_ao_ajustado_entre_eventos():
    """
    O ponto que justifica a via inteira: o fator de ajuste e constante entre
    eventos e cancela na razao.
    """
    brutos = [100.0, 110.0, 99.0, 99.0]
    ajustados = [x * 0.87 for x in brutos]      # fator de ajuste qualquer
    cot = _cotahist(brutos, ["ON      NM"] * 4)
    vol = retornos.painel_volume_financeiro(cot)

    r_bruto = retornos.retornos_preco_bruto_cotahist(cot, cal(4), vol)
    r_ajust = retornos.retornos_simples(
        pd.DataFrame({"XPTO3": ajustados}, index=cal(4)))

    for i in (1, 2, 3):
        assert r_bruto["XPTO3"].iloc[i] == pytest.approx(r_ajust["XPTO3"].iloc[i])


def test_so_o_retorno_que_atravessa_a_fronteira_e_removido():
    """
    No dia ex o preco cai pelo provento sem perda economica. A politica do
    projeto e perder a observacao, nunca estimar o numero.

    O pregao seguinte ao evento nao atravessa fronteira nenhuma e continua
    valido - mascara-lo custaria observacao sem ganho medido.
    """
    cot = _cotahist([100.0, 95.0, 96.0, 97.0],
                    ["ON      NM", "ON  ED  NM", "ON      NM", "ON      NM"])
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, cal(4), vol)

    assert np.isnan(r["XPTO3"].iloc[1])                             # dia ex
    assert r["XPTO3"].iloc[2] == pytest.approx(96.0 / 95.0 - 1.0)   # seguinte, valido
    assert r["XPTO3"].iloc[3] == pytest.approx(97.0 / 96.0 - 1.0)

    opt = retornos.retornos_preco_bruto_cotahist(
        cot, cal(4), vol, mascarar_dia_seguinte=True)
    assert np.isnan(opt["XPTO3"].iloc[2])


def test_evento_de_um_ticker_nao_mascara_os_outros():
    """
    A remocao e por ticker. Se o dia ex de um papel derrubasse o pregao inteiro,
    o painel perderia o mercado todo em datas de dividendo concentrado e a rede
    seria estimada num calendario esburacado por motivo alheio a cada par.
    """
    idx = cal(3)
    cot = pd.DataFrame({
        "DATA": list(idx) * 2,
        "CODNEG": ["COMEV3"] * 3 + ["SEMEV3"] * 3,
        "PREULT": [100.0, 95.0, 96.0, 50.0, 51.0, 52.0],
        "ESPECI": ["ON      NM", "ON  ED  NM", "ON      NM"] + ["ON      NM"] * 3,
        "VOLTOT": [1e6] * 6,
    })
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, idx, vol)

    assert np.isnan(r["COMEV3"].iloc[1])
    assert r["SEMEV3"].iloc[1] == pytest.approx(51.0 / 50.0 - 1.0)


def test_preco_zero_nao_vira_retorno_infinito():
    """
    O COTAHIST registra PREULT zero em papel que nao negociou, e um unico
    infinito contamina qualquer media ou correlacao a jusante.
    """
    cot = _cotahist([10.0, 0.0, 12.0, 13.0], ["ON      NM"] * 4)
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, cal(4), vol)
    assert np.isfinite(r["XPTO3"].dropna()).all()
    assert np.isnan(r["XPTO3"].iloc[1])


def test_retorno_bruto_exige_negociacao_nos_dois_pregoes():
    """A disciplina de nao-sincronia vale igual nesta via."""
    cot = _cotahist([10.0, 11.0, 11.0, 12.0], ["ON      NM"] * 4,
                    volume=[1e6, 1e6, 0.0, 1e6])
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, cal(4), vol)
    assert r["XPTO3"].iloc[1] == pytest.approx(0.10)
    assert np.isnan(r["XPTO3"].iloc[2])         # nao negociou hoje
    assert np.isnan(r["XPTO3"].iloc[3])         # nao negociou ontem


def test_periodo_marcado_remove_so_a_transicao():
    """
    O bug que a auditoria pegou. O marcador do ESPECI dura varios pregoes, e
    nao so o dia ex. Descartar todo dia marcado tirava metade da serie de
    ITUB4 e BBDC4 - justamente os papeis mais liquidos, que sao as lideres.

    O degrau de preco esta na entrada do periodo, nao em cada dia dele.
    """
    marcado = ["ON      NM", "ON  EJ  NM", "ON  EJ  NM", "ON  EJ  NM", "ON      NM"]
    cot = _cotahist([100.0, 95.0, 96.0, 97.0, 98.0], marcado)
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, cal(5), vol)

    assert np.isnan(r["XPTO3"].iloc[1])                             # entrada
    assert r["XPTO3"].iloc[2] == pytest.approx(96.0 / 95.0 - 1.0)   # dentro, valido
    assert r["XPTO3"].iloc[3] == pytest.approx(97.0 / 96.0 - 1.0)
    assert r["XPTO3"].iloc[4] == pytest.approx(98.0 / 97.0 - 1.0)   # saida, valido


def test_eventos_colados_contam_como_dois():
    """
    ED seguido de EJ sao duas fronteiras, com dois degraus. Uma regra de
    "inicio de sequencia marcada" veria uma so e deixaria o segundo degrau
    entrar como retorno.
    """
    cot = _cotahist([100.0, 95.0, 90.0, 91.0],
                    ["ON      NM", "ON  ED  NM", "ON  EJ  NM", "ON      NM"])
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, cal(4), vol)

    assert np.isnan(r["XPTO3"].iloc[1])
    assert np.isnan(r["XPTO3"].iloc[2])
    assert r["XPTO3"].iloc[3] == pytest.approx(91.0 / 90.0 - 1.0)


def test_token_de_evento_ignora_segmento_de_governanca():
    t = retornos.token_evento_especi(
        pd.Series(["ON      NM", "ON  ED  NM", "PN  EJ  N1", "UNT     N2"]))
    assert list(t) == ["", "ED", "EJ", ""]


def test_grupamento_redondo_nao_cai_em_sem_causa():
    """
    MAPT4 saltou de R$ 10,00 para R$ 40,00 num pregao, sem marcador no ESPECI.
    Quatro vezes exatos e grupamento que a B3 nao sinalizou. Sem esta checagem
    o caso cairia em "sem causa", que e onde erro de dado se esconde.
    """
    assert retornos._razao_de_grupamento(10.0, 40.0)[1] == "1:4"
    assert retornos._razao_de_grupamento(40.0, 10.0)[1] == "4:1"
    assert retornos._razao_de_grupamento(10.0, 16.3)[1] is None      # alta real
    assert retornos._razao_de_grupamento(0.0, 5.0)[1] is None        # preco zero


def test_auditoria_de_extremos_nao_altera_o_painel():
    """A funcao audita. Winsorizar ou cortar aqui esconderia o problema."""
    idx = cal(3)
    cot = _cotahist([10.0, 40.0, 41.0], ["ON      NM"] * 3, ticker="MAPT4")
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, idx, vol)
    antes = r.copy()

    ext = retornos.auditar_retornos_extremos(r, cot, idx)
    assert len(ext) == 1
    assert ext.iloc[0]["fator_redondo"] == "1:4"
    assert ext.iloc[0]["causa_provavel"] == "possivel_grupamento_nao_marcado"
    pd.testing.assert_frame_equal(r, antes)


# A tabela classifica a força da prova, não autoriza alterar a máscara. Token
# com prova fraca segue mascarado; abrir excecao olhando o resultado seria
# ajustar a regra ao que ela produziu.
def _painel_com_evento(n_tickers, token="ED", precos=(100.0, 95.0, 96.0, 97.0)):
    """n tickers identicos, cada um com uma fronteira do mesmo token."""
    idx = cal(len(precos))
    especi = ["ON      NM"] * len(precos)
    especi[1] = f"ON  {token:<3} NM"
    linhas = []
    for i in range(n_tickers):
        linhas.append(pd.DataFrame({
            "DATA": idx, "CODNEG": f"T{i:03d}", "PREULT": list(precos),
            "ESPECI": especi, "VOLTOT": [1e6] * len(precos),
            "TOTNEG": [500] * len(precos),
        }))
    return pd.concat(linhas, ignore_index=True), idx


def _referencia(cot, idx, tickers, erro_na_fronteira):
    """Painel de referencia: retorno igual ao bruto, menos o erro na fronteira."""
    bruto = retornos.retornos_simples(
        cot.pivot_table(index="DATA", columns="CODNEG", values="PREULT",
                        aggfunc="last").reindex(idx))
    ref = bruto.copy()
    for tk in tickers:
        ref.loc[idx[1], tk] = bruto.loc[idx[1], tk] + erro_na_fronteira
    return ref


def test_token_com_fronteira_sustentada():
    """Erro material nas remocoes: a remocao esta comprando alguma coisa."""
    cot, idx = _painel_com_evento(40)
    ref = _referencia(cot, idx, [f"T{i:03d}" for i in range(40)], 0.05)
    vol = retornos.painel_volume_financeiro(cot)

    v = retornos.validar_fronteiras_por_token(cot, idx, ref, vol)
    linha = v[v["token"] == "ED"].iloc[0]
    assert linha["classificacao_evidencia"] == retornos.EVID_SUSTENTADA
    assert linha["fronteira_erro_mediano"] == pytest.approx(0.05)
    assert linha["fronteira_frac_acima_50bps"] == pytest.approx(1.0)


def test_token_com_evidencia_mista():
    """
    Mediana no ruido, mas parte das fronteiras se move. Nao vira "sustentado"
    nem autoriza tirar o token da mascara.
    """
    cot, idx = _painel_com_evento(40)
    tickers = [f"T{i:03d}" for i in range(40)]
    ref = _referencia(cot, idx, tickers, 0.0)
    for tk in tickers[:5]:                       # so alguns realmente se movem
        ref.loc[idx[1], tk] += 0.08

    v = retornos.validar_fronteiras_por_token(
        cot, idx, ref, retornos.painel_volume_financeiro(cot))
    linha = v[v["token"] == "ED"].iloc[0]
    assert linha["classificacao_evidencia"] == retornos.EVID_MISTA
    assert linha["fronteira_erro_mediano"] < retornos.LIMIAR_ERRO_MATERIAL
    assert linha["fronteira_frac_acima_50bps"] == pytest.approx(5 / 40)


def test_token_sem_referencia_externa():
    """Sem serie de referencia nao ha prova nem contraprova - e uma terceira coisa."""
    cot, idx = _painel_com_evento(40)
    vazio = pd.DataFrame(index=idx)

    v = retornos.validar_fronteiras_por_token(
        cot, idx, vazio, retornos.painel_volume_financeiro(cot))
    linha = v[v["token"] == "ED"].iloc[0]
    assert linha["classificacao_evidencia"] == retornos.EVID_SEM_REF
    assert linha["fronteira_n_total_cotahist"] == 40
    assert linha["fronteira_n_suporte_comum"] == 0


def test_token_com_amostra_insuficiente():
    cot, idx = _painel_com_evento(5)
    ref = _referencia(cot, idx, [f"T{i:03d}" for i in range(5)], 0.05)

    v = retornos.validar_fronteiras_por_token(
        cot, idx, ref, retornos.painel_volume_financeiro(cot))
    assert v[v["token"] == "ED"].iloc[0]["classificacao_evidencia"] == retornos.EVID_AMOSTRA


def test_denominadores_cotahist_e_suporte_comum_nao_se_misturam():
    """
    A fracao removida vale sobre o COTAHIST inteiro; o erro so pode ser medido
    onde ha referencia. Reportar um numero sobre o denominador do outro seria
    inflar ou desinflar a conclusao.
    """
    cot, idx = _painel_com_evento(40)
    tickers = [f"T{i:03d}" for i in range(40)]
    ref = _referencia(cot, idx, tickers, 0.05)[tickers[:12]]   # so 12 com referencia

    v = retornos.validar_fronteiras_por_token(
        cot, idx, ref, retornos.painel_volume_financeiro(cot))
    linha = v[v["token"] == "ED"].iloc[0]
    assert linha["fronteira_n_total_cotahist"] == 40
    assert linha["fronteira_n_suporte_comum"] == 12
    assert linha["fronteira_cobertura_referencia"] == pytest.approx(0.3)


def test_segundo_evento_com_mesmo_token_escapa_da_r3():
    """
    Limitacao conhecida e nao corrigida. Se o marcador continua ligado e um
    segundo evento do mesmo tipo acontece, o token nao muda e a fronteira nao e
    detectada. O degrau entra no painel como retorno.

    Fica registrado em teste para nao ser redescoberto como surpresa.
    """
    cot = _cotahist([100.0, 95.0, 90.0, 91.0],
                    ["ON      NM", "ON  ED  NM", "ON  ED  NM", "ON      NM"])
    vol = retornos.painel_volume_financeiro(cot)
    r = retornos.retornos_preco_bruto_cotahist(cot, cal(4), vol)

    assert np.isnan(r["XPTO3"].iloc[1])                              # 1o evento, pego
    assert r["XPTO3"].iloc[2] == pytest.approx(90.0 / 95.0 - 1.0)    # 2o, escapa


def test_validacao_sem_nenhum_token_devolve_tabela_vazia():
    cot = _cotahist([10.0, 11.0, 12.0], ["ON      NM"] * 3)
    v = retornos.validar_fronteiras_por_token(
        cot, cal(3), pd.DataFrame(index=cal(3)),
        retornos.painel_volume_financeiro(cot))
    assert v.empty


def test_validacao_nao_altera_o_painel_de_entrada():
    cot, idx = _painel_com_evento(40)
    antes = cot.copy()
    ref = _referencia(cot, idx, [f"T{i:03d}" for i in range(40)], 0.05)
    ref_antes = ref.copy()

    retornos.validar_fronteiras_por_token(
        cot, idx, ref, retornos.painel_volume_financeiro(cot))

    pd.testing.assert_frame_equal(cot, antes)
    pd.testing.assert_frame_equal(ref, ref_antes)


def test_token_sem_erro_nenhum_nao_cai_em_evidencia_mista():
    """
    A mediana sozinha nao separa. Um token cuja remocao nunca move nada e
    diferente de um cuja mediana e baixa mas a cauda se move: o primeiro e
    ausencia de sustentacao, o segundo e evidencia mista.

    Nenhum dos dois autoriza tirar o token da mascara.
    """
    cot, idx = _painel_com_evento(40)
    ref = _referencia(cot, idx, [f"T{i:03d}" for i in range(40)], 0.0)

    v = retornos.validar_fronteiras_por_token(
        cot, idx, ref, retornos.painel_volume_financeiro(cot))
    linha = v[v["token"] == "ED"].iloc[0]
    assert linha["classificacao_evidencia"] == retornos.EVID_NAO_SUSTENTADA
    assert linha["fronteira_frac_acima_10bps"] == pytest.approx(0.0)


def test_fracao_acima_de_10bps_entra_na_regra():
    """
    Mediana identica, classificacoes diferentes: o que separa e quantas
    fronteiras se movem. Se a fracao nao fosse usada, os dois cairiam juntos.
    """
    tickers = [f"T{i:03d}" for i in range(40)]

    def _classificar(quantos_se_movem):
        cot, idx = _painel_com_evento(40)
        ref = _referencia(cot, idx, tickers, 0.0)
        for tk in tickers[:quantos_se_movem]:
            ref.loc[idx[1], tk] += 0.08
        v = retornos.validar_fronteiras_por_token(
            cot, idx, ref, retornos.painel_volume_financeiro(cot))
        return v[v["token"] == "ED"].iloc[0]

    poucos = _classificar(2)          # 5% das fronteiras
    varios = _classificar(12)         # 30% das fronteiras
    assert poucos["fronteira_erro_mediano"] == varios["fronteira_erro_mediano"]
    assert poucos["classificacao_evidencia"] == retornos.EVID_NAO_SUSTENTADA
    assert varios["classificacao_evidencia"] == retornos.EVID_MISTA


def test_corte_de_amostra_minima_e_parametro_e_nao_conclusao():
    """
    O corte de 30 e convencao. Se a leitura muda quando ele muda, isso precisa
    ser visivel - por isso ele e parametro nomeado e ha reclassificacao barata.
    """
    cot, idx = _painel_com_evento(20)
    ref = _referencia(cot, idx, [f"T{i:03d}" for i in range(20)], 0.05)
    vol = retornos.painel_volume_financeiro(cot)

    v = retornos.validar_fronteiras_por_token(cot, idx, ref, vol)
    assert v.iloc[0]["classificacao_evidencia"] == retornos.EVID_AMOSTRA

    frouxo = retornos.validar_fronteiras_por_token(
        cot, idx, ref, vol, min_fronteiras_com_referencia=10)
    assert frouxo.iloc[0]["classificacao_evidencia"] == retornos.EVID_SUSTENTADA

    assert list(retornos.reclassificar_evidencia(v, min_fronteiras=10)) == \
        list(frouxo["classificacao_evidencia"])


def test_pnl_mantem_o_dia_ex_que_a_estimacao_remove():
    """
    A estimacao descarta o degrau do dia ex; o painel de P&L o mantem como
    retorno de preco bruto. O teste nao inclui fluxo de proventos em caixa.
    """
    cot = _cotahist([100.0, 95.0, 96.0, 97.0],
                    ["ON      NM", "ON  ED  NM", "ON      NM", "ON      NM"])
    vol = retornos.painel_volume_financeiro(cot)

    estimacao = retornos.retornos_preco_bruto_cotahist(cot, cal(4), vol)
    pnl = retornos.retornos_pnl_marcacao(cot, cal(4))

    assert np.isnan(estimacao["XPTO3"].iloc[1])                    # mascarado
    assert pnl["XPTO3"].iloc[1] == pytest.approx(95.0 / 100.0 - 1)  # mantido


def test_pnl_marca_a_ultimo_preco_no_dia_sem_negocio():
    """
    Papel que nao negociou fica marcado no ultimo preco (retorno zero) e o
    salto aparece inteiro no pregao em que volta a negociar. A soma no
    periodo e a variacao entre precos observados - o resultado economico.
    """
    cot = _cotahist([100.0, 0.0, 121.0, 121.0],
                    ["ON      NM"] * 4)
    pnl = retornos.retornos_pnl_marcacao(cot, cal(4))

    assert np.isnan(pnl["XPTO3"].iloc[0])
    assert pnl["XPTO3"].iloc[1] == pytest.approx(0.0)      # marcado no ultimo
    assert pnl["XPTO3"].iloc[2] == pytest.approx(0.21)     # salto inteiro
    assert pnl["XPTO3"].iloc[3] == pytest.approx(0.0)


def test_pnl_nao_preenche_para_tras():
    """Antes da primeira observacao o papel nao existe - NaN, nunca preco."""
    cot = _cotahist([0.0, 0.0, 100.0, 110.0], ["ON      NM"] * 4)
    pnl = retornos.retornos_pnl_marcacao(cot, cal(4))
    assert pnl["XPTO3"].iloc[:3].isna().all()
    assert pnl["XPTO3"].iloc[3] == pytest.approx(0.10)


def test_medir_travessias_conta_posicao_dias_no_dia_ex():
    cot = _cotahist([100.0, 95.0, 96.0, 97.0],
                    ["ON      NM", "ON  ED  NM", "ON      NM", "ON      NM"])
    fronteiras = retornos.painel_fronteiras_evento(cot, cal(4))

    posicoes = pd.DataFrame({"XPTO3": [0.0, 0.5, 0.5, 0.0]}, index=cal(4))
    m = retornos.medir_travessias_de_evento(posicoes, fronteiras)

    assert m["posicao_dias"] == 2
    assert m["travessias_de_evento"] == 1      # so o dia ex com posicao ativa
    assert m["frac_travessias"] == pytest.approx(0.5)
