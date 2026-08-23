"""
Testes de aquisicao, cache e classificacao de instrumentos.

Os testes de classificacao usam dados sinteticos (sem rede) e travam decisoes
que custaram investigacao para chegar. Cada um corresponde a um caso real
encontrado no COTAHIST de 2015-2026.
"""

import pandas as pd
import pytest

from src import dados

FASE_1 = "pendente: implementacao na fase 1"


def linha(codneg, especi, isin, tpmerc="010", codbdi="02", preult=10.0,
          totneg=100, quatot=1000, voltot=10000.0, data="2020-01-02"):
    """Monta um registro sintetico no formato que o parser produz."""
    return {
        "DATA": pd.Timestamp(data), "CODNEG": codneg, "ESPECI": especi,
        "CODISI": isin, "TPMERC": tpmerc, "CODBDI": codbdi,
        "PREULT": preult, "PREMIN": preult, "PREMAX": preult, "PREMED": preult,
        "TOTNEG": totneg, "QUATOT": quatot, "VOLTOT": voltot,
        # Usa a funcao de producao em vez de recalcular a mao, para o teste
        # nunca divergir do codigo que ele deveria estar testando.
        "TIPO_PAPEL": dados.tipo_instrumento(pd.Series([especi])).iat[0],
        "ISIN_TIPO": isin[6:9],
    }


def painel(*linhas):
    return pd.DataFrame(list(linhas))


@pytest.mark.parametrize("especi,esperado", [
    ("ON      NM", "ON"),      # ordinaria, Novo Mercado
    ("ON  ED  NM", "ON"),      # mesma acao no dia ex-dividendo
    ("ON  EJ  NM", "ON"),      # mesma acao no dia ex-juros
    ("PN      N1", "PN"),
    ("UNT     N2", "UNT"),
    ("PNA ED  N1", "PNA"),
    ("DRN       ", "DRN"),
    # A B3 escreve rotulos truncados com pontuacao solta. O mesmo IBOV11 e
    # "IBO" ate 2024 e "IBO/" a partir de 2025; o SMLL11 e "SML)". Normalizar
    # colapsa as variantes em vez de o codigo catalogar cada pontuacao nova.
    ("IBO/      ", "IBO"),
    ("SML)      ", "SML"),
    ("          ", ""),
])
def test_tipo_instrumento_normaliza_token_e_pontuacao(especi, esperado):
    """
    O ESPECI e composto: tipo + marcador de evento + segmento. Os marcadores
    aparecem so nos dias de evento, entao o campo do mesmo ticker muda de um
    dia para o outro. Comparar a string inteira quebraria de forma
    intermitente - o pior tipo de defeito.
    """
    assert dados.tipo_instrumento(pd.Series([especi])).iat[0] == esperado


def test_normalizacao_nao_colide_tipos_distintos():
    """
    A normalizacao so vale se nao fundir categorias diferentes. Verifica que
    todos os tipos observados em 2015-2026 continuam distintos depois dela.
    """
    observados = ["ON", "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF", "UNT",
                  "DRN", "DR1", "DR2", "DR3", "DRE", "CI", "IBO", "IBO/", "SML)"]
    norm = dados.tipo_instrumento(pd.Series(observados))
    assert norm.tolist().count("IBO") == 2
    sem_ibo = [t for t in norm.tolist() if t != "IBO"]
    assert len(sem_ibo) == len(set(sem_ibo))


def test_bdr_e_excluido_mesmo_sendo_lote_padrao():
    """
    O caso que justifica o filtro de ESPECI existir. BDR e negociado em lote
    padrao ate 2022, entao passa inteiro pelo filtro de CODBDI. Em 2015 sao
    ~90 dos 555 tickers do pool. Um BDR acompanha mecanicamente o fechamento
    da bolsa de origem e quase nao negocia aqui; como seguidora, produziria um
    lead-lag enorme e completamente falso.
    """
    df = painel(
        linha("PETR4", "PN      N2", "BRPETRACNPR6"),
        linha("CHVX34", "DRN       ", "BRCHVXBDR008", totneg=1),
    )
    out = dados.filtrar_acoes_a_vista(df)
    assert set(out["CODNEG"]) == {"PETR4"}


def test_units_e_preferenciais_terminadas_em_11_sao_mantidas():
    """
    Por que nao se filtra por sufixo de ticker: units terminam em 11 (SANB11,
    TAEE11, ALUP11) e preferenciais de classe E e F tambem (BRGE11, BRGE12).
    Uma regra "termina em 11 e ETF" jogaria fora acoes brasileiras legitimas.
    """
    df = painel(
        linha("SANB11", "UNT       ", "BRSANBCDAM13"),
        linha("BBTG11", "UNT       ", "BRBBTGUNT007"),   # unit com ISIN UNT
        linha("BRGE11", "PNE       ", "BRBRGEACNPE1"),   # preferencial classe E
        linha("BOVA11", "CI        ", "BRBOVACTF003", codbdi="14"),  # ETF
    )
    out = dados.filtrar_acoes_a_vista(df)
    assert set(out["CODNEG"]) == {"SANB11", "BBTG11", "BRGE11"}


def test_instrumento_de_indice_com_rotulo_estranho_e_excluido_pelo_isin():
    """
    A B3 escreve o ESPECI de instrumentos de indice de forma inconsistente: o
    mesmo IBOV11 aparece como "IBO" em 2015 e "IBO/" em 2025, e o SMLL11 como
    "SML)". Sao rotulos truncados pela propria bolsa. O ISIN, ao contrario,
    diz IND de forma estavel em qualquer ano - por isso ele classifica
    primeiro.
    """
    df = painel(
        linha("IBOV11", "IBO/      ", "BRIBOVINDM18", preult=120000.0),
        linha("SMLL11", "SML)      ", "BRSMLLINDM18", preult=2033.0),
        linha("VALE3", "ON      NM", "BRVALEACNOR0"),
    )
    out = dados.filtrar_acoes_a_vista(df)
    assert set(out["CODNEG"]) == {"VALE3"}


def test_recibo_de_subscricao_e_excluido():
    """
    GOLL54, AZUL53, JSLG11: o ESPECI diz ON/PN mas o ISIN diz A01/A02/A03,
    que sao instrumentos temporarios de aumento de capital. Exigir que as
    duas fontes concordem descarta esses casos automaticamente.
    """
    df = painel(
        linha("GOLL4", "PN      N2", "BRGOLLACNPR8"),
        linha("GOLL54", "PN      N2", "BRGOLLA01PR3"),
    )
    out = dados.filtrar_acoes_a_vista(df)
    assert set(out["CODNEG"]) == {"GOLL4"}


def test_tipo_desconhecido_com_isin_de_acao_levanta_erro():
    """
    A rede de seguranca. Se a B3 criar uma categoria nova de acao num ano
    futuro, o pipeline para e pede classificacao humana, em vez de deixar o
    instrumento entrar sem ninguem perceber. Foi assim que DRE (BDR de ETF,
    2022) e SML) (indice small cap, 2024) foram descobertos.
    """
    df = painel(
        linha("PETR4", "PN      N2", "BRPETRACNPR6"),
        linha("XPTO9", "ZZZ       ", "BRXPTOACNOR0"),
    )
    with pytest.raises(dados.TipoInstrumentoDesconhecido, match="ZZZ"):
        dados.filtrar_acoes_a_vista(df)


def test_tipo_desconhecido_com_isin_de_nao_acao_nao_levanta_erro():
    """
    Contrapartida do teste anterior: se o ISIN ja diz que nao e acao, nao ha
    o que um humano decidir. Bloqueia e segue, sem interromper o pipeline.
    """
    df = painel(
        linha("PETR4", "PN      N2", "BRPETRACNPR6"),
        linha("XPTO39", "ZZZ       ", "BRXPTOBDR000"),
    )
    out = dados.filtrar_acoes_a_vista(df)
    assert set(out["CODNEG"]) == {"PETR4"}


def test_mercado_a_termo_e_opcoes_ficam_de_fora():
    """So mercado a vista (TPMERC=010) entra no universo."""
    df = painel(
        linha("PETR4", "PN      N2", "BRPETRACNPR6", tpmerc="010"),
        linha("BBAS3T", "ON      NM", "BRBBASACNOR3", tpmerc="030", codbdi="62"),
    )
    out = dados.filtrar_acoes_a_vista(df)
    assert set(out["CODNEG"]) == {"PETR4"}


def test_duplicata_no_pool_a_vista_levanta_erro():
    """
    Duas linhas para o mesmo (data, ticker) no mercado a vista deixariam o
    painel de precos ambiguo. No termo a duplicidade e legitima (varios
    contratos com prazos diferentes), por isso a checagem e escopada ao pool.
    """
    df = painel(
        linha("PETR4", "PN      N2", "BRPETRACNPR6"),
        linha("PETR4", "PN      N2", "BRPETRACNPR6", preult=11.0),
    )
    df["DISMES"] = "000"
    with pytest.raises(dados.ErroLayoutCotahist, match="duplicatas"):
        dados.validar_dataframe(df)


def test_voltot_negativo_levanta_erro():
    df = painel(linha("PETR4", "PN      N2", "BRPETRACNPR6", voltot=-1.0))
    df["DISMES"] = "000"
    with pytest.raises(dados.ErroLayoutCotahist, match="VOLTOT"):
        dados.validar_dataframe(df)


def test_retornos_usam_preco_ajustado():
    """
    Retorno calculado sobre preco nao ajustado registraria a queda do dia do
    provento como se fosse perda economica real.
    """
    pytest.skip(FASE_1)


def test_volume_financeiro_nao_usa_preco_ajustado():
    """
    Resolvido pela fonte: o COTAHIST traz VOLTOT, o volume financeiro em reais
    calculado pela propria bolsa. Este teste passa a verificar que o pipeline
    usa VOLTOT e nao recalcula a partir de preco.
    """
    pytest.skip(FASE_1)


def test_dia_sem_pregao_nao_vira_dado_faltante():
    """
    Feriado da B3 e ausencia de pregao, nao buraco no dado. Confundir os dois
    contamina a metrica de cobertura e o detector de preco velho.
    """
    pytest.skip(FASE_1)


def test_flags_calculadas_antes_de_qualquer_preenchimento():
    """
    Se o painel for alinhado ao calendario e preenchido antes de calcular as
    flags, um papel que ficou 5 dias sem negociar ganha 5 fechamentos
    identicos e retornos zero fabricados pelo proprio pre-processamento. O
    detector de preco velho passaria a detectar a si mesmo.
    """
    pytest.skip(FASE_1)


def test_cache_evita_rebaixar_dado_existente():
    """O pipeline precisa rodar offline depois do primeiro download."""
    pytest.skip(FASE_1)


def test_auditoria_marca_ticker_com_serie_quebrada():
    """Nenhum ticker entra no backtest sem passar pela auditoria."""
    pytest.skip(FASE_1)
