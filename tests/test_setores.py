"""
Testes da tabela setorial curada.

O setor e uma entrada da metodologia, nao um resultado. As garantias que
importam: a fonte do Yahoo nunca entra, sucessao nao herda classificacao,
classificacao de hoje nao vira historia por omissao, e quem nao tem evidencia
vigente fica fora dos pares em vez de entrar por chute.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import setores


TABELA_CURADA = (Path(__file__).resolve().parents[1]
                 / "data" / "reference" / "setores_b3.csv")


def _linha(ticker="ABCD3", **kwargs):
    base = {
        "ticker_observado": ticker,
        "emissor_id": "ABCD",
        "setor": "Financeiro",
        "subsetor": "Intermediarios Financeiros",
        "segmento": "Bancos",
        "fonte": "B3 - Classificacao Setorial",
        "data_fonte": "2026-07-31",
        "evidencia": "planilha oficial, aba Setorial",
        "estabilidade_documentada": "",
        "validade_inicio": "2015-01-01",
        "validade_fim": "",
        "confianca": "alta",
        "status_revisao": "confirmado",
    }
    base.update(kwargs)
    return base


def _tabela(*linhas):
    return pd.DataFrame(list(linhas) or [_linha()])[list(setores.COLUNAS)]


def _periodos(*pares):
    return pd.DataFrame(
        [{"ticker_observado": t, "inicio": i, "fim": f} for t, i, f in pares])


def test_tabela_bem_formada_passa():
    assert setores.validar(_tabela()) == []


def test_tabela_curada_versionada_e_integra():
    """A entrada metodológica real também passa pelo contrato, não só fixtures."""
    tab = setores.carregar(TABELA_CURADA)
    assert setores.validar(tab) == []
    assert len(tab) == 192
    assert tab["ticker_observado"].nunique() == 189
    assert tab["status_revisao"].value_counts().to_dict() == {
        "confirmado": 191,
        "evidencia_insuficiente": 1,
    }


def test_fronteiras_anuais_curadas_de_hype_e_raiz():
    """Congela as duas convenções interval-censored declaradas na referência."""
    tab = setores.carregar(TABELA_CURADA)

    hype_antigo = setores.setor_vigente(tab, "HYPE3", "2017-12-31")
    hype_novo = setores.setor_vigente(tab, "HYPE3", "2018-01-01")
    assert hype_antigo["setor"] == "Consumo não Cíclico"
    assert hype_novo["setor"] == "Saúde"

    raiz_antiga = setores.setor_vigente(tab, "RAIZ4", "2023-12-31")
    raiz_nova = setores.setor_vigente(tab, "RAIZ4", "2024-01-01")
    assert raiz_antiga["subsetor"] == "Agropecuária"
    assert raiz_nova["setor"] == "Petróleo, Gás e Biocombustíveis"


def test_lacuna_final_de_itsa_nao_recebe_classe_por_suposição():
    """A reclassificação foi observada depois, mas o dia exato é desconhecido."""
    tab = setores.carregar(TABELA_CURADA)
    assert setores.setor_vigente(tab, "ITSA4", "2026-07-23")["segmento"] == "Bancos"
    assert setores.setor_vigente(tab, "ITSA4", "2026-07-24") is None
    assert setores.setor_vigente(tab, "ITSA4", "2026-07-27") is None


def test_esqueleto_tem_o_schema():
    assert list(setores.tabela_vazia().columns) == list(setores.COLUNAS)


def test_setor_do_yahoo_reprova_a_tabela():
    """
    Proibicao metodologica, nao preferencia: o campo do Yahoo nao e oficial,
    nao e point-in-time e nao cobre papel deslistado.
    """
    for fonte in ("yfinance", "Yahoo Finance", "yf.Ticker.info"):
        erros = setores.validar(_tabela(_linha(fonte=fonte)))
        assert any("fonte proibida" in e for e in erros), fonte


def test_confirmado_exige_a_chave_de_grupo_fonte_evidencia_e_emissor():
    """Na V1 os grupos se formam por (setor, subsetor): os dois sao exigidos."""
    for coluna in ("setor", "subsetor", "fonte", "data_fonte", "evidencia",
                   "emissor_id"):
        erros = setores.validar(_tabela(_linha(**{coluna: ""})))
        assert any(coluna in e for e in erros), coluna


def test_segmento_pode_ficar_vazio_em_confirmado():
    """`segmento` e informativo na V1, reservado para robustez posterior."""
    assert setores.validar(_tabela(_linha(segmento=""))) == []


def test_classificacao_atual_nao_passa_como_historica():
    """
    A planilha da B3 diz o setor de hoje. Confirmada sem validade_inicio seria
    lida como "valeu desde sempre", que e afirmacao que a fonte nao faz.
    """
    erros = setores.validar(_tabela(_linha(validade_inicio="")))
    assert any("validade_inicio" in e for e in erros)

    # Com estabilidade documentada, o inicio aberto e aceito.
    ok = _linha(validade_inicio="", estabilidade_documentada="sim",
                evidencia="prospecto 2014 e formularios de referencia 2015-2026")
    assert setores.validar(_tabela(ok)) == []


def test_fim_aberto_e_permitido():
    """Classificacao que continua vigente nao precisa de data de fim."""
    assert setores.validar(_tabela(_linha(validade_fim=""))) == []


def test_pendente_pode_ficar_incompleto():
    """Linha pendente existe justamente para registrar o que falta pesquisar."""
    incompleta = _linha(setor="", fonte="", data_fonte="", evidencia="",
                        emissor_id="", validade_inicio="", confianca="",
                        status_revisao="pendente")
    assert setores.validar(_tabela(incompleta)) == []


def test_vocabulario_fechado():
    assert setores.validar(_tabela(_linha(status_revisao="mais_ou_menos")))
    assert setores.validar(_tabela(_linha(confianca="talvez")))
    assert setores.validar(_tabela(_linha(estabilidade_documentada="acho_que_sim")))


def test_validades_sobrepostas_reprovam():
    """Sobreposicao tornaria 'qual setor vigia?' ambiguo, resolvido em silencio."""
    t = _tabela(_linha("XPTO3", validade_inicio="2015-01-01",
                       validade_fim="2021-06-30"),
                _linha("XPTO3", validade_inicio="2021-01-01"))
    assert any("sobrepoem" in e for e in setores.validar(t))


def test_duas_classes_do_emissor_preservam_ticker_observado():
    """
    O emissor agrupa a pesquisa; o ticker observado continua sendo a chave. Sem
    isso PETR3 e PETR4 colapsariam numa linha so e a tabela perderia a
    capacidade de dizer qual codigo foi classificado.
    """
    t = _tabela(_linha("PETR3", emissor_id="PETR", setor="Petroleo"),
                _linha("PETR4", emissor_id="PETR", setor="Petroleo"))
    assert setores.validar(t) == []
    assert setores.setor_vigente(t, "PETR3", "2020-01-02")["ticker_observado"] == "PETR3"
    assert setores.setor_vigente(t, "PETR4", "2020-01-02")["ticker_observado"] == "PETR4"


def test_data_ilegivel_reprova():
    assert setores.validar(_tabela(_linha(data_fonte="ontem")))


def test_setor_vigente_ignora_linha_pendente():
    """
    Linha pendente registra o que falta pesquisar. Devolve-la seria usar um
    setor nao confirmado como se fosse.
    """
    t = _tabela(_linha("XPTO3", setor="Consumo", status_revisao="pendente"))
    assert setores.setor_vigente(t, "XPTO3", "2020-01-02") is None

    t2 = _tabela(_linha("XPTO3", status_revisao="evidencia_insuficiente"))
    assert setores.setor_vigente(t2, "XPTO3", "2020-01-02") is None


def test_validade_temporal_escolhe_a_linha_certa():
    t = _tabela(
        _linha("XPTO3", setor="Consumo", validade_inicio="2015-01-01",
               validade_fim="2020-12-31"),
        _linha("XPTO3", setor="Tecnologia", validade_inicio="2021-01-01"))
    assert setores.setor_vigente(t, "XPTO3", "2019-06-01")["setor"] == "Consumo"
    assert setores.setor_vigente(t, "XPTO3", "2022-06-01")["setor"] == "Tecnologia"


def test_confirmado_fora_da_validade_nao_e_operavel():
    """
    Confirmado a partir de 2021 nao classifica 2017. Tratar como se
    classificasse usaria informacao posterior para montar universo antigo.
    """
    t = _tabela(_linha("XPTO3", validade_inicio="2021-01-01"))
    assert setores.setor_vigente(t, "XPTO3", "2019-06-01") is None
    assert setores.tickers_operaveis_em(t, "2019-06-01") == set()
    assert setores.tickers_operaveis_em(t, "2022-06-01") == {"XPTO3"}


def test_operabilidade_exige_confirmacao_e_vigencia():
    t = _tabela(_linha("AAAA3", validade_inicio="2015-01-01"),
                _linha("BBBB3", emissor_id="BBBB", status_revisao="pendente"),
                _linha("CCCC3", emissor_id="CCCC",
                       status_revisao="evidencia_insuficiente"))
    assert setores.tickers_operaveis_em(t, "2020-01-02") == {"AAAA3"}


def test_sucessor_nao_herda_setor_do_antecessor():
    """
    ESTC3 confirmado nao classifica YDUQ3. Fusao pode mudar o setor da empresa
    resultante, e herdar calado registraria suposicao como fato.
    """
    t = _tabela(_linha("ESTC3", emissor_id="ESTC", setor="Consumo"))
    assert setores.setor_vigente(t, "YDUQ3", "2020-01-02") is None
    assert "YDUQ3" not in setores.tickers_operaveis_em(t, "2020-01-02")


def test_lacuna_distingue_sem_linha_de_sem_confirmacao():
    t = _tabela(_linha("AAAA3", validade_inicio="2015-01-01"),
                _linha("BBBB3", emissor_id="BBBB", status_revisao="pendente"))
    lac = setores.lacunas(t, _periodos(
        ("AAAA3", "2015-01-01", "2026-01-01"),
        ("BBBB3", "2015-01-01", "2026-01-01"),
        ("CCCC3", "2015-01-01", "2026-01-01")))
    sit = dict(zip(lac["ticker_observado"], lac["situacao"]))
    assert "AAAA3" not in sit
    assert sit["BBBB3"] == "sem_confirmacao"
    assert sit["CCCC3"] == "sem_linha"


def test_lacuna_temporal_nao_e_engolida_pelo_ultimo_status():
    """
    Confirmado de 2021 em diante nao cobre 2015-2020. Reduzir o ticker ao
    ultimo status daria o papel por resolvido com metade do periodo descoberta.
    """
    t = _tabela(_linha("XPTO3", validade_inicio="2021-01-01"))
    lac = setores.lacunas(t, _periodos(("XPTO3", "2015-01-01", "2026-01-01")))
    assert len(lac) == 1
    assert lac.iloc[0]["situacao"] == "lacuna_temporal"
    assert str(lac.iloc[0]["inicio"]) == "2015-01-01"
    assert str(lac.iloc[0]["fim"]) == "2020-12-31"


def test_lacuna_no_meio_do_periodo():
    t = _tabela(
        _linha("XPTO3", validade_inicio="2015-01-01", validade_fim="2017-12-31"),
        _linha("XPTO3", validade_inicio="2020-01-01"))
    lac = setores.lacunas(t, _periodos(("XPTO3", "2015-01-01", "2026-01-01")))
    assert len(lac) == 1
    assert str(lac.iloc[0]["inicio"]) == "2018-01-01"
    assert str(lac.iloc[0]["fim"]) == "2019-12-31"


def test_periodo_totalmente_coberto_nao_gera_lacuna():
    t = _tabela(_linha("XPTO3", validade_inicio="2010-01-01"))
    assert setores.lacunas(t, _periodos(("XPTO3", "2015-01-01", "2026-01-01"))).empty


def test_emissor_sai_do_isin_bem_formado():
    assert setores.emissor_do_isin("BRPETRACNPR6") == "PETR"
    assert setores.emissor_do_isin("BRVALEACNOR0") == "VALE"


def test_isin_ausente_ou_invalido_nao_vira_emissor_inventado():
    """
    Fatiar CODISI[2:6] sobre campo vazio ou truncado produziria uma string
    qualquer, e essa string agruparia empresas sem relacao nenhuma.
    """
    for ruim in ("", "   ", None, "BR123", "BRPETRACNPR6X", 42):
        assert setores.emissor_do_isin(ruim) is None


def test_auditoria_detecta_isin_invalido_e_colisao_de_emissor():
    painel = pd.DataFrame({
        "CODNEG": ["AAAA3", "BBBB3", "BBBB4", "CCCC3"],
        "CODISI": ["BRAAAAACNOR0", "BRBBBBACNOR0", "BRBBBBACNPR0", "lixo"],
        "NOMRES": ["ALFA", "BETA", "GAMA DIFERENTE", "DELTA"],
    })
    aud = setores.auditar_emissores(painel)
    alertas = dict(zip(aud.index, aud["alerta"]))
    assert alertas["CCCC3"] == "isin_ausente_ou_invalido"
    assert alertas["BBBB3"] == "nomes_divergentes_no_emissor"
    assert "AAAA3" not in alertas


# A regra so vale durante sobreposicao. Comparar por emissor sem olhar periodo
# proibiria a mudanca historica que a validade temporal existe para permitir.
def test_classes_simultaneas_com_mesma_classificacao_passam():
    t = _tabela(_linha("PETR3", emissor_id="PETR", setor="Petroleo",
                       subsetor="Exploracao", segmento="E&P"),
                _linha("PETR4", emissor_id="PETR", setor="Petroleo",
                       subsetor="Exploracao", segmento="E&P"))
    assert setores.validar(t) == []


def test_classes_simultaneas_com_setor_diferente_reprovam():
    t = _tabela(_linha("PETR3", emissor_id="PETR", setor="Petroleo"),
                _linha("PETR4", emissor_id="PETR", setor="Financeiro"))
    erros = setores.validar(t)
    assert any("simultaneos" in e and "setor" in e for e in erros)


def test_subsetor_divergente_em_simultaneas_reprova():
    """Mesmo setor nao basta: a chave de grupo e (setor, subsetor)."""
    t = _tabela(_linha("PETR3", emissor_id="PETR", subsetor="Exploracao"),
                _linha("PETR4", emissor_id="PETR", subsetor="Refino"))
    assert any("subsetor" in e for e in setores.validar(t))


def test_segmentos_divergentes_reprovam_quando_os_dois_estao_preenchidos():
    t = _tabela(_linha("PETR3", emissor_id="PETR", segmento="E&P"),
                _linha("PETR4", emissor_id="PETR", segmento="Distribuicao"))
    assert any("segmento" in e for e in setores.validar(t))


def test_segmento_vazio_nao_cria_divergencia_artificial():
    """
    Campo vazio significa "nao registrado", nao "diferente". Trata-lo como
    divergencia reprovaria a tabela por falta de informacao opcional.
    """
    t = _tabela(_linha("PETR3", emissor_id="PETR", segmento="E&P"),
                _linha("PETR4", emissor_id="PETR", segmento=""))
    assert setores.validar(t) == []


def test_emissor_pode_mudar_de_setor_em_periodos_nao_sobrepostos():
    """
    Mudanca documentada entre intervalos sucessivos e legitima: holding troca
    de atividade, fusao redefine o negocio. Proibir isso esvaziaria a validade
    temporal.
    """
    t = _tabela(
        _linha("XPTO3", emissor_id="XPTO", setor="Consumo",
               validade_inicio="2015-01-01", validade_fim="2019-12-31"),
        _linha("XPTO3", emissor_id="XPTO", setor="Tecnologia",
               validade_inicio="2020-01-01"))
    assert setores.validar(t) == []


def test_classes_diferentes_do_emissor_em_periodos_distintos_podem_divergir():
    """
    VIVT4 ate 2020 e VIVT3 depois nao sao simultaneos. Exigir a mesma
    classificacao entre eles impediria registrar mudanca real da empresa.
    """
    t = _tabela(
        _linha("VIVT4", emissor_id="VIVT", setor="Telecomunicacoes",
               validade_inicio="2015-01-01", validade_fim="2020-11-20"),
        _linha("VIVT3", emissor_id="VIVT", setor="Tecnologia",
               validade_inicio="2020-11-21"))
    assert setores.validar(t) == []


def test_inicio_posterior_ao_fim_reprova():
    t = _tabela(_linha(validade_inicio="2022-01-01", validade_fim="2021-01-01"))
    assert any("posterior" in e for e in setores.validar(t))


def test_ticker_observado_vazio_em_confirmado_reprova():
    t = _tabela(_linha(ticker=""))
    assert any("ticker_observado" in e for e in setores.validar(t))


def test_emissor_id_malformado_em_confirmado_reprova():
    for ruim in ("PET", "PETRO", "pe$r"):
        assert any("emissor_id" in e and "malformado" in e
                   for e in setores.validar(_tabela(_linha(emissor_id=ruim)))), ruim


def test_alerta_de_nomes_divergentes_nao_afirma_colisao():
    """
    BIDI alterna entre BANCO INTER e INTER BANCO: e grafia, nao colisao. O
    rotulo tem que dizer que e alerta de revisao.
    """
    painel = pd.DataFrame({
        "CODNEG": ["BIDI3", "BIDI4"],
        "CODISI": ["BRBIDIACNOR3", "BRBIDIACNPR0"],
        "NOMRES": ["BANCO INTER", "INTER BANCO"],
    })
    aud = setores.auditar_emissores(painel)
    assert set(aud["alerta"]) == {"nomes_divergentes_no_emissor"}
