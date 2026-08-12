from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import Janela
from src.v2 import eventos


def _janela(indice: int, inicio: str, fim: str) -> Janela:
    teste_inicio = pd.Timestamp(inicio)
    return Janela(
        indice=indice,
        treino_inicio=teste_inicio - pd.DateOffset(years=2),
        treino_fim=teste_inicio,
        teste_inicio=teste_inicio,
        teste_fim=pd.Timestamp(fim),
    )


def _par(**alteracoes) -> dict[str, object]:
    base = {
        "janela": "2026-01",
        "lider": "LID3",
        "seguidora": "SEG3",
        "emissor_lider": "EMISSOR-L",
        "emissor_seguidora": "EMISSOR-S",
        "faixa_minima": 20,
    }
    base.update(alteracoes)
    return base


def test_seleciona_ap_original_vazio_ou_v1_sem_promover_revisao():
    documentos = pd.DataFrame(
        {
            "id": [
                "original-v1",
                "original-vazio",
                "revisao-ap",
                "reapresentacao",
                "reapresentacao-rc",
                "codigo-incompleto",
            ],
            "Tipo_Apresentacao": [
                "AP - Apresentação",
                "AP - Apresentação",
                "AP - Apresentação",
                "RE - Reapresentação",
                "RC - Reapresentação por exigência",
                "AP",
            ],
            "Versao": ["1", "", "4", "1", "1", "1"],
        }
    )

    originais = eventos.selecionar_apresentacoes_originais(documentos)

    assert originais["id"].tolist() == ["original-v1", "original-vazio"]
    assert documentos["id"].tolist()[0] == "original-v1"


def test_sessao_e_estritamente_posterior_e_fim_do_calendario_e_diagnosticado():
    base = pd.DataFrame(
        {
            "id": ["sexta", "sabado", "segunda", "sem-futuro"],
            "Data_Entrega": [
                "2026-01-09",
                "2026-01-10",
                "2026-01-12",
                "2026-01-13",
            ],
        }
    )
    calendario = pd.DatetimeIndex(["2026-01-09", "2026-01-12", "2026-01-13"])

    resultado = eventos.mapear_sessoes_disponiveis(base, calendario)

    assert resultado.eventos["id"].tolist() == ["sexta", "sabado", "segunda"]
    assert resultado.eventos["Sessao_Disponivel"].tolist() == [
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-01-13"),
    ]
    entregas = pd.to_datetime(resultado.eventos["Data_Entrega"])
    assert (resultado.eventos["Sessao_Disponivel"] > entregas).all()
    assert resultado.diagnosticos["id"].tolist() == ["sem-futuro"]
    assert resultado.diagnosticos["Sessao_Disponivel"].isna().all()
    assert resultado.diagnosticos["motivo_diagnostico"].tolist() == [
        eventos.SEM_SESSAO_POSTERIOR
    ]


def test_rotulo_de_janela_duplicado_e_recusado():
    base = pd.DataFrame({"Sessao_Disponivel": [pd.Timestamp("2026-01-12")]})
    janelas = [
        _janela(0, "2026-01-12", "2026-01-14"),
        _janela(1, "2026-01-14", "2026-01-16"),
    ]

    with pytest.raises(ValueError, match="rotulo duplicado"):
        eventos.atribuir_janelas_teste(base, janelas)


def test_fronteira_vai_para_janela_seguinte_e_fora_vira_diagnostico():
    base = pd.DataFrame(
        {
            "id": ["j0", "fronteira", "fora"],
            "Sessao_Disponivel": pd.to_datetime(
                ["2026-01-31", "2026-02-01", "2026-03-01"]
            ),
        }
    )
    janelas = [
        _janela(0, "2026-01-01", "2026-02-01"),
        _janela(1, "2026-02-01", "2026-03-01"),
    ]

    resultado = eventos.atribuir_janelas_teste(base, janelas)

    assert resultado.eventos[["id", "janela"]].values.tolist() == [
        ["j0", "2026-01"],
        ["fronteira", "2026-02"],
    ]
    assert resultado.diagnosticos["id"].tolist() == ["fora"]
    assert resultado.diagnosticos["motivo_diagnostico"].tolist() == [
        eventos.FORA_DAS_JANELAS
    ]


def test_filtra_lideres_top20_sem_multiplicar_evento_por_seguidora():
    pares = pd.DataFrame(
        [
            _par(),
            _par(seguidora="OUT4", emissor_seguidora="EMISSOR-O"),
            _par(
                lider="FOR3",
                seguidora="FOR4",
                emissor_lider="EMISSOR-FORA",
                emissor_seguidora="EMISSOR-FORA-S",
                faixa_minima=40,
            ),
        ]
    )
    base = pd.DataFrame(
        {
            "id": ["lider", "seguidora", "fora-top20", "janela-errada"],
            "janela": ["2026-01", "2026-01", "2026-01", "2026-04"],
            "emissor_id": [
                "EMISSOR-L",
                "EMISSOR-S",
                "EMISSOR-FORA",
                "EMISSOR-L",
            ],
            # O ticker corrente nao participa da associacao PIT.
            "ticker_atual": ["FUTU3", "LID3", "FOR3", "LID3"],
        }
    )

    resultado = eventos.filtrar_eventos_lideres_top20(base, pares)

    assert resultado.eventos[["id", "lider"]].values.tolist() == [
        ["lider", "LID3"]
    ]
    assert resultado.diagnosticos["id"].tolist() == [
        "seguidora",
        "fora-top20",
        "janela-errada",
    ]
    assert set(resultado.diagnosticos["motivo_diagnostico"]) == {
        eventos.NAO_LIDER_TOP20
    }


def test_agrega_concordancia_neutralidade_e_conflito_preservando_contagens():
    classificados = pd.DataFrame(
        {
            "janela": ["2026-01"] * 9,
            "lider": ["POS3"] * 3 + ["NEG3"] * 2 + ["NEU3"] * 2 + ["CON3"] * 2,
            "Data_Entrega": ["2026-01-09"] * 9,
            "Sessao_Disponivel": [pd.Timestamp("2026-01-12")] * 9,
            "classificacao": ["positiva", 1, "neutra", -1, 0, "neutra", 0, 1, -1],
        }
    )

    sinais = eventos.agregar_sinais_eventos(classificados).set_index("lider")

    colunas_completas = [
        "n_documentos",
        "n_positivas",
        "n_negativas",
        "n_neutras",
        "sinal",
        "abstencao",
        "motivo",
    ]
    colunas_sinal = [
        "n_positivas",
        "n_negativas",
        "n_neutras",
        "sinal",
        "abstencao",
        "motivo",
    ]
    assert sinais.loc["POS3", colunas_completas].tolist() == [
        3, 2, 0, 1, 1, False, "positivas_concordantes"
    ]
    assert sinais.loc["NEG3", colunas_completas].tolist() == [
        2, 0, 1, 1, -1, False, "negativas_concordantes"
    ]
    assert sinais.loc["NEU3", colunas_sinal].tolist() == [
        0, 0, 2, 0, True, "somente_neutras"
    ]
    assert sinais.loc["CON3", colunas_sinal].tolist() == [
        1, 1, 0, 0, True, "conflito_positivo_negativo"
    ]
    assert (sinais["Sessao_Disponivel"] == pd.Timestamp("2026-01-12")).all()


def test_agregacao_recusa_sessoes_divergentes_no_mesmo_lider_dia():
    base = pd.DataFrame(
        {
            "lider": ["LID3", "LID3"],
            "Data_Entrega": ["2026-01-09", "2026-01-09"],
            "Sessao_Disponivel": pd.to_datetime(["2026-01-12", "2026-01-13"]),
            "classificacao": ["positiva", "positiva"],
        }
    )

    with pytest.raises(ValueError, match="deve ser único"):
        eventos.agregar_sinais_eventos(base)


def test_agregacao_aceita_sessao_disponivel_e_rejeita_classe_desconhecida():
    base = pd.DataFrame(
        {
            "lider": ["LID3"],
            "Sessao_Disponivel": [pd.Timestamp("2026-01-12")],
            "rotulo_modelo": ["incerta"],
        }
    )

    with pytest.raises(ValueError, match="classificacao invalida"):
        eventos.agregar_sinais_eventos(
            base,
            coluna_classificacao="rotulo_modelo",
            coluna_data="Sessao_Disponivel",
        )
