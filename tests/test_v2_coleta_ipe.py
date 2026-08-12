from __future__ import annotations

import pandas as pd
import pytest

from scripts import v2_01_coletar_ipe as coleta


def _referencia(**alteracoes: object) -> pd.DataFrame:
    linha = {
        "emissor_id": "PETR",
        "codigo_cvm": "09512",
        "cnpj": "33.000.167/0001-01",
        "denominacao_cvm": "PETRÓLEO BRASILEIRO S.A. - PETROBRAS",
        "validade_inicio": "1953-10-03",
        "validade_fim": "",
        "status_revisao": "confirmado",
    }
    linha.update(alteracoes)
    return pd.DataFrame([linha])


def _fato(**alteracoes: object) -> pd.DataFrame:
    linha = {
        "Codigo_CVM": "009512",
        "CNPJ_Companhia": "33.000.167/0001-01",
        "Data_Entrega": "2024-05-10",
        "ID_Documento": "doc-1",
    }
    linha.update(alteracoes)
    return pd.DataFrame([linha])


def test_carregar_lideres_normaliza_codigo_e_datas(tmp_path):
    caminho = tmp_path / "lideres.csv"
    _referencia().to_csv(caminho, index=False)

    resultado = coleta._carregar_lideres(caminho)

    assert resultado.loc[0, "codigo_cvm"] == "9512"
    assert resultado.loc[0, "validade_inicio"] == pd.Timestamp("1953-10-03")
    assert pd.isna(resultado.loc[0, "validade_fim"])


def test_carregar_lideres_recusa_referencia_nao_confirmada(tmp_path):
    caminho = tmp_path / "lideres.csv"
    _referencia(status_revisao="pendente").to_csv(caminho, index=False)

    with pytest.raises(ValueError, match="confirmada"):
        coleta._carregar_lideres(caminho)


def test_carregar_lideres_recusa_emissor_duplicado(tmp_path):
    caminho = tmp_path / "lideres.csv"
    referencia = pd.concat([_referencia(), _referencia()], ignore_index=True)
    referencia.to_csv(caminho, index=False)

    with pytest.raises(ValueError, match="emissor_id duplicado"):
        coleta._carregar_lideres(caminho)


def test_carregar_lideres_recusa_chave_vazia_e_intervalo_invertido(tmp_path):
    caminho = tmp_path / "lideres.csv"
    _referencia(cnpj="").to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="cnpj vazio"):
        coleta._carregar_lideres(caminho)

    _referencia(validade_inicio="2024-01-02", validade_fim="2024-01-01").to_csv(
        caminho, index=False
    )
    with pytest.raises(ValueError, match="anterior"):
        coleta._carregar_lideres(caminho)


def test_juntar_lideres_exige_codigo_e_cnpj_concordantes(tmp_path):
    caminho = tmp_path / "lideres.csv"
    _referencia().to_csv(caminho, index=False)
    lideres = coleta._carregar_lideres(caminho)

    resultado = coleta._juntar_lideres(_fato(), lideres)
    assert resultado.loc[0, "emissor_id"] == "PETR"

    com_cnpj_errado = _fato(CNPJ_Companhia="00.000.000/0001-00")
    with pytest.raises(ValueError, match="CNPJ diferente"):
        coleta._juntar_lideres(com_cnpj_errado, lideres)


def test_juntar_lideres_respeita_intervalo_de_validade(tmp_path):
    caminho = tmp_path / "lideres.csv"
    _referencia(validade_inicio="2020-01-01", validade_fim="2024-01-01").to_csv(
        caminho, index=False
    )
    lideres = coleta._carregar_lideres(caminho)

    fora = coleta._juntar_lideres(_fato(Data_Entrega="2024-05-10"), lideres)
    dentro = coleta._juntar_lideres(_fato(Data_Entrega="2023-05-10"), lideres)

    assert fora.empty
    assert dentro["ID_Documento"].tolist() == ["doc-1"]
