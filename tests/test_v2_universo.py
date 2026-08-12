from __future__ import annotations

import pandas as pd
import pytest

from src.v2 import universo as v2


def _universo(janelas=("2024-01",), ticker_por_janela=None):
    ticker_por_janela = ticker_por_janela or {}
    linhas = []
    for janela in janelas:
        for posicao in range(1, 21):
            emissor = f"E{posicao:02d}"
            ticker = ticker_por_janela.get((janela, emissor), f"T{posicao:02d}3")
            linhas.append({
                "janela": janela,
                "CODNEG": ticker,
                "emissor_id": emissor,
                "posicao_final": float(posicao),
                "motivo_exclusao": "",
                "faixa": "top20",
                "setor": "Setor",
                "subsetor": "Subsetor",
                "data_formacao": f"{int(janela[:4]) - 1}-12-29",
            })
        linhas.extend([
            {
                "janela": janela,
                "CODNEG": "FORA3",
                "emissor_id": "FORA",
                "posicao_final": 21.0,
                "motivo_exclusao": "",
            },
            {
                "janela": janela,
                "CODNEG": "EXCL3",
                "emissor_id": "EXCL",
                "posicao_final": 10.0,
                "motivo_exclusao": "sem_classificacao_confirmada_na_formacao",
            },
        ])
    return pd.DataFrame(linhas)


def _par(faixa=20, lider="T013", seguidora="T023"):
    return {
        "janela": "2024-01",
        "lider": lider,
        "seguidora": seguidora,
        "emissor_lider": "E01",
        "emissor_seguidora": "E02",
        "faixa_minima": faixa,
    }


def _rede(faixa=20, beta=-0.2, lider="T013", seguidora="T023", **extras):
    linha = {
        "janela": "2024-01",
        "lider": lider,
        "seguidora": seguidora,
        "faixa_minima": faixa,
        "beta": beta,
    }
    linha.update(extras)
    return linha


def test_universo_top20_filtra_posicao_e_motivo_sem_reordenar_emissores():
    top20 = v2.selecionar_universo_top20(_universo())

    assert len(top20) == 20
    assert top20["posicao_final"].tolist() == list(range(1, 21))
    assert top20["emissor_id"].nunique() == 20
    assert "FORA3" not in set(top20["CODNEG"])
    assert "EXCL3" not in set(top20["CODNEG"])


def test_universo_top20_exige_vinte_ativos_em_toda_janela():
    incompleto = _universo(("2024-01", "2024-04"))
    incompleto = incompleto[
        ~((incompleto["janela"] == "2024-04") & (incompleto["CODNEG"] == "T203"))
    ]

    with pytest.raises(ValueError, match="2024-04=19"):
        v2.selecionar_universo_top20(incompleto)


def test_universo_top20_recusa_duas_classes_do_mesmo_emissor():
    universo = _universo()
    universo.loc[universo["CODNEG"] == "T203", "emissor_id"] = "E01"

    with pytest.raises(ValueError, match="mais de uma classe"):
        v2.selecionar_universo_top20(universo)


def test_carregar_universo_top20_le_csv_sem_depender_de_output_real(tmp_path):
    caminho = tmp_path / "universo_por_janela.csv"
    _universo().to_csv(caminho, index=False)

    top20 = v2.carregar_universo_top20(caminho)

    assert len(top20) == 20
    assert top20.iloc[0]["CODNEG"] == "T013"


def test_pares_e_rede_filtram_faixa_sem_selecionar_resultado():
    pares = pd.DataFrame([_par(), _par(40, "T033", "T043")])
    rede = pd.DataFrame([
        _rede(beta=-0.2, p_valor=0.99, aprovado_fdr=False, lag=1),
        _rede(40, 0.8, "T033", "T043", p_valor=0.001, aprovado_fdr=True, lag=1),
    ])

    junto = v2.juntar_pares_com_betas_top20(pares, rede)

    assert len(junto) == 1
    assert junto.iloc[0]["beta"] == pytest.approx(-0.2)
    assert junto.iloc[0]["p_valor"] == pytest.approx(0.99)
    assert not bool(junto.iloc[0]["aprovado_fdr"])


def test_juncao_recusa_par_sem_beta_correspondente():
    pares = pd.DataFrame([_par()])
    rede = pd.DataFrame([_rede(lider="T033", seguidora="T043")])

    with pytest.raises(ValueError, match="sem beta"):
        v2.juntar_pares_com_betas_top20(pares, rede)


def test_rede_recusa_mais_de_um_beta_para_a_mesma_aresta():
    rede = pd.DataFrame([_rede(beta=0.1), _rede(beta=0.2)])

    with pytest.raises(ValueError, match="mais de um beta"):
        v2.selecionar_rede_top20(rede)


def test_evento_e_mapeado_por_emissor_e_janela_nao_por_ticker_atual():
    tickers = {
        ("2024-01", "E01"): "ANTG3",
        ("2024-04", "E01"): "NOVO3",
    }
    universo = _universo(("2024-01", "2024-04"), tickers)
    eventos = pd.DataFrame([
        {"evento_id": 1, "janela": "2024-01", "emissor_id": "E01",
         "ticker_atual": "NOVO3"},
        {"evento_id": 2, "janela": "2024-04", "emissor_id": "E01",
         "ticker_atual": "NOVO3"},
    ])

    mapeados = v2.mapear_eventos_por_emissor_e_janela(eventos, universo)

    assert mapeados["CODNEG"].tolist() == ["ANTG3", "NOVO3"]
    assert mapeados["ticker_atual"].tolist() == ["NOVO3", "NOVO3"]


def test_evento_fora_da_top20_pode_ser_descartado_ou_tratado_como_erro():
    eventos = pd.DataFrame([
        {"janela": "2024-01", "emissor_id": "E01"},
        {"janela": "2024-01", "emissor_id": "FORA"},
    ])

    mapeados = v2.mapear_eventos_por_emissor_e_janela(eventos, _universo())
    assert mapeados["emissor_id"].tolist() == ["E01"]

    with pytest.raises(ValueError, match="FORA"):
        v2.mapear_eventos_por_emissor_e_janela(
            eventos, _universo(), exigir_correspondencia=True
        )


def test_evento_nao_pode_fornecer_codneg_corrente():
    eventos = pd.DataFrame([
        {"janela": "2024-01", "emissor_id": "E01", "CODNEG": "ATUAL3"}
    ])

    with pytest.raises(ValueError, match="ticker PIT"):
        v2.mapear_eventos_por_emissor_e_janela(eventos, _universo())
