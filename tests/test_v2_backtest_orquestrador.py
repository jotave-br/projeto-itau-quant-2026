from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_SCRIPT = RAIZ / "scripts" / "v2_07_backtest_eventos.py"
SPEC = importlib.util.spec_from_file_location("v2_07_backtest_eventos", CAMINHO_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_pares_estruturais_filtra_estritamente_top20(tmp_path: Path):
    rede = pd.DataFrame(
        {
            "janela": ["2026-01", "2026-01"],
            "lider": ["L1", "L1"],
            "seguidora": ["A1", "B1"],
            "faixa_minima": [20, 40],
            "beta": [0.2, 0.3],
            "direcao": ["lider_para_seguidora", "lider_para_seguidora"],
            "setor": ["Financeiro", "Financeiro"],
            "subsetor": ["Bancos", "Bancos"],
        }
    )
    caminho = tmp_path / "rede.csv"
    rede.to_csv(caminho, index=False)

    pares = SCRIPT._pares_estruturais(caminho)

    assert pares[["lider", "seguidora"]].to_records(index=False).tolist() == [
        ("L1", "A1")
    ]


def test_placebo_preserva_contagem_setor_unicidade_e_remove_self_pairs():
    pares = pd.DataFrame(
        {
            "janela": ["2026-01", "2026-01"],
            "lider": ["L1", "L1"],
            "seguidora": ["A1", "B1"],
            "setor": ["Financeiro", "Financeiro"],
            "subsetor": ["Bancos", "Bancos"],
        }
    )
    universo = pd.DataFrame(
        {
            "janela": ["2026-01"] * 4,
            "CODNEG": ["L1", "A1", "B1", "C1"],
            "setor": ["Financeiro"] * 4,
        }
    )

    placebo = SCRIPT._pares_placebo(pares, universo, seed=7)

    assert len(placebo) == len(pares)
    assert not placebo.duplicated().any()
    assert not placebo["lider"].eq(placebo["seguidora"]).any()
    assert set(placebo["seguidora"]) <= {"A1", "B1", "C1"}
    pd.testing.assert_frame_equal(
        placebo,
        SCRIPT._pares_placebo(pares, universo, seed=7),
    )


def test_rede_sem_ia_usa_retorno_da_lider_e_execucao_na_sessao_seguinte():
    calendario = pd.bdate_range("2026-01-05", periods=5)
    pares = pd.DataFrame(
        {
            "janela": ["2026-01"],
            "lider": ["L1"],
            "seguidora": ["A1"],
            "setor": ["Financeiro"],
            "subsetor": ["Bancos"],
        }
    )
    painel = pd.DataFrame(
        {"L1": [float("nan"), 0.02, -0.01, 0.0, 0.03]},
        index=calendario,
    )

    sinais = SCRIPT._sinais_rede_sem_ia(pares, painel, calendario)

    assert sinais["Data_Entrega"].tolist() == [calendario[1], calendario[2]]
    assert sinais["Sessao_Disponivel"].tolist() == [calendario[2], calendario[3]]
    assert sinais["direcao"].tolist() == [1, -1]
    assert sinais["Data_Entrega"].lt(sinais["Sessao_Disponivel"]).all()
