from __future__ import annotations

import math

import pandas as pd
import pytest

from src.v2 import validacao_ia


def _classificacoes() -> pd.DataFrame:
    linhas = []
    numero = 0
    for classe in validacao_ia.CLASSES_CONJUNTAS:
        especifico = classe != "nao_especifico"
        direcao = (
            classe.removeprefix("especifico_") if especifico else "neutra"
        )
        for repeticao in range(5):
            numero += 1
            linhas.append(
                {
                    "ID_Documento": f"doc-{numero:02d}",
                    "Data_Entrega": f"{2020 + repeticao}-01-10",
                    "emissor_id": f"E{repeticao}",
                    "Nome_Companhia": f"Companhia {repeticao}",
                    "texto_llm": f"Texto publico {numero}",
                    "direcao": direcao,
                    "especifico_empresa": especifico,
                    "evidencia": "trecho do modelo",
                    "abster": not especifico or direcao == "neutra",
                    "motivo_abstencao": "neutro" if direcao == "neutra" else "",
                    "prompt_hash": "segredo",
                    "modelo_digest": "segredo",
                    "preco_futuro": 100 + numero,
                    "pnl": numero / 100,
                    "retorno_5d": numero / 1000,
                }
            )
    return pd.DataFrame(linhas)


def _rotulos(registros: list[tuple[str, bool, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        registros,
        columns=["ID_Documento", "especifico_empresa", "direcao"],
    )


def _predicoes(registros: list[tuple[str, bool, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        registros,
        columns=["ID_Documento", "especifico_empresa", "direcao", "abster"],
    )


def test_amostra_e_deterministica_estratificada_e_independe_da_ordem():
    base = _classificacoes()

    primeira = validacao_ia.selecionar_amostra_cega(base, 8, seed=42)
    segunda = validacao_ia.selecionar_amostra_cega(base, 8, seed=42)
    embaralhada = validacao_ia.selecionar_amostra_cega(
        base.sample(frac=1, random_state=99), 8, seed=42
    )

    pd.testing.assert_frame_equal(primeira, segunda)
    pd.testing.assert_frame_equal(primeira, embaralhada)
    por_id = base.set_index("ID_Documento")
    selecionadas = por_id.loc[primeira["ID_Documento"]]
    assert selecionadas["direcao"].value_counts().to_dict() == {
        "positiva": 3,
        "negativa": 3,
        "neutra": 2,
    }
    neutras = selecionadas.loc[selecionadas["direcao"].eq("neutra")]
    assert (~neutras["especifico_empresa"]).all()


def test_amostra_entregue_cega_ia_preco_e_pnl_mas_preserva_contexto():
    amostra = validacao_ia.selecionar_amostra_cega(
        _classificacoes(), 6, seed=7
    )

    proibidas = {
        "direcao",
        "especifico_empresa",
        "evidencia",
        "abster",
        "motivo_abstencao",
        "prompt_hash",
        "modelo_digest",
        "preco_futuro",
        "pnl",
        "retorno_5d",
    }
    assert proibidas.isdisjoint(amostra.columns)
    assert {
        "ID_Documento",
        "Data_Entrega",
        "emissor_id",
        "Nome_Companhia",
        "texto_llm",
    }.issubset(amostra.columns)
    assert not any(coluna.startswith("_") for coluna in amostra.columns)


def test_amostra_balanceada_recusa_direcao_sem_casos_suficientes():
    base = _classificacoes()
    positivo = base.loc[base["direcao"].eq("positiva"), "ID_Documento"].iloc[0]
    negativos = base.loc[
        base["direcao"].eq("negativa"), "ID_Documento"
    ].head(2)
    manter = (
        ((base["direcao"] == "positiva") & base["ID_Documento"].eq(positivo))
        | (
            (base["direcao"] == "negativa")
            & base["ID_Documento"].isin(negativos)
        )
        | base["direcao"].eq("neutra")
    )
    base = base.loc[manter].copy()

    with pytest.raises(ValueError, match="direcoes insuficientes"):
        validacao_ia.selecionar_amostra_cega(base, 7, seed=11)


def test_rotulos_sao_alinhados_por_id_e_invariantes_sao_estritas():
    a = _rotulos(
        [("d2", True, "negativa"), ("d1", False, "neutra")]
    )
    b = _rotulos(
        [("d1", False, "neutra"), ("d2", True, "negativa")]
    )

    validados = validacao_ia.validar_rotulos_humanos(
        a, b, ids_esperados=["d1", "d2"]
    )

    assert validados["ID_Documento"].tolist() == ["d1", "d2"]
    assert validados["classe_a"].tolist() == [
        "nao_especifico",
        "especifico_negativa",
    ]

    invalido = _rotulos([("d1", False, "positiva")])
    with pytest.raises(ValueError, match="nao especifico"):
        validacao_ia.validar_rotulos_humanos(invalido, invalido)

    com_pnl = a.assign(pnl=[0.1, 0.2])
    with pytest.raises(ValueError, match="colunas devem ser somente"):
        validacao_ia.validar_rotulos_humanos(com_pnl, b)


def test_kappa_perfeito_desacordo_maximo_e_classe_constante_indefinida():
    perfeito = validacao_ia.validar_rotulos_humanos(
        _rotulos([("d1", True, "positiva"), ("d2", True, "negativa")]),
        _rotulos([("d1", True, "positiva"), ("d2", True, "negativa")]),
    )
    opostos = validacao_ia.validar_rotulos_humanos(
        _rotulos([("d1", True, "positiva"), ("d2", True, "negativa")]),
        _rotulos([("d1", True, "negativa"), ("d2", True, "positiva")]),
    )
    constante = validacao_ia.validar_rotulos_humanos(
        _rotulos([("d1", False, "neutra"), ("d2", False, "neutra")]),
        _rotulos([("d1", False, "neutra"), ("d2", False, "neutra")]),
    )

    assert validacao_ia.calcular_cohen_kappa(perfeito) == pytest.approx(1.0)
    assert validacao_ia.calcular_cohen_kappa(opostos) == pytest.approx(-1.0)
    assert math.isnan(validacao_ia.calcular_cohen_kappa(constante))


def test_divergencias_criam_fila_de_adjudicacao_sem_ia():
    validados = validacao_ia.validar_rotulos_humanos(
        _rotulos([("d1", True, "positiva"), ("d2", False, "neutra")]),
        _rotulos([("d1", True, "negativa"), ("d2", False, "neutra")]),
    )

    divergencias = validacao_ia.criar_tabela_divergencias(validados)

    assert divergencias["ID_Documento"].tolist() == ["d1"]
    assert divergencias["gold_especifico_empresa"].isna().all()
    assert divergencias["gold_direcao"].isna().all()
    assert not any("ia" in coluna.casefold() for coluna in divergencias.columns)


def test_neutralidade_abstida_e_avaliada_e_cobertura_fica_separada():
    gold = _rotulos(
        [
            ("p", True, "positiva"),
            ("n", True, "negativa"),
            ("e", True, "neutra"),
            ("x", False, "neutra"),
        ]
    )
    ia = _predicoes(
        [
            ("p", True, "positiva", False),
            ("n", True, "negativa", False),
            ("e", True, "neutra", True),
            ("x", False, "neutra", True),
        ]
    )

    resultado = validacao_ia.avaliar_ia_contra_gold(ia, gold, 0.80)

    assert resultado.matriz_confusao.at[
        "especifico_neutra", "especifico_neutra"
    ] == 1
    assert resultado.matriz_confusao.at[
        "nao_especifico", "nao_especifico"
    ] == 1
    assert resultado.macro_f1 == pytest.approx(1.0)
    assert resultado.cobertura == pytest.approx(0.5)
    assert resultado.taxa_abstencao == pytest.approx(0.5)
    assert resultado.aprovado_kappa is True
    assert resultado.aprovado_macro_f1 is True
    assert resultado.aprovado is True


def test_abstencao_em_gold_positivo_e_falso_negativo():
    gold = _rotulos(
        [("p", True, "positiva"), ("n", True, "negativa")]
    )
    ia = _predicoes(
        [
            ("p", True, "neutra", True),
            ("n", True, "negativa", False),
        ]
    )

    resultado = validacao_ia.avaliar_ia_contra_gold(ia, gold, 0.80)
    metricas = resultado.metricas_por_classe.set_index("classe")

    assert resultado.matriz_confusao.at[
        "especifico_positiva", "especifico_neutra"
    ] == 1
    assert metricas.at["especifico_positiva", "fn"] == 1
    assert metricas.at["especifico_positiva", "f1"] == 0.0
    assert resultado.macro_f1 == pytest.approx(0.25)
    assert resultado.aprovado is False


def test_classes_ausentes_reprovam_macro_f1():
    gold = _rotulos(
        [("p1", True, "positiva"), ("p2", True, "positiva")]
    )
    ia = _predicoes(
        [
            ("p1", True, "positiva", False),
            ("p2", True, "positiva", False),
        ]
    )

    aprovado = validacao_ia.avaliar_ia_contra_gold(ia, gold, 0.60)
    reprovado = validacao_ia.avaliar_ia_contra_gold(ia, gold, 0.59)

    assert aprovado.metricas_por_classe["classe"].tolist() == list(
        validacao_ia.CLASSES_CONJUNTAS
    )
    ausentes = aprovado.metricas_por_classe.query("suporte == 0")
    assert (ausentes[["precisao", "recall", "f1"]] == 0.0).all().all()
    assert aprovado.macro_f1 == pytest.approx(0.25)
    assert aprovado.cobertura == pytest.approx(1.0)
    assert aprovado.aprovado_macro_f1 is False
    assert aprovado.aprovado is False
    assert reprovado.aprovado_macro_f1 is False
    assert reprovado.aprovado_kappa is False
    assert reprovado.aprovado is False


def test_classe_sem_predicao_reprova_gate():
    gold = _rotulos(
        [
            ("x", False, "neutra"),
            ("p", True, "positiva"),
            ("n", True, "negativa"),
            ("e", True, "neutra"),
        ]
    )
    ia = _predicoes(
        [
            ("x", True, "neutra", True),
            ("p", True, "positiva", False),
            ("n", True, "negativa", False),
            ("e", True, "neutra", True),
        ]
    )

    resultado = validacao_ia.avaliar_ia_contra_gold(ia, gold, 1.0)

    assert resultado.metricas_por_classe.set_index("classe").at[
        "nao_especifico", "preditos"
    ] == 0
    assert resultado.aprovado_macro_f1 is False
    assert resultado.aprovado is False


def test_avaliacao_recusa_ids_e_invariantes_de_abstencao_invalidos():
    gold = _rotulos([("d1", True, "positiva")])
    abstencao_positiva = _predicoes([("d1", True, "positiva", True)])
    with pytest.raises(ValueError, match="invariantes"):
        validacao_ia.avaliar_ia_contra_gold(abstencao_positiva, gold, 0.8)

    outro_id = _predicoes([("d2", True, "positiva", False)])
    with pytest.raises(ValueError, match="mesmos IDs"):
        validacao_ia.avaliar_ia_contra_gold(outro_id, gold, 0.8)
