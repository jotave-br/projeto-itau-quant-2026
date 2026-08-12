from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from scripts import v2_04_preparar_validacao_humana as preparar
from src.v2 import ia_eventos, validacao_ia


def _dados(repeticoes: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = []
    classificacoes = []
    numero = 0
    for especifico, direcao in (
        (False, "neutra"),
        (True, "positiva"),
        (True, "negativa"),
        (True, "neutra"),
    ):
        for repeticao in range(repeticoes):
            numero += 1
            documento = f"doc-{numero:02d}"
            texto = f"Texto publico do documento {numero}"
            corpus.append(
                {
                    "ID_Documento": documento,
                    "Data_Entrega": f"{2020 + repeticao}-03-10",
                    "emissor_id": f"E{repeticao:02d}",
                    "texto_llm": texto,
                    "status_documento": "ok",
                    "seguidora": "SEG3",
                    "preco_futuro": 100 + numero,
                    "retorno_5d": numero / 100,
                    "pnl": numero / 10,
                }
            )
            classificacoes.append(
                {
                    "ID_Documento": documento,
                    "status_ia": "ok",
                    "erro_ia": "",
                    "especifico_empresa": especifico,
                    "direcao": direcao,
                    "abster": not especifico or direcao == "neutra",
                    "evidencia": "trecho interno",
                    "prompt_versao": ia_eventos.PROMPT_VERSAO,
                    "prompt_hash": ia_eventos.HASH_PROMPT,
                    "alvo_direcao": ia_eventos.ALVO_DIRECAO,
                    "modelo": "qwen3:14b",
                    "modelo_digest": preparar.DIGEST_MODELO_CONGELADO,
                    "modelo_quantizacao": preparar.QUANTIZACAO_CONGELADA,
                    "ollama_versao": "0.32.9",
                    "texto_hash": ia_eventos.identidade_requisicao(texto).hash_texto,
                }
            )
    return pd.DataFrame(corpus), pd.DataFrame(classificacoes)


def test_planilhas_sao_cegas_estratificadas_e_tem_ordens_independentes():
    corpus, classificacoes = _dados()

    resultado = preparar.preparar_validacao(corpus, classificacoes, tamanho=8)

    assert tuple(resultado.avaliador_a.columns) == preparar.COLUNAS_PLANILHA
    assert tuple(resultado.avaliador_b.columns) == preparar.COLUNAS_PLANILHA
    assert resultado.avaliador_a["especifico_empresa"].eq("").all()
    assert resultado.avaliador_a["direcao"].eq("").all()
    assert resultado.avaliador_b["especifico_empresa"].eq("").all()
    assert resultado.avaliador_b["direcao"].eq("").all()
    assert set(resultado.avaliador_a["id_anonimo"]) == set(
        resultado.avaliador_b["id_anonimo"]
    )
    assert resultado.avaliador_a["id_anonimo"].tolist() != (
        resultado.avaliador_b["id_anonimo"].tolist()
    )
    chave = resultado.chave_interna
    assert chave["direcao_ia"].value_counts().to_dict() == {
        "positiva": 3,
        "negativa": 3,
        "neutra": 2,
    }
    assert chave.loc[chave["direcao_ia"].eq("neutra"), "classe_ia"].eq(
        "nao_especifico"
    ).all()
    assert resultado.chave_interna["id_anonimo"].str.fullmatch(
        r"VH-[0-9A-F]{12}"
    ).all()
    protocolo_hash = hashlib.sha256(
        preparar.PROTOCOLO_ROTULAGEM.read_bytes()
    ).hexdigest()
    for coluna, valor in {
        "prompt_versao": ia_eventos.PROMPT_VERSAO,
        "prompt_hash": ia_eventos.HASH_PROMPT,
        "alvo_direcao": ia_eventos.ALVO_DIRECAO,
        "modelo": "qwen3:14b",
        "modelo_digest": preparar.DIGEST_MODELO_CONGELADO,
        "modelo_quantizacao": preparar.QUANTIZACAO_CONGELADA,
        "ollama_versao": "0.32.9",
        "tamanho_amostra": 8,
        "seed_amostra": preparar.SEED_VALIDACAO,
        "protocolo_rotulagem_hash": protocolo_hash,
    }.items():
        assert resultado.chave_interna[coluna].eq(valor).all()
    assert resultado.chave_interna["lote_hash"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    resumo = resultado.resumo_interno.iloc[0]
    assert resumo["documentos_elegiveis"] == 24
    assert resumo["classificacoes_ok"] == 24
    assert resumo["erros_tecnicos"] == 0
    assert resumo["taxa_erro_tecnico"] == 0
    assert resumo["abstencoes"] == 12
    assert resumo["taxa_abstencao"] == pytest.approx(0.5)


def test_saida_independe_da_ordem_das_entradas_e_nao_vaza_metadados():
    corpus, classificacoes = _dados()

    primeira = preparar.preparar_validacao(corpus, classificacoes, tamanho=12)
    segunda = preparar.preparar_validacao(
        corpus.sample(frac=1, random_state=7),
        classificacoes.sample(frac=1, random_state=8),
        tamanho=12,
    )

    pd.testing.assert_frame_equal(primeira.avaliador_a, segunda.avaliador_a)
    pd.testing.assert_frame_equal(primeira.avaliador_b, segunda.avaliador_b)
    pd.testing.assert_frame_equal(primeira.chave_interna, segunda.chave_interna)
    nomes = " ".join(primeira.avaliador_a.columns).casefold()
    for proibido in ("ia", "preco", "retorno", "seguidora", "pnl", "p&l"):
        assert proibido not in nomes
    ids_reais = set(corpus["ID_Documento"])
    assert ids_reais.isdisjoint(primeira.avaliador_a["id_anonimo"])
    assert set(primeira.chave_interna["ID_Documento"]).issubset(ids_reais)


def test_classificacoes_com_erro_nao_entram_na_amostra():
    corpus, classificacoes = _dados()
    classificacoes.loc[0, "status_ia"] = "erro"
    classificacoes.loc[0, "erro_ia"] = "TimeoutError: expirou"
    for coluna in (
        "especifico_empresa",
        "direcao",
        "abster",
        "prompt_versao",
        "prompt_hash",
        "alvo_direcao",
        "modelo",
        "modelo_digest",
        "modelo_quantizacao",
        "ollama_versao",
        "texto_hash",
    ):
        classificacoes[coluna] = classificacoes[coluna].astype(object)
        classificacoes.at[0, coluna] = pd.NA

    resultado = preparar.preparar_validacao(
        corpus, classificacoes, tamanho=15
    )

    assert "doc-01" not in set(resultado.chave_interna["ID_Documento"])
    resumo = resultado.resumo_interno.iloc[0]
    assert resumo["documentos_elegiveis"] == 24
    assert resumo["classificacoes_ok"] == 23
    assert resumo["erros_tecnicos"] == 1
    assert resumo["taxa_erro_tecnico"] == pytest.approx(1 / 24)


@pytest.mark.parametrize("mutacao", ["faltante", "extra", "duplicada"])
def test_lote_exige_cobertura_exata_dos_elegiveis(mutacao):
    corpus, classificacoes = _dados()
    if mutacao == "faltante":
        classificacoes = classificacoes.iloc[1:].copy()
    elif mutacao == "extra":
        extra = classificacoes.iloc[[0]].copy()
        extra["ID_Documento"] = "doc-extra"
        classificacoes = pd.concat([classificacoes, extra], ignore_index=True)
    else:
        classificacoes = pd.concat(
            [classificacoes, classificacoes.iloc[[0]]], ignore_index=True
        )

    with pytest.raises(ValueError, match="duplicado|cobrem exatamente"):
        preparar.preparar_validacao(corpus, classificacoes, tamanho=8)


@pytest.mark.parametrize(
    ("coluna", "valor", "mensagem"),
    [
        ("prompt_versao", "ia-eventos-antigo", "prompt_versao"),
        ("prompt_hash", "0" * 64, "prompt_hash"),
        ("alvo_direcao", "movimento_preco", "alvo_direcao"),
        ("modelo", "qwen3:8b", "modelo"),
        ("modelo_digest", "", "modelo_digest"),
        ("modelo_quantizacao", "Q8_0", "modelo_quantizacao"),
        ("ollama_versao", "", "ollama_versao"),
        ("texto_hash", "0" * 64, "texto_hash"),
    ],
)
def test_lote_recusa_identidade_antiga_ou_texto_desatualizado(
    coluna, valor, mensagem
):
    corpus, classificacoes = _dados()
    classificacoes.loc[0, coluna] = valor

    with pytest.raises(ValueError, match=mensagem):
        preparar.preparar_validacao(corpus, classificacoes, tamanho=8)


def test_lote_exige_identidade_unica_nas_linhas_ok():
    corpus, classificacoes = _dados()
    classificacoes.loc[0, "ollama_versao"] = "0.33.0"

    with pytest.raises(ValueError, match="identidade unica em ollama_versao"):
        preparar.preparar_validacao(corpus, classificacoes, tamanho=8)


def test_lote_ignora_documento_nao_elegivel_do_corpus():
    corpus, classificacoes = _dados()
    corpus.loc[0, "status_documento"] = "sem_texto"
    corpus.loc[0, "texto_llm"] = ""
    classificacoes = classificacoes.iloc[1:].copy()

    resultado = preparar.preparar_validacao(corpus, classificacoes, tamanho=8)

    assert resultado.resumo_interno.loc[0, "documentos_elegiveis"] == 23


def test_cli_recusa_amostra_diferente_de_90():
    with pytest.raises(SystemExit, match="deve ser 90"):
        preparar.main(["--tamanho", "8"])


def test_main_grava_csvs_e_protege_arquivos_existentes(tmp_path):
    corpus, classificacoes = _dados(repeticoes=30)
    corpus_path = tmp_path / "corpus.csv"
    classificacoes_path = tmp_path / "classificacoes.csv"
    avaliador_a = tmp_path / "entrega" / "a.csv"
    avaliador_b = tmp_path / "entrega" / "b.csv"
    chave = tmp_path / "interno" / "chave.csv"
    resumo = tmp_path / "interno" / "resumo.csv"
    corpus.to_csv(corpus_path, index=False)
    classificacoes.to_csv(classificacoes_path, index=False)
    argumentos = [
        "--corpus",
        str(corpus_path),
        "--classificacoes",
        str(classificacoes_path),
        "--avaliador-a",
        str(avaliador_a),
        "--avaliador-b",
        str(avaliador_b),
        "--chave-interna",
        str(chave),
        "--resumo-interno",
        str(resumo),
    ]

    assert preparar.main(argumentos) == 0
    assert len(pd.read_csv(avaliador_a)) == 90
    assert len(pd.read_csv(avaliador_b)) == 90
    assert len(pd.read_csv(chave)) == 90
    assert len(pd.read_csv(resumo)) == 1
    with pytest.raises(FileExistsError, match="ja existe"):
        preparar.main(argumentos)
    assert preparar.main([*argumentos, "--sobrescrever"]) == 0


def test_recusa_saida_compartilhada_e_classificacao_sem_texto(tmp_path):
    corpus, classificacoes = _dados()
    resultado = preparar.preparar_validacao(corpus, classificacoes, tamanho=6)
    mesmo = tmp_path / "mesmo.csv"

    with pytest.raises(ValueError, match="devem ser distintos"):
        preparar.gravar_resultado(
            resultado,
            mesmo,
            mesmo,
            tmp_path / "chave.csv",
            tmp_path / "resumo.csv",
        )

    corpus.loc[corpus["ID_Documento"].eq("doc-01"), "texto_llm"] = ""
    with pytest.raises(ValueError, match="cobrem exatamente"):
        preparar.preparar_validacao(corpus, classificacoes, tamanho=6)
