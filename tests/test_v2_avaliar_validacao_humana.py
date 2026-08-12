from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import v2_04_preparar_validacao_humana as preparar
from scripts import v2_05_avaliar_validacao_humana as avaliar
from src.v2 import ia_eventos, validacao_ia


ROTULOS = [
    (True, "positiva"),
    (True, "negativa"),
    (True, "neutra"),
    (False, "neutra"),
] * 2


def _dados():
    ids = [f"VH-{numero:012X}" for numero in range(1, len(ROTULOS) + 1)]
    textos = {codigo: f"Texto público {numero}" for numero, codigo in enumerate(ids)}
    ficha_a = pd.DataFrame(
        {
            "id_anonimo": ids,
            "texto": [textos[codigo] for codigo in ids],
            "especifico_empresa": [valor for valor, _direcao in ROTULOS],
            "direcao": [direcao for _valor, direcao in ROTULOS],
        }
    )
    ordem_b = list(reversed(ids))
    rotulos_por_id = dict(zip(ids, ROTULOS, strict=True))
    ficha_b = pd.DataFrame(
        {
            "id_anonimo": ordem_b,
            "texto": [textos[codigo] for codigo in ordem_b],
            "especifico_empresa": [rotulos_por_id[codigo][0] for codigo in ordem_b],
            "direcao": [rotulos_por_id[codigo][1] for codigo in ordem_b],
        }
    )
    ficha_b.loc[ficha_b["id_anonimo"].eq(ids[0]), "direcao"] = "negativa"
    posicao_a = {codigo: numero for numero, codigo in enumerate(ids, start=1)}
    posicao_b = {
        codigo: numero for numero, codigo in enumerate(ordem_b, start=1)
    }
    chave = pd.DataFrame(
        {
            "id_anonimo": ids,
            "ID_Documento": [f"doc-{numero:02d}" for numero in range(1, 9)],
            "especifico_empresa_ia": [valor for valor, _direcao in ROTULOS],
            "direcao_ia": [direcao for _valor, direcao in ROTULOS],
            "abster_ia": [
                not especifico or direcao == "neutra"
                for especifico, direcao in ROTULOS
            ],
            "ordem_avaliador_a": [posicao_a[codigo] for codigo in ids],
            "ordem_avaliador_b": [posicao_b[codigo] for codigo in ids],
            "protocolo_rotulagem": preparar.VERSAO_PROTOCOLO,
            "desenho_amostra": preparar.DESENHO_AMOSTRA,
        }
    )
    chave["classe_ia"] = [
        f"especifico_{direcao}" if especifico else "nao_especifico"
        for especifico, direcao in ROTULOS
    ]
    chave["protocolo_rotulagem_hash"] = hashlib.sha256(
        preparar.PROTOCOLO_ROTULAGEM.read_bytes()
    ).hexdigest()
    chave["tamanho_amostra"] = len(chave)
    chave["seed_amostra"] = preparar.SEED_VALIDACAO
    chave["lote_hash"] = "a" * 64
    chave["prompt_versao"] = ia_eventos.PROMPT_VERSAO
    chave["prompt_hash"] = ia_eventos.HASH_PROMPT
    chave["alvo_direcao"] = ia_eventos.ALVO_DIRECAO
    chave["modelo"] = preparar.MODELO_CONGELADO
    chave["modelo_digest"] = preparar.DIGEST_MODELO_CONGELADO
    chave["modelo_quantizacao"] = preparar.QUANTIZACAO_CONGELADA
    chave["ollama_versao"] = "0.32.9"
    adjudicacao = pd.DataFrame(
        {
            "id_anonimo": [ids[0]],
            "texto": [textos[ids[0]]],
            "gold_especifico_empresa": [True],
            "gold_direcao": ["positiva"],
        }
    )
    return ficha_a, ficha_b, chave, adjudicacao


def _dados_oficiais(com_divergencia: bool = False):
    rotulos = (
        [(True, "positiva")] * 30
        + [(True, "negativa")] * 30
        + [(False, "neutra")] * 3
        + [(True, "neutra")] * 27
    )
    ids = [f"VH-{numero:012X}" for numero in range(1, 91)]
    textos = {codigo: f"Texto público {numero}" for numero, codigo in enumerate(ids)}
    ordem_b = list(reversed(ids))
    por_id = dict(zip(ids, rotulos, strict=True))
    ficha_a = pd.DataFrame(
        {
            "id_anonimo": ids,
            "texto": [textos[codigo] for codigo in ids],
            "especifico_empresa": [valor for valor, _ in rotulos],
            "direcao": [direcao for _, direcao in rotulos],
        }
    )
    ficha_b = pd.DataFrame(
        {
            "id_anonimo": ordem_b,
            "texto": [textos[codigo] for codigo in ordem_b],
            "especifico_empresa": [por_id[codigo][0] for codigo in ordem_b],
            "direcao": [por_id[codigo][1] for codigo in ordem_b],
        }
    )
    if com_divergencia:
        ficha_b.loc[ficha_b["id_anonimo"].eq(ids[0]), "direcao"] = "negativa"
    posicao_a = {codigo: numero for numero, codigo in enumerate(ids, start=1)}
    posicao_b = {
        codigo: numero for numero, codigo in enumerate(ordem_b, start=1)
    }
    chave = pd.DataFrame(
        {
            "id_anonimo": ids,
            "ID_Documento": [f"doc-{numero:03d}" for numero in range(1, 91)],
            "especifico_empresa_ia": [valor for valor, _ in rotulos],
            "direcao_ia": [direcao for _, direcao in rotulos],
            "abster_ia": [
                not especifico or direcao == "neutra"
                for especifico, direcao in rotulos
            ],
            "ordem_avaliador_a": [posicao_a[codigo] for codigo in ids],
            "ordem_avaliador_b": [posicao_b[codigo] for codigo in ids],
        }
    )
    chave["protocolo_rotulagem"] = preparar.VERSAO_PROTOCOLO
    chave["desenho_amostra"] = preparar.DESENHO_AMOSTRA
    chave["classe_ia"] = [
        f"especifico_{direcao}" if especifico else "nao_especifico"
        for especifico, direcao in rotulos
    ]
    chave["protocolo_rotulagem_hash"] = hashlib.sha256(
        preparar.PROTOCOLO_ROTULAGEM.read_bytes()
    ).hexdigest()
    chave["tamanho_amostra"] = 90
    chave["seed_amostra"] = preparar.SEED_VALIDACAO
    chave["lote_hash"] = "b" * 64
    chave["prompt_versao"] = ia_eventos.PROMPT_VERSAO
    chave["prompt_hash"] = ia_eventos.HASH_PROMPT
    chave["alvo_direcao"] = ia_eventos.ALVO_DIRECAO
    chave["modelo"] = preparar.MODELO_CONGELADO
    chave["modelo_digest"] = preparar.DIGEST_MODELO_CONGELADO
    chave["modelo_quantizacao"] = preparar.QUANTIZACAO_CONGELADA
    chave["ollama_versao"] = "0.32.9"
    resumo = pd.DataFrame(
        [
            {
                "lote_hash": "b" * 64,
                "prompt_versao": ia_eventos.PROMPT_VERSAO,
                "prompt_hash": ia_eventos.HASH_PROMPT,
                "alvo_direcao": ia_eventos.ALVO_DIRECAO,
                "modelo": preparar.MODELO_CONGELADO,
                "modelo_digest": preparar.DIGEST_MODELO_CONGELADO,
                "modelo_quantizacao": preparar.QUANTIZACAO_CONGELADA,
                "ollama_versao": "0.32.9",
                "documentos_elegiveis": 90,
                "classificacoes_ok": 90,
                "erros_tecnicos": 0,
                "taxa_erro_tecnico": 0.0,
                "abstencoes": 30,
                "taxa_abstencao": 1 / 3,
                "tamanho_amostra": 90,
                "seed_amostra": preparar.SEED_VALIDACAO,
                "protocolo_rotulagem": preparar.VERSAO_PROTOCOLO,
                "protocolo_rotulagem_hash": chave.at[
                    0, "protocolo_rotulagem_hash"
                ],
                "desenho_amostra": preparar.DESENHO_AMOSTRA,
                "nao_especifico": 3,
                "especifico_positiva": 30,
                "especifico_negativa": 30,
                "especifico_neutra": 27,
            }
        ]
    )
    return ficha_a, ficha_b, chave, resumo


def _mock_execucao(monkeypatch, tmp_path):
    execucao = SimpleNamespace(tabelas=tmp_path / "tabelas")
    execucao.tabelas.mkdir(parents=True)
    log = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        exception=lambda *_args, **_kwargs: None,
    )
    manifestos = []

    def gravar(_execucao, config, **kwargs):
        manifestos.append((config, kwargs))

    monkeypatch.setattr(avaliar, "criar_execucao", lambda *_args: execucao)
    monkeypatch.setattr(avaliar, "configurar_log", lambda *_args: log)
    monkeypatch.setattr(avaliar, "gravar_manifesto", gravar)
    return execucao, manifestos


def test_reconcilia_divergencia_e_aprova_gates_com_aliases(monkeypatch):
    monkeypatch.setattr(validacao_ia, "SUPORTE_MINIMO_CLASSE", 1)
    ficha_a, ficha_b, chave, adjudicacao = _dados()
    ficha_b = ficha_b.rename(
        columns={
            "id_anonimo": "codigo_anonimo",
            "texto": "texto_documento",
            "especifico_empresa": "evento_especifico",
            "direcao": "direção",
        }
    )
    ficha_b["evento_especifico"] = ficha_b["evento_especifico"].map(
        {True: "SIM", False: "NÃO"}
    )

    resultado = avaliar.avaliar_validacao(
        ficha_a, ficha_b, chave, adjudicacao
    )

    assert resultado.avaliacao.macro_f1 == pytest.approx(1.0)
    assert resultado.avaliacao.kappa_avaliadores == pytest.approx(5 / 6)
    assert resultado.avaliacao.cobertura == pytest.approx(0.5)
    assert resultado.avaliacao.aprovado is True
    assert resultado.gold_adjudicado["origem_rotulo"].value_counts().to_dict() == {
        "concordancia": 7,
        "adjudicacao": 1,
    }
    assert len(resultado.divergencias) == 1


def test_consumo_direto_do_contrato_v2_04():
    rotulos = [
        (False, "neutra"),
        (True, "positiva"),
        (True, "negativa"),
        (True, "neutra"),
    ] * 3
    corpus = pd.DataFrame(
        {
            "ID_Documento": [f"doc-{numero:02d}" for numero in range(12)],
            "Data_Entrega": [f"{2020 + numero % 4}-03-10" for numero in range(12)],
            "emissor_id": [f"E{numero % 5}" for numero in range(12)],
            "texto_llm": [f"Texto público {numero}" for numero in range(12)],
            "status_documento": "ok",
        }
    )
    classificacoes = pd.DataFrame(
        {
            "ID_Documento": corpus["ID_Documento"],
            "status_ia": "ok",
            "especifico_empresa": [especifico for especifico, _ in rotulos],
            "direcao": [direcao for _, direcao in rotulos],
            "abster": [
                not especifico or direcao == "neutra"
                for especifico, direcao in rotulos
            ],
        }
    )
    classificacoes["erro_ia"] = ""
    classificacoes["evidencia"] = "trecho interno"
    classificacoes["prompt_versao"] = ia_eventos.PROMPT_VERSAO
    classificacoes["prompt_hash"] = ia_eventos.HASH_PROMPT
    classificacoes["alvo_direcao"] = ia_eventos.ALVO_DIRECAO
    classificacoes["modelo"] = preparar.MODELO_CONGELADO
    classificacoes["modelo_digest"] = preparar.DIGEST_MODELO_CONGELADO
    classificacoes["modelo_quantizacao"] = preparar.QUANTIZACAO_CONGELADA
    classificacoes["ollama_versao"] = "0.32.9"
    classificacoes["texto_hash"] = [
        ia_eventos.identidade_requisicao(texto).hash_texto
        for texto in corpus["texto_llm"]
    ]
    preparado = preparar.preparar_validacao(corpus, classificacoes, tamanho=9)
    chave = preparado.chave_interna.set_index("id_anonimo")
    fichas = []
    for ficha in (preparado.avaliador_a, preparado.avaliador_b):
        preenchida = ficha.copy()
        preenchida["especifico_empresa"] = preenchida["id_anonimo"].map(
            chave["especifico_empresa_ia"]
        )
        preenchida["direcao"] = preenchida["id_anonimo"].map(chave["direcao_ia"])
        fichas.append(preenchida)

    resultado = avaliar.avaliar_validacao(
        fichas[0], fichas[1], preparado.chave_interna
    )

    assert resultado.avaliacao.macro_f1 == pytest.approx(0.75)
    assert resultado.avaliacao.kappa_avaliadores == pytest.approx(1.0)
    assert resultado.avaliacao.aprovado is False


def test_divergencia_sem_adjudicacao_gera_ficha_cega():
    ficha_a, ficha_b, chave, _adjudicacao = _dados()

    with pytest.raises(avaliar.AdjudicacaoPendente) as erro:
        avaliar.avaliar_validacao(ficha_a, ficha_b, chave)

    tabela = erro.value.tabela
    assert len(tabela) == 1
    assert tabela["gold_especifico_empresa"].isna().all()
    assert tabela["gold_direcao"].isna().all()
    assert "texto" in tabela
    assert "ID_Documento" not in tabela
    assert "direcao_ia" not in tabela
    assert "direcao_a" not in tabela
    assert "direcao_b" not in tabela


def test_recusa_ids_valores_e_adjudicacao_incompleta():
    ficha_a, ficha_b, chave, adjudicacao = _dados()
    duplicada = ficha_a.copy()
    duplicada.loc[1, "id_anonimo"] = duplicada.loc[0, "id_anonimo"]
    with pytest.raises(ValueError, match="duplicado"):
        avaliar.avaliar_validacao(duplicada, ficha_b, chave, adjudicacao)

    invalida = ficha_a.copy()
    invalida.loc[0, "direcao"] = "alta"
    with pytest.raises(ValueError, match="positiva, negativa ou neutra"):
        avaliar.avaliar_validacao(invalida, ficha_b, chave, adjudicacao)

    id_estranho = ficha_a.copy()
    id_estranho.loc[0, "id_anonimo"] = "VH-FFFFFFFFFFFF"
    with pytest.raises(ValueError, match="mesmos IDs"):
        avaliar.avaliar_validacao(id_estranho, ficha_b, chave, adjudicacao)

    adjudicacao_vazia = adjudicacao.iloc[0:0]
    with pytest.raises(ValueError, match="exatamente os IDs"):
        avaliar.avaliar_validacao(ficha_a, ficha_b, chave, adjudicacao_vazia)

    # o terceiro avaliador tem que julgar o mesmo documento, nao outro
    texto_trocado = adjudicacao.copy()
    texto_trocado.loc[0, "texto"] = "Texto de outro documento"
    with pytest.raises(ValueError, match="texto diverge"):
        avaliar.avaliar_validacao(ficha_a, ficha_b, chave, texto_trocado)


def test_recusa_colunas_extras_amostra_curta_e_resumo_adulterado():
    ficha_a, _ficha_b, chave, adjudicacao = _dados()
    with pytest.raises(ValueError, match="colunas extras"):
        avaliar.normalizar_ficha(
            ficha_a.assign(preco_futuro=100.0),
            "avaliador A",
        )
    with pytest.raises(ValueError, match="colunas extras"):
        avaliar._normalizar_adjudicacao(adjudicacao.assign(direcao_ia="positiva"))
    with pytest.raises(ValueError, match="90 documentos"):
        avaliar.normalizar_chave(chave, exigir_amostra_oficial=True)

    _a, _b, chave_oficial, resumo = _dados_oficiais()
    chave_validada = avaliar.normalizar_chave(
        chave_oficial,
        exigir_amostra_oficial=True,
    )
    resumo.loc[0, "erros_tecnicos"] = 1
    with pytest.raises(ValueError, match="contagens de cobertura"):
        avaliar.normalizar_resumo_lote(resumo, chave_validada)


def test_macro_f1_e_kappa_reprovam_seus_gates(monkeypatch):
    monkeypatch.setattr(validacao_ia, "SUPORTE_MINIMO_CLASSE", 1)
    ficha_a, ficha_b, chave, adjudicacao = _dados()
    trocas = {"positiva": "negativa", "negativa": "positiva"}
    chave_f1 = chave.copy()
    chave_f1["direcao_ia"] = chave_f1["direcao_ia"].replace(trocas)
    chave_f1["classe_ia"] = [
        f"especifico_{direcao}" if especifico else "nao_especifico"
        for especifico, direcao in zip(
            chave_f1["especifico_empresa_ia"],
            chave_f1["direcao_ia"],
            strict=True,
        )
    ]
    resultado_f1 = avaliar.avaliar_validacao(
        ficha_a, ficha_b, chave_f1, adjudicacao
    )

    assert resultado_f1.avaliacao.macro_f1 == pytest.approx(0.5)
    assert resultado_f1.avaliacao.aprovado_macro_f1 is False
    assert resultado_f1.avaliacao.aprovado_kappa is True
    assert resultado_f1.avaliacao.aprovado is False

    ficha_b_kappa = ficha_b.copy()
    adjudicacoes = []
    por_id = ficha_a.set_index("id_anonimo")
    for indice, linha in ficha_b_kappa.iterrows():
        codigo = linha["id_anonimo"]
        especifico = bool(por_id.at[codigo, "especifico_empresa"])
        direcao = str(por_id.at[codigo, "direcao"])
        if direcao == "positiva":
            novo_especifico, nova_direcao = True, "negativa"
        elif direcao == "negativa":
            novo_especifico, nova_direcao = True, "positiva"
        elif especifico:
            novo_especifico, nova_direcao = False, "neutra"
        else:
            novo_especifico, nova_direcao = True, "neutra"
        ficha_b_kappa.at[indice, "especifico_empresa"] = novo_especifico
        ficha_b_kappa.at[indice, "direcao"] = nova_direcao
        adjudicacoes.append(
            {
                "id_anonimo": codigo,
                "texto": str(por_id.at[codigo, "texto"]),
                "gold_especifico_empresa": especifico,
                "gold_direcao": direcao,
            }
        )
    resultado_kappa = avaliar.avaliar_validacao(
        ficha_a, ficha_b_kappa, chave, pd.DataFrame(adjudicacoes)
    )

    assert resultado_kappa.avaliacao.macro_f1 == pytest.approx(1.0)
    assert resultado_kappa.avaliacao.kappa_avaliadores < 0.60
    assert resultado_kappa.avaliacao.aprovado_macro_f1 is True
    assert resultado_kappa.avaliacao.aprovado_kappa is False
    assert resultado_kappa.avaliacao.aprovado is False


def test_main_grava_tabelas_manifesto_e_gate(monkeypatch, tmp_path):
    ficha_a, ficha_b, chave, resumo_lote = _dados_oficiais()
    caminhos = {
        "a": tmp_path / "avaliador_a.csv",
        "b": tmp_path / "avaliador_b.csv",
        "chave": tmp_path / "chave.csv",
        "resumo": tmp_path / "resumo.csv",
    }
    for tabela, caminho in zip(
        (ficha_a, ficha_b, chave, resumo_lote), caminhos.values(), strict=True
    ):
        tabela.to_csv(caminho, index=False, encoding="utf-8-sig")
    execucao, manifestos = _mock_execucao(monkeypatch, tmp_path / "run")

    codigo = avaliar.main(
        [
            "--avaliador-a",
            str(caminhos["a"]),
            "--avaliador-b",
            str(caminhos["b"]),
            "--chave-interna",
            str(caminhos["chave"]),
            "--resumo-interno",
            str(caminhos["resumo"]),
        ]
    )

    assert codigo == 0
    assert {caminho.name for caminho in execucao.tabelas.glob("*.csv")} == {
        "rotulos_humanos_reconciliados.csv",
        "divergencias_adjudicadas.csv",
        "gold_adjudicado.csv",
        "matriz_confusao.csv",
        "metricas_por_classe.csv",
        "resumo_validacao.csv",
        "resumo_lote_ia.csv",
    }
    resumo = pd.read_csv(execucao.tabelas / "resumo_validacao.csv")
    assert resumo.loc[0, "limiar_macro_f1"] == pytest.approx(0.70)
    assert resumo.loc[0, "limiar_kappa"] == pytest.approx(0.60)
    assert bool(resumo.loc[0, "aprovado"])
    config, manifesto = manifestos[-1]
    assert config["limiar_macro_f1"] == 0.70
    assert config["limiar_kappa"] == 0.60
    assert manifesto["status"] == "concluida"
    assert manifesto["extras"]["aprovado"] is True
    assert set(manifesto["arquivos_dados"]) == set(caminhos.values())


def test_main_para_e_grava_divergencias_pendentes(monkeypatch, tmp_path):
    ficha_a, ficha_b, chave, resumo_lote = _dados_oficiais(
        com_divergencia=True
    )
    caminhos = [
        tmp_path / nome for nome in ("a.csv", "b.csv", "chave.csv", "resumo.csv")
    ]
    for tabela, caminho in zip(
        (ficha_a, ficha_b, chave, resumo_lote), caminhos, strict=True
    ):
        tabela.to_csv(caminho, index=False, encoding="utf-8-sig")
    execucao, manifestos = _mock_execucao(monkeypatch, tmp_path / "run")

    codigo = avaliar.main(
        [
            "--avaliador-a",
            str(caminhos[0]),
            "--avaliador-b",
            str(caminhos[1]),
            "--chave-interna",
            str(caminhos[2]),
            "--resumo-interno",
            str(caminhos[3]),
        ]
    )

    assert codigo == 3
    pendencias = pd.read_csv(
        execucao.tabelas / "divergencias_para_adjudicacao.csv"
    )
    assert len(pendencias) == 1
    assert "direcao_ia" not in pendencias
    assert manifestos[-1][1]["status"] == "aguardando_adjudicacao"
