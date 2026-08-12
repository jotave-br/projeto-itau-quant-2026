from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import v2_03_classificar_eventos as classificar
from src.v2 import ia_eventos


def _modelo() -> dict[str, object]:
    return {
        "modelo": "qwen3:14b",
        "modelo_digest": "a" * 64,
        "modelo_bytes": 9_000_000_000,
        "modelo_modificado_em": "2026-01-01T00:00:00Z",
        "modelo_formato": "gguf",
        "modelo_familia": "qwen3",
        "modelo_parametros": "14.8B",
        "modelo_quantizacao": "Q4_K_M",
        "ollama_versao": "0.32.9",
    }


def _preparar_lote(monkeypatch, tmp_path, corpus):
    corpus_path = tmp_path / "corpus.parquet"
    saida_path = tmp_path / "classificacoes.parquet"
    tabelas = tmp_path / "tabelas"
    tabelas.mkdir()
    corpus.to_parquet(corpus_path, index=False)
    args = argparse.Namespace(
        corpus=corpus_path,
        saida=saida_path,
        modelo="qwen3:14b",
        ollama_url="http://local",
        limite=None,
        forcar=False,
        somente_verificar=False,
    )
    execucao = SimpleNamespace(tabelas=tabelas)
    log = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        exception=lambda *_args, **_kwargs: None,
    )
    manifestos = []

    def gravar(_execucao, _config, **kwargs):
        manifestos.append(kwargs)

    monkeypatch.setattr(classificar, "_argumentos", lambda: args)
    monkeypatch.setattr(classificar, "criar_execucao", lambda *_args: execucao)
    monkeypatch.setattr(classificar, "configurar_log", lambda *_args: log)
    monkeypatch.setattr(classificar, "gravar_manifesto", gravar)
    monkeypatch.setattr(
        classificar, "identidade_modelo_instalado", lambda *_args: _modelo()
    )
    return saida_path, tabelas, manifestos


def test_identidade_exige_tag_univoca_e_preserva_digest(monkeypatch):
    respostas = {
        "http://local/api/version": {"version": "0.32.9"},
        "http://local/api/tags": {
            "models": [
                {
                    "name": "qwen3:14b",
                    "model": "qwen3:14b",
                    "digest": "b" * 64,
                    "size": 9_300_000_000,
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "format": "gguf",
                        "family": "qwen3",
                        "parameter_size": "14.8B",
                        "quantization_level": "Q4_K_M",
                    },
                }
            ]
        },
    }
    monkeypatch.setattr(
        classificar, "_json_http", lambda url, **_kwargs: respostas[url]
    )

    identidade = classificar.identidade_modelo_instalado(
        "qwen3:14b", "http://local/"
    )

    assert identidade["modelo_digest"] == "b" * 64
    assert identidade["ollama_versao"] == "0.32.9"
    assert identidade["modelo_quantizacao"] == "Q4_K_M"


def test_identidade_recusa_modelo_ausente_ou_digest_invalido(monkeypatch):
    def ausente(url, **_kwargs):
        return {"version": "0.32.9"} if url.endswith("version") else {"models": []}

    monkeypatch.setattr(classificar, "_json_http", ausente)
    with pytest.raises(RuntimeError, match="não está instalado"):
        classificar.identidade_modelo_instalado("qwen3:14b")

    def invalido(url, **_kwargs):
        if url.endswith("version"):
            return {"version": "0.32.9"}
        return {
            "models": [
                {
                    "name": "qwen3:14b",
                    "digest": "curto",
                    "size": 100,
                }
            ]
        }

    monkeypatch.setattr(classificar, "_json_http", invalido)
    with pytest.raises(RuntimeError, match="digest"):
        classificar.identidade_modelo_instalado("qwen3:14b")


def test_cache_depende_do_digest_real_do_modelo():
    primeira = classificar._chave_cache("c" * 64, "a" * 64, "0.32.9")
    segunda = classificar._chave_cache("c" * 64, "b" * 64, "0.32.9")
    outra_versao = classificar._chave_cache("c" * 64, "a" * 64, "0.33.0")

    assert len(primeira) == 64
    assert primeira != segunda
    assert primeira != outra_versao


def test_classificacao_grava_cache_atomico_e_reutiliza(monkeypatch, tmp_path):
    monkeypatch.setattr(classificar, "DIR_CACHE", tmp_path)
    texto = "A companhia assinou contrato que amplia sua capacidade."
    identidade = ia_eventos.identidade_requisicao(texto)
    resultado = ia_eventos.ResultadoClassificacao(
        classificacao=ia_eventos.ClassificacaoEvento(
            especifico_empresa=True,
            direcao="positiva",
            evidencia="assinou contrato que amplia sua capacidade",
            abster=False,
            motivo_abstencao="",
        ),
        identidade=identidade,
        modelo="qwen3:14b",
        think_suportado=True,
        metricas=ia_eventos.MetricasInferencia(
            total_duration_ns=1_000_000_000,
            prompt_eval_count=100,
            eval_count=30,
        ),
    )
    chamadas = []

    def fake(*_args, **_kwargs):
        chamadas.append(1)
        return resultado

    monkeypatch.setattr(ia_eventos, "classificar_com_ollama", fake)
    registro = {"ID_Documento": "doc-1", "texto_llm": texto}

    primeira, cache1 = classificar._classificar_um(
        registro, _modelo(), base_url="http://local", forcar=False
    )
    segunda, cache2 = classificar._classificar_um(
        registro, _modelo(), base_url="http://local", forcar=False
    )

    assert cache1 is False
    assert cache2 is True
    assert len(chamadas) == 1
    assert primeira == segunda
    assert primeira["modelo_digest"] == "a" * 64
    assert primeira["direcao"] == "positiva"
    assert primeira["prompt_versao"] == ia_eventos.PROMPT_VERSAO
    assert primeira["prompt_hash"] == ia_eventos.HASH_PROMPT
    assert primeira["alvo_direcao"] == ia_eventos.ALVO_DIRECAO
    assert primeira["resposta_bruta"] == ""
    assert segunda["resposta_bruta"] == ""
    arquivos = list(tmp_path.glob("*.json"))
    assert len(arquivos) == 1
    assert json.loads(arquivos[0].read_text("utf-8"))["status_ia"] == "ok"


def test_limite_exige_saida_separada_da_oficial(monkeypatch):
    args = argparse.Namespace(
        corpus=classificar.CORPUS_PADRAO,
        saida=classificar.CLASSIFICACOES_PADRAO,
        modelo="qwen3:14b",
        ollama_url="http://local",
        limite=5,
        forcar=False,
        somente_verificar=False,
    )
    monkeypatch.setattr(classificar, "_argumentos", lambda: args)
    monkeypatch.setattr(
        classificar,
        "criar_execucao",
        lambda *_args: pytest.fail("execucao oficial nao deveria ser criada"),
    )

    with pytest.raises(SystemExit, match="saida diferente"):
        classificar.main()


def test_lote_recusa_corpus_sem_documento_elegivel(monkeypatch, tmp_path):
    corpus = pd.DataFrame(
        {
            "ID_Documento": ["doc-1"],
            "status_documento": ["sem_texto"],
            "texto_llm": [""],
        }
    )
    saida, _tabelas, manifestos = _preparar_lote(
        monkeypatch, tmp_path, corpus
    )
    monkeypatch.setattr(
        classificar,
        "_classificar_um",
        lambda *_args, **_kwargs: pytest.fail("classificacao nao deveria iniciar"),
    )

    codigo = classificar.main()

    assert codigo == 1
    assert not saida.exists()
    assert manifestos[-1]["status"] == "falhou"
    assert "nenhum documento elegível" in manifestos[-1]["extras"]["erro"]


def test_lote_so_com_erros_grava_saida_e_resumo(monkeypatch, tmp_path):
    corpus = pd.DataFrame(
        {
            "ID_Documento": ["doc-2", "doc-1"],
            "status_documento": ["ok", "ok"],
            "texto_llm": ["texto dois", "texto um"],
        }
    )
    saida_path, tabelas, manifestos = _preparar_lote(
        monkeypatch, tmp_path, corpus
    )

    def falhar(registro, *_args, **_kwargs):
        erro = ia_eventos.ErroClassificacao("evidencia invalida")
        erro.resposta_json = json.dumps({"documento": registro["ID_Documento"]})
        raise erro

    monkeypatch.setattr(classificar, "_classificar_um", falhar)

    codigo = classificar.main()

    assert codigo == 2
    saida = pd.read_parquet(saida_path)
    assert {
        "status_ia",
        "erro_ia",
        "especifico_empresa",
        "direcao",
        "evidencia",
        "abster",
        "motivo_abstencao",
        "resposta_bruta",
    } <= set(saida.columns)
    assert saida["status_ia"].eq("erro").all()
    assert saida["direcao"].isna().all()
    assert saida["resposta_bruta"].str.contains("doc-").all()
    resumo = pd.read_csv(tabelas / "ia_resumo_direcao.csv")
    assert resumo.to_dict("records") == [{"direcao": "erro", "documentos": 2}]
    assert manifestos[-1]["status"] == "concluida_com_erros"


@pytest.mark.parametrize("status", [None, 503])
def test_lote_aborta_no_primeiro_erro_sistemico(
    monkeypatch, tmp_path, status
):
    corpus = pd.DataFrame(
        {
            "ID_Documento": ["doc-1", "doc-2"],
            "status_documento": ["ok", "ok"],
            "texto_llm": ["texto um", "texto dois"],
        }
    )
    saida, _tabelas, manifestos = _preparar_lote(
        monkeypatch, tmp_path, corpus
    )
    chamadas = []

    def falhar(registro, *_args, **_kwargs):
        chamadas.append(registro["ID_Documento"])
        raise ia_eventos.ErroHTTPollama(status, "indisponivel")

    monkeypatch.setattr(classificar, "_classificar_um", falhar)

    codigo = classificar.main()

    assert codigo == 1
    assert chamadas == ["doc-1"]
    assert not saida.exists()
    assert manifestos[-1]["status"] == "falhou"
    assert "Ollama" in manifestos[-1]["extras"]["erro"]
