from __future__ import annotations

import json

import pytest

from src.v2 import ia_eventos


TEXTO = (
    "A companhia aprovou novo contrato de fornecimento por cinco anos. "
    "O acordo amplia a capacidade contratada da emissora."
)


def _classificacao_valida(**alteracoes):
    resposta = {
        "especifico_empresa": True,
        "direcao": "positiva",
        "evidencia": "aprovou novo contrato de fornecimento por cinco anos",
    }
    resposta.update(alteracoes)
    return resposta


def _envelope(classificacao: dict) -> bytes:
    return json.dumps(
        {
            "model": ia_eventos.MODELO_DEFAULT,
            "message": {
                "role": "assistant",
                "content": json.dumps(classificacao, ensure_ascii=False),
            },
            "done": True,
            "total_duration": 2_000_000_000,
            "prompt_eval_count": 321,
            "eval_count": 45,
        },
        ensure_ascii=False,
    ).encode("utf-8")


def test_payload_local_e_deterministico_com_schema_estrito():
    payload = ia_eventos.montar_payload(TEXTO)

    assert payload["model"] == "qwen3:14b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0,
        "seed": ia_eventos.SEED_FIXA,
        "num_ctx": 8192,
        "num_predict": 300,
    }
    assert payload["keep_alive"] == "30m"
    assert payload["format"] == ia_eventos.SCHEMA_RESPOSTA
    assert payload["format"] is not ia_eventos.SCHEMA_RESPOSTA
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert TEXTO in payload["messages"][1]["content"]

    serializado = payload["messages"][1]["content"].casefold()
    for termo in ("preço", "preco", "p&l", "seguidora"):
        assert termo not in serializado
    assert "não use preço" in payload["messages"][0]["content"].casefold()


def test_schema_exige_tres_campos_sem_extras_e_enum_fechado():
    schema = ia_eventos.SCHEMA_RESPOSTA

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "especifico_empresa",
        "direcao",
        "evidencia",
    }
    assert schema["properties"]["direcao"]["enum"] == [
        "positiva",
        "negativa",
        "neutra",
    ]
    assert schema["properties"]["especifico_empresa"]["type"] == "boolean"


def test_resposta_valida_e_evidencia_literal():
    classificacao = ia_eventos.validar_classificacao(
        json.dumps(_classificacao_valida(), ensure_ascii=False), TEXTO
    )

    assert classificacao.especifico_empresa is True
    assert classificacao.direcao == "positiva"
    assert classificacao.evidencia in TEXTO
    assert classificacao.abster is False
    assert classificacao.motivo_abstencao == ""


@pytest.mark.parametrize("direcao", ["alta", "POSITIVA", "", 1, None])
def test_enum_direcao_rejeita_valores_fora_do_contrato(direcao):
    with pytest.raises(ia_eventos.ErroClassificacao, match="direcao"):
        ia_eventos.validar_classificacao(
            _classificacao_valida(direcao=direcao), TEXTO
        )


def test_campos_extras_e_ausentes_sao_rejeitados():
    extra = _classificacao_valida(confianca=0.9)
    ausente = _classificacao_valida()
    del ausente["evidencia"]

    with pytest.raises(ia_eventos.ErroClassificacao, match="extras"):
        ia_eventos.validar_classificacao(extra, TEXTO)
    with pytest.raises(ia_eventos.ErroClassificacao, match="faltantes"):
        ia_eventos.validar_classificacao(ausente, TEXTO)


def test_booleanos_nao_aceitam_inteiros():
    with pytest.raises(ia_eventos.ErroClassificacao, match="boolean"):
        ia_eventos.validar_classificacao(
            _classificacao_valida(especifico_empresa=1), TEXTO
        )


def test_evidencia_inventada_e_rejeitada():
    with pytest.raises(ia_eventos.ErroClassificacao, match="trecho literal"):
        ia_eventos.validar_classificacao(
            _classificacao_valida(
                evidencia="A receita crescerá vinte por cento no próximo ano"
            ),
            TEXTO,
        )


@pytest.mark.parametrize(
    "evidencia",
    [
        "novo contrato",
        "aprovou " + "contrato " * 30,
        "aprovou novo contrato... por cinco anos",
        "aprovou novo contrato . . . por cinco anos",
        "aprovou novo contrato… por cinco anos",
    ],
)
def test_evidencia_respeita_tamanho_e_nao_usa_reticencias(evidencia):
    with pytest.raises(ia_eventos.ErroClassificacao, match="evidencia"):
        ia_eventos.validar_classificacao(
            _classificacao_valida(evidencia=evidencia),
            TEXTO,
        )


def test_abstencao_e_derivada_da_direcao_e_especificidade():
    valida = ia_eventos.validar_classificacao(
        _classificacao_valida(
            direcao="neutra",
        ),
        TEXTO,
    )
    assert valida.abster is True
    assert valida.motivo_abstencao == "direcao_neutra"


def test_nao_especifico_precisa_ser_neutro():
    with pytest.raises(ia_eventos.ErroClassificacao, match="direção neutra"):
        ia_eventos.validar_classificacao(
            _classificacao_valida(especifico_empresa=False), TEXTO
        )


def test_evidencia_pode_juntar_quebras_de_linha_do_pdf():
    texto = "A companhia assinou um contrato\nque amplia sua capacidade ."
    resposta = _classificacao_valida(
        evidencia="assinou um contrato que amplia sua capacidade"
    )
    resultado = ia_eventos.validar_classificacao(resposta, texto)
    assert resultado.direcao == "positiva"


def test_evidencia_tolera_espacos_de_ocr_e_pontuacao():
    texto = (
        "foi aprovado o valor de R$ 527.136.000,00 a títu lo de "
        "remuneração aos acionistas relativo a COVID -19"
    )
    resposta = _classificacao_valida(
        evidencia=(
            "Foi aprovado o valor de R$ 527.136.000,00 a título de "
            "remuneração aos acionistas relativo a COVID-19"
        )
    )

    resultado = ia_eventos.validar_classificacao(resposta, texto)

    assert resultado.direcao == "positiva"


def test_evidencia_nao_pode_costurar_titulo_e_corpo():
    texto = (
        "Petrobras obtém decisão favorável no processo da Refinaria de "
        "Manguinhos. A companhia obteve decisão favorável no processo "
        "envolvendo a Refinaria de Manguinhos."
    )
    resposta = _classificacao_valida(
        evidencia=(
            "Petrobras obtém decisão favorável no processo envolvendo a "
            "Refinaria de Manguinhos"
        )
    )

    with pytest.raises(ia_eventos.ErroClassificacao, match="trecho literal"):
        ia_eventos.validar_classificacao(resposta, texto)


def test_erro_de_classificacao_preserva_resposta_para_diagnostico():
    resposta = _classificacao_valida(evidencia="trecho inventado")

    def transporte(_url, _payload, _timeout):
        return ia_eventos.RespostaHTTP(200, _envelope(resposta))

    with pytest.raises(ia_eventos.ErroClassificacao) as erro:
        ia_eventos.classificar_com_ollama(TEXTO, transporte=transporte)

    assert "trecho inventado" in erro.value.resposta_json


def test_json_com_prosa_cerca_ou_chave_duplicada_e_rejeitado():
    classificacao = json.dumps(_classificacao_valida(), ensure_ascii=False)
    com_prosa = "Resultado: " + classificacao
    duplicado = classificacao.replace(
        '"direcao": "positiva"',
        '"direcao": "positiva", "direcao": "neutra"',
    )

    with pytest.raises(ia_eventos.ErroClassificacao, match="JSON puro"):
        ia_eventos.validar_classificacao(com_prosa, TEXTO)
    with pytest.raises(ia_eventos.ErroClassificacao, match="duplicado"):
        ia_eventos.validar_classificacao(duplicado, TEXTO)


def test_hashes_e_chave_de_cache_sao_deterministicos_e_sensiveis_as_entradas():
    primeira = ia_eventos.identidade_requisicao(TEXTO)
    segunda = ia_eventos.identidade_requisicao("  " + TEXTO + "  ")
    outro_texto = ia_eventos.identidade_requisicao(TEXTO + " Outro fato.")
    outro_modelo = ia_eventos.identidade_requisicao(TEXTO, modelo="qwen3:8b")

    assert primeira == segunda
    assert primeira.hash_prompt == ia_eventos.HASH_PROMPT
    assert all(
        len(hash_) == 64
        for hash_ in (
            primeira.hash_prompt,
            primeira.hash_texto,
            primeira.hash_modelo,
            primeira.chave_cache,
        )
    )
    assert primeira.hash_texto != outro_texto.hash_texto
    assert primeira.chave_cache != outro_texto.chave_cache
    assert primeira.hash_modelo != outro_modelo.hash_modelo
    assert primeira.chave_cache != outro_modelo.chave_cache
    assert ia_eventos.montar_payload(TEXTO) == ia_eventos.montar_payload(TEXTO)


def test_cliente_injetavel_chama_api_local_e_valida_resposta():
    chamadas = []

    def transporte(url, payload, timeout):
        chamadas.append((url, payload, timeout))
        return ia_eventos.RespostaHTTP(200, _envelope(_classificacao_valida()))

    resultado = ia_eventos.classificar_com_ollama(
        TEXTO, transporte=transporte, timeout=7.5
    )

    assert len(chamadas) == 1
    assert chamadas[0][0] == "http://127.0.0.1:11434/api/chat"
    assert chamadas[0][1]["think"] is False
    assert chamadas[0][2] == 7.5
    assert resultado.classificacao.direcao == "positiva"
    assert resultado.modelo == "qwen3:14b"
    assert resultado.identidade == ia_eventos.identidade_requisicao(TEXTO)
    assert resultado.metricas.total_duration_ns == 2_000_000_000
    assert resultado.metricas.prompt_eval_count == 321
    assert resultado.metricas.eval_count == 45


def test_cliente_retenta_sem_think_somente_se_servidor_for_incompativel():
    payloads = []

    def transporte(_url, payload, _timeout):
        payloads.append(payload)
        if len(payloads) == 1:
            return ia_eventos.RespostaHTTP(
                400, b'json: unknown field "think"'
            )
        return ia_eventos.RespostaHTTP(200, _envelope(_classificacao_valida()))

    resultado = ia_eventos.classificar_com_ollama(TEXTO, transporte=transporte)

    assert len(payloads) == 2
    assert payloads[0]["think"] is False
    assert "think" not in payloads[1]
    assert resultado.think_suportado is False
    assert resultado.identidade == ia_eventos.identidade_requisicao(
        TEXTO, incluir_think=False
    )


def test_falha_http_e_explicita_e_nao_e_convertida_em_classificacao():
    def transporte(_url, _payload, _timeout):
        return ia_eventos.RespostaHTTP(503, b"modelo indisponivel")

    with pytest.raises(ia_eventos.ErroHTTPollama, match="HTTP 503") as erro:
        ia_eventos.classificar_com_ollama(TEXTO, transporte=transporte)
    assert erro.value.status == 503


def test_envelope_ollama_sem_message_content_e_rejeitado():
    def transporte(_url, _payload, _timeout):
        return ia_eventos.RespostaHTTP(200, b'{"done":true}')

    with pytest.raises(ia_eventos.ErroRespostaOllama, match="message"):
        ia_eventos.classificar_com_ollama(TEXTO, transporte=transporte)
