from __future__ import annotations

import hashlib
import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.v2 import documentos


def _escapar_literal_pdf(texto: str) -> bytes:
    return (
        texto.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .encode("cp1252")
    )


def _pdf_sintetico(paginas: list[list[str] | None], senha: str | None = None) -> bytes:
    """Cria PDF mínimo com fonte Type1 e camada textual controlada."""
    writer = PdfWriter()
    for linhas in paginas:
        pagina = writer.add_blank_page(width=612, height=792)
        if linhas is None:
            continue

        fonte = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            }
        )
        recursos = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(fonte)}
                )
            }
        )
        pagina[NameObject("/Resources")] = recursos

        comandos = [b"BT", b"/F1 12 Tf", b"72 720 Td"]
        for indice, linha in enumerate(linhas):
            if indice:
                comandos.append(b"0 -18 Td")
            comandos.append(b"(" + _escapar_literal_pdf(linha) + b") Tj")
        comandos.append(b"ET")
        stream = DecodedStreamObject()
        stream.set_data(b"\n".join(comandos))
        pagina[NameObject("/Contents")] = writer._add_object(stream)

    if senha is not None:
        writer.encrypt(senha)
    saida = io.BytesIO()
    writer.write(saida)
    return saida.getvalue()


def test_extrai_bytes_com_hash_paginas_texto_normalizado_e_status_ok():
    pdf = _pdf_sintetico(
        [
            ["Primeira   linha", "Segunda linha"],
            None,
            ["Última página com texto suficiente para concluir o documento."],
        ]
    )

    resultado = documentos.extrair_texto_pdf(pdf, min_caracteres=20)

    assert resultado.sha256 == hashlib.sha256(pdf).hexdigest()
    assert resultado.paginas_total == 3
    assert resultado.paginas_com_texto == 2
    assert resultado.status == documentos.STATUS_OK
    assert resultado.caracteres == len(resultado.texto)
    assert "Primeira linha" in resultado.texto
    assert "Segunda linha" in resultado.texto
    assert "Última página" in resultado.texto
    assert resultado.ocr_aplicado is False


def test_extracao_por_path_e_identica_a_de_bytes(tmp_path):
    pdf = _pdf_sintetico([["Documento no disco"]])
    caminho = tmp_path / "documento.pdf"
    caminho.write_bytes(pdf)

    por_bytes = documentos.extrair_texto_pdf(pdf, min_caracteres=1)
    por_path = documentos.extrair_texto_pdf(caminho, min_caracteres=1)

    assert por_path == por_bytes


def test_pdf_sem_camada_textual_nao_dispara_ocr():
    resultado = documentos.extrair_texto_pdf(
        _pdf_sintetico([None, None]), min_caracteres=1
    )

    assert resultado.status == documentos.STATUS_SEM_TEXTO
    assert resultado.paginas_total == 2
    assert resultado.paginas_com_texto == 0
    assert resultado.texto == ""
    assert resultado.caracteres == 0
    assert resultado.ocr_aplicado is False


def test_texto_curto_recebe_status_texto_insuficiente():
    resultado = documentos.extrair_texto_pdf(
        _pdf_sintetico([["Aviso curto"]]), min_caracteres=100
    )

    assert resultado.status == documentos.STATUS_TEXTO_INSUFICIENTE
    assert resultado.paginas_com_texto == 1
    assert 0 < resultado.caracteres < 100


def test_magic_pdf_e_obrigatorio_mesmo_se_extensao_for_pdf(tmp_path):
    caminho = tmp_path / "falso.pdf"
    caminho.write_bytes(b"<html>login</html>")

    with pytest.raises(documentos.PDFInvalido, match="%PDF-"):
        documentos.extrair_texto_pdf(caminho)


def test_magic_sem_estrutura_pdf_produz_erro_claro():
    with pytest.raises(documentos.PDFInvalido, match="pypdf"):
        documentos.extrair_texto_pdf(b"%PDF-1.7\nisto nao e um PDF")


def test_pdf_criptografado_exige_tratamento_explicito():
    pdf = _pdf_sintetico([["Texto protegido"]], senha="segredo")

    with pytest.raises(documentos.PDFCriptografado, match="senha"):
        documentos.extrair_texto_pdf(pdf)


def test_normalizacao_e_deterministica_e_preserva_paragrafos():
    bruto = "  Café\u00a0com\t espaços  \r\n\r\n\r\n  Segundo   parágrafo\x00  "

    assert documentos.normalizar_texto(bruto) == (
        "Café com espaços\n\nSegundo parágrafo"
    )


def test_preparacao_curta_so_normaliza_sem_marcar_truncamento():
    preparado = documentos.preparar_para_llm(
        "  começo   e fim  ", limite_caracteres=80
    )

    assert preparado == "começo e fim"
    assert documentos.MARCADOR_TRUNCAMENTO not in preparado


def test_preparacao_truncada_preserva_inicio_fim_e_respeita_limite():
    texto = "INICIO-" + ("0123456789" * 30) + "-FIM"
    limite = 100

    primeiro = documentos.preparar_para_llm(texto, limite_caracteres=limite)
    segundo = documentos.preparar_para_llm(texto, limite_caracteres=limite)
    inicio, fim = primeiro.split(documentos.MARCADOR_TRUNCAMENTO)

    assert primeiro == segundo
    assert len(primeiro) == limite
    assert primeiro.startswith("INICIO-")
    assert primeiro.endswith("-FIM")
    assert inicio == texto[: len(inicio)]
    assert fim == texto[-len(fim) :]


def test_preparacao_aceita_resultado_da_extracao():
    resultado = documentos.extrair_texto_pdf(
        _pdf_sintetico([["Texto para o modelo"]]), min_caracteres=1
    )

    assert documentos.preparar_para_llm(
        resultado, limite_caracteres=80
    ) == resultado.texto


def test_limite_precisa_comportar_inicio_marcador_e_fim():
    limite_invalido = len(documentos.MARCADOR_TRUNCAMENTO) + 1

    with pytest.raises(ValueError, match="pelo menos"):
        documentos.preparar_para_llm(
            "texto longo" * 20, limite_caracteres=limite_invalido
        )
