"""Extração auditável da camada textual de documentos PDF."""

from __future__ import annotations

import hashlib
import io
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


STATUS_OK = "ok"
STATUS_TEXTO_INSUFICIENTE = "texto_insuficiente"
STATUS_SEM_TEXTO = "sem_texto"
STATUS_VALIDOS = frozenset(
    {STATUS_OK, STATUS_TEXTO_INSUFICIENTE, STATUS_SEM_TEXTO}
)

MIN_CARACTERES_TEXTO = 200
LIMITE_CARACTERES_LLM = 24_000
MARCADOR_TRUNCAMENTO = "\n\n[... TRECHO CENTRAL TRUNCADO ...]\n\n"


class ErroDocumento(RuntimeError):
    """Erro ao ler ou interpretar um documento."""


class PDFInvalido(ErroDocumento):
    """O conteúdo não é um PDF legível."""


class PDFCriptografado(ErroDocumento):
    """O PDF exige uma senha não fornecida pelo repositório público."""


class ErroExtracaoPDF(ErroDocumento):
    """A camada textual de uma página não pôde ser extraída."""


@dataclass(frozen=True)
class ResultadoExtracao:
    """Texto e contagens da extração."""

    sha256: str
    paginas_total: int
    paginas_com_texto: int
    texto: str
    caracteres: int
    status: str

    @property
    def ocr_aplicado(self) -> bool:
        return False


def _ler_bytes(origem: bytes | bytearray | memoryview | str | os.PathLike[str]) -> bytes:
    if isinstance(origem, (bytes, bytearray, memoryview)):
        return bytes(origem)
    if isinstance(origem, (str, os.PathLike)):
        caminho = Path(origem)
        try:
            return caminho.read_bytes()
        except OSError as exc:
            raise ErroDocumento(f"não foi possível ler {caminho}: {exc}") from exc
    raise TypeError("origem deve ser bytes ou Path")


def _validar_magic_pdf(conteudo: bytes) -> None:
    # Alguns PDFs têm lixo antes do cabeçalho.
    if b"%PDF-" not in conteudo[:1024]:
        raise PDFInvalido("assinatura %PDF- ausente no primeiro KiB")


def normalizar_texto(texto: str) -> str:
    """Normaliza Unicode, quebras de linha e espaços."""
    if not isinstance(texto, str):
        raise TypeError("texto deve ser str")
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = texto.replace("\x00", "").replace("\f", "\n")
    linhas = [re.sub(r"[\t\v \u00a0]+", " ", linha).strip() for linha in texto.split("\n")]
    texto = "\n".join(linhas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _abrir_reader(conteudo: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(conteudo), strict=False)
    except Exception as exc:
        raise PDFInvalido(f"pypdf não conseguiu abrir o documento: {exc}") from exc

    if reader.is_encrypted:
        try:
            desbloqueado = reader.decrypt("")
        except Exception as exc:
            raise PDFCriptografado(
                f"PDF criptografado não aceita senha vazia: {exc}"
            ) from exc
        if not desbloqueado:
            raise PDFCriptografado("PDF criptografado exige senha")
    return reader


def extrair_texto_pdf(
    origem: bytes | bytearray | memoryview | str | os.PathLike[str],
    *,
    min_caracteres: int = MIN_CARACTERES_TEXTO,
) -> ResultadoExtracao:
    """Extrai a camada de texto do PDF. Scan sem texto fica para revisão."""
    if isinstance(min_caracteres, bool) or not isinstance(min_caracteres, int):
        raise TypeError("min_caracteres deve ser inteiro")
    if min_caracteres < 1:
        raise ValueError("min_caracteres deve ser positivo")

    conteudo = _ler_bytes(origem)
    _validar_magic_pdf(conteudo)
    sha256 = hashlib.sha256(conteudo).hexdigest()
    reader = _abrir_reader(conteudo)

    paginas: list[str] = []
    paginas_com_texto = 0
    for numero, pagina in enumerate(reader.pages, start=1):
        try:
            bruto = pagina.extract_text() or ""
        except Exception as exc:
            raise ErroExtracaoPDF(
                f"falha ao extrair a camada textual da página {numero}: {exc}"
            ) from exc
        normalizado = normalizar_texto(bruto)
        if normalizado:
            paginas_com_texto += 1
            paginas.append(normalizado)

    texto = normalizar_texto("\n\n".join(paginas))
    if not texto:
        status = STATUS_SEM_TEXTO
    elif len(texto) < min_caracteres:
        status = STATUS_TEXTO_INSUFICIENTE
    else:
        status = STATUS_OK

    return ResultadoExtracao(
        sha256=sha256,
        paginas_total=len(reader.pages),
        paginas_com_texto=paginas_com_texto,
        texto=texto,
        caracteres=len(texto),
        status=status,
    )


def preparar_para_llm(
    documento: ResultadoExtracao | str,
    *,
    limite_caracteres: int,
) -> str:
    """Normaliza e limita o texto, preservando início e fim em partes iguais."""
    if isinstance(limite_caracteres, bool) or not isinstance(limite_caracteres, int):
        raise TypeError("limite_caracteres deve ser inteiro")
    minimo = len(MARCADOR_TRUNCAMENTO) + 2
    if limite_caracteres < minimo:
        raise ValueError(
            f"limite_caracteres deve ser pelo menos {minimo} para preservar "
            "início, marcador e fim"
        )

    texto = documento.texto if isinstance(documento, ResultadoExtracao) else documento
    texto = normalizar_texto(texto)
    if len(texto) <= limite_caracteres:
        return texto

    disponivel = limite_caracteres - len(MARCADOR_TRUNCAMENTO)
    caracteres_inicio = (disponivel + 1) // 2
    caracteres_fim = disponivel - caracteres_inicio
    return (
        texto[:caracteres_inicio]
        + MARCADOR_TRUNCAMENTO
        + texto[-caracteres_fim:]
    )
