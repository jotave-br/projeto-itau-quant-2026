"""Coleta point-in-time dos documentos IPE publicados pela CVM."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pandas as pd


URL_IPE_ANUAL = (
    "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/"
    "ipe_cia_aberta_{ano}.zip"
)

COLUNAS_IPE: tuple[str, ...] = (
    "CNPJ_Companhia",
    "Nome_Companhia",
    "Codigo_CVM",
    "Data_Referencia",
    "Categoria",
    "Tipo",
    "Especie",
    "Assunto",
    "Data_Entrega",
    "Tipo_Apresentacao",
    "Protocolo_Entrega",
    "Versao",
    "Link_Download",
)
COLUNAS_DATA = ("Data_Referencia", "Data_Entrega")
CATEGORIA_FATO_RELEVANTE = "Fato Relevante"
COLUNA_ID = "ID_Documento"
COLUNA_SESSAO = "Sessao_Disponivel"

RAIZ = Path(__file__).resolve().parents[2]
DIR_ZIPS = RAIZ / "data" / "raw" / "cvm_ipe" / "zips"
DIR_DOCUMENTOS = RAIZ / "data" / "raw" / "cvm_ipe" / "documentos"
_USER_AGENT = "itau-quant-project/2.0 (pesquisa academica)"


class ErroIPE(RuntimeError):
    """Erro de coleta ou validacao do conjunto IPE."""


class ErroSchemaIPE(ErroIPE):
    """O CSV nao segue o contrato publicado pela CVM."""


class ErroDownloadIPE(ErroIPE):
    """A resposta HTTP ou o artefato baixado e invalido."""


class ErroDocumentoIPE(ErroDownloadIPE):
    """O documento recebido nao tem o formato esperado."""


@dataclass(frozen=True)
class RespostaHTTP:
    """Resposta usada pelo downloader."""

    conteudo: bytes
    content_type: str | None = None
    status: int = 200
    url_final: str | None = None


ClienteHTTP = Callable[[str], RespostaHTTP | bytes]


@dataclass(frozen=True)
class ArtefatoBaixado:
    """Metadados do arquivo baixado."""

    caminho: Path
    url: str
    sha256: str
    tamanho: int
    content_type: str | None
    de_cache: bool
    eh_pdf: bool | None = None


def url_ipe_ano(ano: int) -> str:
    """Monta a URL oficial do ZIP pelo ano de entrega."""
    if isinstance(ano, bool) or not isinstance(ano, int) or not 2003 <= ano <= 9999:
        raise ValueError("ano do IPE deve ser inteiro entre 2003 e 9999")
    return URL_IPE_ANUAL.format(ano=ano)


def _origem_zip(
    origem: bytes | bytearray | memoryview | str | os.PathLike[str] | BinaryIO,
):
    if isinstance(origem, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(origem))
    if isinstance(origem, (str, os.PathLike)):
        return Path(origem)
    if hasattr(origem, "read") and hasattr(origem, "seek"):
        origem.seek(0)
        return origem
    raise TypeError("origem deve ser bytes, arquivo em memoria ou Path")


def validar_dataframe_ipe(df: pd.DataFrame) -> None:
    """Exige as 13 colunas oficiais, na ordem publicada, e datas sem hora."""
    encontradas = tuple(str(c) for c in df.columns)
    if encontradas != COLUNAS_IPE:
        faltantes = [c for c in COLUNAS_IPE if c not in encontradas]
        extras = [c for c in encontradas if c not in COLUNAS_IPE]
        detalhe = []
        if faltantes:
            detalhe.append(f"faltantes={faltantes}")
        if extras:
            detalhe.append(f"extras={extras}")
        if not faltantes and not extras:
            detalhe.append("ordem das colunas diverge")
        raise ErroSchemaIPE("schema IPE invalido: " + "; ".join(detalhe))

    for coluna in COLUNAS_DATA:
        valores = df[coluna].astype("string")
        formato = valores.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        if coluna == "Data_Entrega":
            # Data_Referencia tem um ano 5019 no arquivo oficial. Só a data de
            # entrega precisa caber no calendário do backtest.
            datas_validas = pd.to_datetime(
                valores.where(formato), format="%Y-%m-%d", errors="coerce"
            ).notna()
        else:
            def data_iso_valida(valor: object) -> bool:
                try:
                    date.fromisoformat(str(valor))
                    return True
                except (TypeError, ValueError):
                    return False

            datas_validas = valores.map(data_iso_valida)
        invalidas = ~formato | ~datas_validas
        if invalidas.any():
            amostra = valores[invalidas].head(3).tolist()
            raise ErroSchemaIPE(
                f"{coluna} deve ser data real YYYY-MM-DD, sem hora; "
                f"valores invalidos: {amostra}"
            )


def ler_zip_ipe(
    origem: bytes | bytearray | memoryview | str | os.PathLike[str] | BinaryIO,
) -> pd.DataFrame:
    """Le o unico CSV do ZIP, em cp1252, sem alterar os campos recebidos."""
    fonte = _origem_zip(origem)
    try:
        with zipfile.ZipFile(fonte) as arquivo:
            csvs = [
                item
                for item in arquivo.infolist()
                if not item.is_dir() and item.filename.lower().endswith(".csv")
            ]
            if len(csvs) != 1:
                raise ErroSchemaIPE(
                    f"ZIP IPE deve conter um CSV; encontrados {len(csvs)}"
                )
            with arquivo.open(csvs[0]) as csv_bruto:
                df = pd.read_csv(
                    csv_bruto,
                    sep=";",
                    encoding="cp1252",
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                )
    except zipfile.BadZipFile as exc:
        raise ErroSchemaIPE("artefato IPE nao e um ZIP valido") from exc
    except UnicodeDecodeError as exc:
        raise ErroSchemaIPE("CSV IPE nao pode ser decodificado como cp1252") from exc
    except pd.errors.ParserError as exc:
        raise ErroSchemaIPE("CSV IPE malformado") from exc

    validar_dataframe_ipe(df)
    return df


def filtrar_fatos_relevantes(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona a categoria exata sem agrupar ou descartar versoes."""
    validar_dataframe_ipe(df)
    return df.loc[df["Categoria"].eq(CATEGORIA_FATO_RELEVANTE)].copy()


def id_documento(registro: Mapping[str, object]) -> str:
    """Hash estavel dos 13 campos brutos; versoes distintas recebem IDs distintos."""
    faltantes = [c for c in COLUNAS_IPE if c not in registro]
    if faltantes:
        raise ErroSchemaIPE(f"campos ausentes para formar ID: {faltantes}")

    valores: list[str] = []
    for coluna in COLUNAS_IPE:
        valor = registro[coluna]
        valores.append("" if pd.isna(valor) else str(valor))
    canonico = json.dumps(
        valores, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonico).hexdigest()


def adicionar_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta o ID deterministico sem mudar a ordem das linhas."""
    validar_dataframe_ipe(df)
    saida = df.copy()
    saida[COLUNA_ID] = [id_documento(registro) for registro in df.to_dict("records")]
    return saida


def carregar_fatos_relevantes(
    origem: bytes | bytearray | memoryview | str | os.PathLike[str] | BinaryIO,
) -> pd.DataFrame:
    """Le, valida, filtra Fatos Relevantes e acrescenta o ID."""
    return adicionar_ids(filtrar_fatos_relevantes(ler_zip_ipe(origem)))


def _calendario_valido(calendario: Sequence[object] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(calendario)
    if cal.tz is not None:
        raise ValueError("calendario da B3 deve conter datas sem timezone")
    if cal.hasnans:
        raise ValueError("calendario da B3 contem NaT")
    cal = cal.normalize().unique().sort_values()
    if cal.empty:
        raise ValueError("calendario da B3 esta vazio")
    return cal


def proxima_sessao_b3(
    data_entrega: str | date | datetime | pd.Timestamp,
    calendario: Sequence[object] | pd.DatetimeIndex,
) -> pd.Timestamp:
    """Primeira sessao do calendario estritamente posterior a entrega."""
    entrega = pd.Timestamp(data_entrega)
    if entrega.tz is not None or entrega != entrega.normalize():
        raise ValueError("data_entrega deve ser data sem hora ou timezone")
    cal = _calendario_valido(calendario)
    posicao = int(cal.searchsorted(entrega, side="right"))
    if posicao == len(cal):
        raise ValueError(f"calendario nao cobre sessao posterior a {entrega.date()}")
    return cal[posicao]


def adicionar_sessao_disponivel(
    df: pd.DataFrame,
    calendario: Sequence[object] | pd.DatetimeIndex,
    coluna_saida: str = COLUNA_SESSAO,
) -> pd.DataFrame:
    """Define como disponível a primeira sessão após a entrega."""
    if "Data_Entrega" not in df:
        raise ErroSchemaIPE("Data_Entrega ausente")
    if coluna_saida in df:
        raise ValueError(f"coluna {coluna_saida!r} ja existe")

    valores = df["Data_Entrega"].astype("string")
    formato = valores.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    entregas = pd.to_datetime(
        valores.where(formato), format="%Y-%m-%d", errors="coerce"
    )
    invalidas = ~formato | entregas.isna()
    if invalidas.any():
        raise ErroSchemaIPE("Data_Entrega invalida para mapear a sessao")

    cal = _calendario_valido(calendario)
    posicoes = cal.searchsorted(pd.DatetimeIndex(entregas), side="right")
    sem_cobertura = posicoes == len(cal)
    if sem_cobertura.any():
        datas = sorted(set(valores[sem_cobertura].tolist()))
        raise ValueError(f"calendario nao cobre entregas: {datas[:3]}")

    saida = df.copy()
    saida[coluna_saida] = cal.take(posicoes)
    return saida


def _sha256_bytes(conteudo: bytes) -> str:
    return hashlib.sha256(conteudo).hexdigest()


def _sha256_path(caminho: Path, tamanho_bloco: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(tamanho_bloco), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _meta_path(destino: Path) -> Path:
    return destino.with_suffix(destino.suffix + ".meta.json")


def _gravar_atomico(destino: Path, conteudo: bytes) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent
    )
    caminho_temporario = Path(temporario)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(conteudo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(caminho_temporario, destino)
    except BaseException:
        caminho_temporario.unlink(missing_ok=True)
        raise


def _parece_pdf_bytes(conteudo: bytes) -> bool:
    return conteudo[:1024].lstrip().startswith(b"%PDF-")


def _parece_pdf_path(caminho: Path) -> bool:
    with caminho.open("rb") as arquivo:
        return _parece_pdf_bytes(arquivo.read(1024))


def _resposta_padrao(url: str, timeout: float) -> RespostaHTTP:
    requisicao = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(requisicao, timeout=timeout) as resposta:
            conteudo = resposta.read()
            return RespostaHTTP(
                conteudo=conteudo,
                content_type=resposta.headers.get("Content-Type"),
                status=getattr(resposta, "status", 200),
                url_final=resposta.geturl(),
            )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ErroDownloadIPE(f"falha ao baixar {url}: {exc}") from exc


def _obter_resposta(
    url: str, cliente_http: ClienteHTTP | None, timeout: float
) -> RespostaHTTP:
    try:
        resposta = cliente_http(url) if cliente_http else _resposta_padrao(url, timeout)
    except ErroIPE:
        raise
    except Exception as exc:
        raise ErroDownloadIPE(f"falha ao baixar {url}: {exc}") from exc

    if isinstance(resposta, (bytes, bytearray, memoryview)):
        resposta = RespostaHTTP(bytes(resposta))
    if not isinstance(resposta, RespostaHTTP):
        raise TypeError("cliente_http deve devolver RespostaHTTP ou bytes")
    if not 200 <= resposta.status < 300:
        raise ErroDownloadIPE(f"HTTP {resposta.status} ao baixar {url}")
    if not resposta.conteudo:
        raise ErroDownloadIPE(f"resposta vazia ao baixar {url}")
    return resposta


def _ler_cache(
    destino: Path, url: str, tipo: str, exigir_pdf: bool
) -> ArtefatoBaixado | None:
    meta_path = _meta_path(destino)
    if not destino.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
        tamanho = destino.stat().st_size
        sha256 = _sha256_path(destino)
    except (OSError, json.JSONDecodeError):
        return None

    if (
        meta.get("url") != url
        or meta.get("tipo") != tipo
        or meta.get("sha256") != sha256
        or meta.get("bytes") != tamanho
    ):
        return None
    eh_pdf = _parece_pdf_path(destino) if tipo == "documento" else None
    if exigir_pdf and not eh_pdf:
        return None
    return ArtefatoBaixado(
        caminho=destino,
        url=url,
        sha256=sha256,
        tamanho=tamanho,
        content_type=meta.get("content_type"),
        de_cache=True,
        eh_pdf=eh_pdf,
    )


def _salvar_download(
    destino: Path,
    url: str,
    resposta: RespostaHTTP,
    tipo: str,
    eh_pdf: bool | None,
) -> ArtefatoBaixado:
    sha256 = _sha256_bytes(resposta.conteudo)
    meta = {
        "url": url,
        "url_final": resposta.url_final,
        "baixado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo": tipo,
        "content_type": resposta.content_type,
        "bytes": len(resposta.conteudo),
        "sha256": sha256,
        "eh_pdf": eh_pdf,
    }
    _gravar_atomico(destino, resposta.conteudo)
    _gravar_atomico(
        _meta_path(destino),
        json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return ArtefatoBaixado(
        caminho=destino,
        url=url,
        sha256=sha256,
        tamanho=len(resposta.conteudo),
        content_type=resposta.content_type,
        de_cache=False,
        eh_pdf=eh_pdf,
    )


def baixar_zip_ipe(
    ano: int,
    diretorio_cache: str | os.PathLike[str] = DIR_ZIPS,
    *,
    cliente_http: ClienteHTTP | None = None,
    forcar: bool = False,
    timeout: float = 120.0,
) -> ArtefatoBaixado:
    """Baixa e valida o ZIP anual; cache valido nao consulta a rede."""
    url = url_ipe_ano(ano)
    destino = Path(diretorio_cache) / f"ipe_cia_aberta_{ano}.zip"
    if not forcar:
        cache = _ler_cache(destino, url, tipo="zip", exigir_pdf=False)
        if cache is not None:
            return cache

    resposta = _obter_resposta(url, cliente_http, timeout)
    # A CVM às vezes devolve uma página de erro com status 200.
    ler_zip_ipe(resposta.conteudo)
    return _salvar_download(destino, url, resposta, tipo="zip", eh_pdf=None)


def baixar_documento_ipe(
    url: str,
    destino: str | os.PathLike[str],
    *,
    cliente_http: ClienteHTTP | None = None,
    forcar: bool = False,
    exigir_pdf: bool = True,
    timeout: float = 120.0,
) -> ArtefatoBaixado:
    """Baixa o documento e valida a assinatura do PDF."""
    caminho = Path(destino)
    if not forcar:
        cache = _ler_cache(caminho, url, tipo="documento", exigir_pdf=exigir_pdf)
        if cache is not None:
            return cache

    resposta = _obter_resposta(url, cliente_http, timeout)
    eh_pdf = _parece_pdf_bytes(resposta.conteudo)
    if exigir_pdf and not eh_pdf:
        raise ErroDocumentoIPE(
            "documento nao contem assinatura %PDF-, independentemente do MIME "
            f"{resposta.content_type!r}"
        )
    return _salvar_download(
        caminho, url, resposta, tipo="documento", eh_pdf=eh_pdf
    )
