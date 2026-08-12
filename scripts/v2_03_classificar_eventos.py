"""Classifica o corpus da V2 com um modelo local e saída estruturada."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402
from src.v2 import ia_eventos  # noqa: E402


RAIZ = Path(__file__).resolve().parent.parent
CORPUS_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "corpus_lideres_top20.parquet"
)
CLASSIFICACOES_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "classificacoes_ia.parquet"
)
DIR_CACHE = RAIZ / "data" / "processed" / "cvm_ipe" / "cache_ia"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PADRAO)
    parser.add_argument("--saida", type=Path, default=CLASSIFICACOES_PADRAO)
    parser.add_argument("--modelo", default=ia_eventos.MODELO_DEFAULT)
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL)
    parser.add_argument("--limite", type=int)
    parser.add_argument("--forcar", action="store_true")
    parser.add_argument("--somente-verificar", action="store_true")
    return parser.parse_args()


def _absoluto(caminho: Path) -> Path:
    return caminho if caminho.is_absolute() else RAIZ / caminho


def _json_http(
    url: str,
    *,
    metodo: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: float = 30.0,
) -> object:
    corpo = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    requisicao = urllib.request.Request(
        url,
        data=corpo,
        method=metodo,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=timeout) as resposta:
            bruto = resposta.read()
            status = getattr(resposta, "status", 200)
    except urllib.error.HTTPError as exc:
        detalhe = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Ollama respondeu HTTP {exc.code}: {detalhe}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"não foi possível acessar o Ollama em {url}: {exc}") from exc
    if not 200 <= status < 300:
        raise RuntimeError(f"Ollama respondeu HTTP {status}")
    try:
        return json.loads(bruto.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"resposta inválida do Ollama em {url}") from exc


def identidade_modelo_instalado(
    modelo: str,
    base_url: str = OLLAMA_BASE_URL,
) -> dict[str, object]:
    """Lê a versão do Ollama e o digest do modelo instalado."""
    base = base_url.rstrip("/")
    versao = _json_http(f"{base}/api/version")
    tags = _json_http(f"{base}/api/tags")
    if not isinstance(versao, dict) or not isinstance(versao.get("version"), str):
        raise RuntimeError("Ollama não informou sua versão")
    if not isinstance(tags, dict) or not isinstance(tags.get("models"), list):
        raise RuntimeError("Ollama não retornou a lista de modelos")

    candidatos = [
        item
        for item in tags["models"]
        if isinstance(item, dict)
        and modelo in {item.get("name"), item.get("model")}
    ]
    if len(candidatos) != 1:
        disponiveis = sorted(
            str(item.get("name"))
            for item in tags["models"]
            if isinstance(item, dict)
        )
        raise RuntimeError(
            f"modelo {modelo!r} não está instalado de forma unívoca; "
            f"disponíveis: {disponiveis}"
        )
    item = candidatos[0]
    digest = str(item.get("digest", ""))
    tamanho = item.get("size")
    if not _DIGEST_RE.fullmatch(digest):
        raise RuntimeError("digest do modelo ausente ou inválido")
    if isinstance(tamanho, bool) or not isinstance(tamanho, int) or tamanho <= 0:
        raise RuntimeError("tamanho do modelo ausente ou inválido")
    detalhes = item.get("details") if isinstance(item.get("details"), dict) else {}
    return {
        "modelo": modelo,
        "modelo_digest": digest,
        "modelo_bytes": tamanho,
        "modelo_modificado_em": item.get("modified_at"),
        "modelo_formato": detalhes.get("format"),
        "modelo_familia": detalhes.get("family"),
        "modelo_parametros": detalhes.get("parameter_size"),
        "modelo_quantizacao": detalhes.get("quantization_level"),
        "ollama_versao": versao["version"],
    }


def _chave_cache(
    chave_requisicao: str,
    digest_modelo: str,
    versao_ollama: str,
) -> str:
    return hashlib.sha256(
        f"{chave_requisicao}:{digest_modelo}:{versao_ollama}".encode("ascii")
    ).hexdigest()


def _gravar_json_atomico(caminho: Path, objeto: dict[str, object]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    descritor, nome = tempfile.mkstemp(
        prefix=f".{caminho.name}.", suffix=".tmp", dir=caminho.parent
    )
    temporario = Path(nome)
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
            json.dump(objeto, arquivo, ensure_ascii=False, indent=2)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, caminho)
    except BaseException:
        temporario.unlink(missing_ok=True)
        raise


def _ler_cache(
    caminho: Path,
    chave: str,
    documento: str,
    texto: str,
    identidade_modelo: dict[str, object],
) -> dict[str, object] | None:
    if not caminho.is_file():
        return None
    try:
        objeto = json.loads(caminho.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(objeto, dict):
        return None
    if (
        objeto.get("cache_key") != chave
        or objeto.get("ID_Documento") != documento
        or objeto.get("status_ia") != "ok"
        or objeto.get("modelo_digest") != identidade_modelo["modelo_digest"]
        or objeto.get("ollama_versao") != identidade_modelo["ollama_versao"]
        or objeto.get("prompt_versao") != ia_eventos.PROMPT_VERSAO
        or objeto.get("prompt_hash") != ia_eventos.HASH_PROMPT
        or objeto.get("alvo_direcao") != ia_eventos.ALVO_DIRECAO
        or type(objeto.get("think_suportado")) is not bool
    ):
        return None
    try:
        identidade = ia_eventos.identidade_requisicao(
            texto,
            str(identidade_modelo["modelo"]),
            incluir_think=bool(objeto.get("think_suportado")),
        )
        if (
            objeto.get("request_key") != identidade.chave_cache
            or objeto.get("texto_hash") != identidade.hash_texto
        ):
            return None
        classificacao = ia_eventos.validar_classificacao(
            {
                "especifico_empresa": objeto.get("especifico_empresa"),
                "direcao": objeto.get("direcao"),
                "evidencia": objeto.get("evidencia"),
            },
            texto,
        )
        if (
            objeto.get("abster") is not classificacao.abster
            or objeto.get("motivo_abstencao") != classificacao.motivo_abstencao
        ):
            return None
    except (TypeError, ValueError, ia_eventos.ErroClassificacao):
        return None
    return objeto


def _classificar_um(
    registro: dict[str, object],
    identidade_modelo: dict[str, object],
    *,
    base_url: str,
    forcar: bool,
) -> tuple[dict[str, object], bool]:
    documento = str(registro["ID_Documento"])
    texto = str(registro["texto_llm"])
    modelo = str(identidade_modelo["modelo"])
    if not forcar:
        for incluir_think in (True, False):
            identidade = ia_eventos.identidade_requisicao(
                texto, modelo, incluir_think=incluir_think
            )
            chave = _chave_cache(
                identidade.chave_cache,
                str(identidade_modelo["modelo_digest"]),
                str(identidade_modelo["ollama_versao"]),
            )
            cache = _ler_cache(
                DIR_CACHE / f"{chave}.json",
                chave,
                documento,
                texto,
                identidade_modelo,
            )
            if cache is not None:
                return cache, True

    inicio = time.perf_counter()
    resultado = ia_eventos.classificar_com_ollama(
        texto,
        modelo,
        url=base_url.rstrip("/") + "/api/chat",
        timeout=300.0,
    )
    classificacao = resultado.classificacao
    chave = _chave_cache(
        resultado.identidade.chave_cache,
        str(identidade_modelo["modelo_digest"]),
        str(identidade_modelo["ollama_versao"]),
    )
    caminho_cache = DIR_CACHE / f"{chave}.json"
    linha: dict[str, object] = {
        "ID_Documento": documento,
        "status_ia": "ok",
        "erro_ia": "",
        **identidade_modelo,
        "prompt_versao": ia_eventos.PROMPT_VERSAO,
        "prompt_hash": resultado.identidade.hash_prompt,
        "alvo_direcao": ia_eventos.ALVO_DIRECAO,
        "texto_hash": resultado.identidade.hash_texto,
        "request_key": resultado.identidade.chave_cache,
        "cache_key": chave,
        "think_suportado": resultado.think_suportado,
        **classificacao.como_dict(),
        "resposta_bruta": "",
        **resultado.metricas.como_dict(),
        "tempo_cliente_segundos": time.perf_counter() - inicio,
        "classificado_em_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    _gravar_json_atomico(caminho_cache, linha)
    return linha, False


def main() -> int:
    args = _argumentos()
    if args.limite is not None and args.limite < 1:
        raise SystemExit("--limite deve ser positivo")
    corpus_path = _absoluto(args.corpus)
    saida_path = _absoluto(args.saida)
    if (
        args.limite is not None
        and saida_path.resolve() == CLASSIFICACOES_PADRAO.resolve()
    ):
        raise SystemExit("--limite exige --saida diferente da saida oficial")

    execucao = criar_execucao("v2_ia")
    log = configurar_log(execucao, "v2_03_ia")
    cfg_manifesto = {
        "modelo": args.modelo,
        "ollama_url": args.ollama_url,
        "prompt_versao": ia_eventos.PROMPT_VERSAO,
        "prompt_hash": ia_eventos.HASH_PROMPT,
        "alvo_direcao": ia_eventos.ALVO_DIRECAO,
        "temperature": 0,
        "seed": ia_eventos.SEED_FIXA,
        "num_ctx": ia_eventos.NUM_CTX,
        "num_predict": ia_eventos.NUM_PREDICT,
        "limite_debug": args.limite,
        "forcar": args.forcar,
        "somente_verificar": args.somente_verificar,
    }
    try:
        identidade_modelo = identidade_modelo_instalado(
            args.modelo, args.ollama_url
        )
        cfg_manifesto.update(identidade_modelo)
        log.info(
            "Ollama %s | %s | %s | %s",
            identidade_modelo["ollama_versao"],
            args.modelo,
            identidade_modelo["modelo_quantizacao"],
            str(identidade_modelo["modelo_digest"])[:12],
        )
        if args.somente_verificar:
            gravar_manifesto(
                execucao,
                cfg_manifesto,
                arquivos_dados=[],
                status="concluida",
                extras={"servidor_e_modelo": "ok"},
            )
            return 0

        corpus = pd.read_parquet(corpus_path)
        requeridas = {"ID_Documento", "status_documento", "texto_llm"}
        faltantes = sorted(requeridas - set(corpus.columns))
        if faltantes:
            raise ValueError(f"corpus sem colunas: {faltantes}")
        elegiveis = corpus.loc[
            corpus["status_documento"].eq("ok")
            & corpus["texto_llm"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        elegiveis = elegiveis.sort_values("ID_Documento").reset_index(drop=True)
        if args.limite is not None:
            elegiveis = elegiveis.head(args.limite)
        if elegiveis.empty:
            raise ValueError("nenhum documento elegível no corpus")

        linhas: list[dict[str, object]] = []
        erros: list[dict[str, object]] = []
        hits = 0
        for numero, registro in enumerate(elegiveis.to_dict("records"), start=1):
            try:
                linha, do_cache = _classificar_um(
                    registro,
                    identidade_modelo,
                    base_url=args.ollama_url,
                    forcar=args.forcar,
                )
                linhas.append(linha)
                hits += int(do_cache)
            except ia_eventos.ErroClassificacao as exc:
                erros.append(
                    {
                        "ID_Documento": registro["ID_Documento"],
                        "status_ia": "erro",
                        "erro_ia": f"{type(exc).__name__}: {exc}",
                        "especifico_empresa": pd.NA,
                        "direcao": pd.NA,
                        "evidencia": "",
                        "abster": pd.NA,
                        "motivo_abstencao": "",
                        "resposta_bruta": getattr(exc, "resposta_json", ""),
                    }
                )
                log.error("%s | %s", registro["ID_Documento"], exc)
            except (ia_eventos.ErroHTTPollama, ia_eventos.ErroRespostaOllama):
                raise
            if numero % 10 == 0 or numero == len(elegiveis):
                log.info(
                    "classificados: %d/%d | cache %d | erros %d",
                    numero,
                    len(elegiveis),
                    hits,
                    len(erros),
                )

        classificacoes = pd.DataFrame([*linhas, *erros])
        metadados = elegiveis.drop(columns=["texto", "texto_llm"], errors="ignore")
        saida = metadados.merge(
            classificacoes, on="ID_Documento", how="left", validate="one_to_one"
        )
        saida_path.parent.mkdir(parents=True, exist_ok=True)
        saida.to_parquet(saida_path, index=False)

        resumo = (
            saida["direcao"].fillna("erro").value_counts().rename_axis("direcao")
            .reset_index(name="documentos")
        )
        resumo.to_csv(
            execucao.tabelas / "ia_resumo_direcao.csv", index=False, encoding="utf-8"
        )
        if erros:
            pd.DataFrame(erros).to_csv(
                execucao.tabelas / "ia_erros.csv", index=False, encoding="utf-8"
            )
        gravar_manifesto(
            execucao,
            cfg_manifesto,
            arquivos_dados=[corpus_path],
            status="concluida" if not erros else "concluida_com_erros",
            extras={
                "arquivo_processado": str(saida_path),
                "documentos_elegiveis": int(len(elegiveis)),
                "classificados_ok": int(len(linhas)),
                "erros": int(len(erros)),
                "cache_hits": int(hits),
            },
        )
        return 0 if not erros else 2
    except Exception as exc:
        log.exception("classificação por IA falhou: %s", exc)
        gravar_manifesto(
            execucao,
            cfg_manifesto,
            arquivos_dados=[corpus_path] if corpus_path.exists() else [],
            status="falhou",
            extras={"erro": str(exc)},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
