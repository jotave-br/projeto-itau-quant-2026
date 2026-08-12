"""Seleciona, baixa e extrai os Fatos Relevantes elegíveis da V2."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.backtest import Janela  # noqa: E402
from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402
from src.v2 import cvm_ipe, documentos, eventos  # noqa: E402


RAIZ = Path(__file__).resolve().parent.parent
FATOS_PROCESSADOS = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "fatos_relevantes_lideres.parquet"
)
CALENDARIO_PADRAO = RAIZ / "data" / "processed" / "precos_ajustados.parquet"
CORPUS_PROCESSADO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "corpus_lideres_top20.parquet"
)
RUN_V1_PADRAO = RAIZ / "outputs" / "runs" / "2026-08-07_025233_oficial"


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-v1", type=Path, default=RUN_V1_PADRAO)
    parser.add_argument("--fatos", type=Path, default=FATOS_PROCESSADOS)
    parser.add_argument("--calendario", type=Path, default=CALENDARIO_PADRAO)
    parser.add_argument("--saida", type=Path, default=CORPUS_PROCESSADO)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limite", type=int)
    parser.add_argument("--forcar", action="store_true")
    parser.add_argument(
        "--somente-selecao",
        action="store_true",
        help="valida o funil sem baixar os documentos",
    )
    return parser.parse_args()


def _caminho_absoluto(caminho: Path) -> Path:
    return caminho if caminho.is_absolute() else RAIZ / caminho


def _carregar_calendario(caminho: Path) -> pd.DatetimeIndex:
    if not caminho.is_file():
        raise FileNotFoundError(f"painel usado como calendário não encontrado: {caminho}")
    painel = pd.read_parquet(caminho)
    calendario = pd.DatetimeIndex(painel.index)
    if calendario.tz is not None or calendario.hasnans:
        raise ValueError("calendário deve conter datas válidas sem timezone")
    calendario = calendario.normalize().unique().sort_values()
    if calendario.empty or calendario.has_duplicates:
        raise ValueError("calendário vazio ou duplicado")
    return calendario


def _janelas_dos_pares(pares: pd.DataFrame) -> list[Janela]:
    if "janela" not in pares:
        raise ValueError("pares V1 sem coluna janela")
    rotulos = sorted(pares["janela"].dropna().astype(str).str.strip().unique())
    if not rotulos:
        raise ValueError("pares V1 sem janelas")

    janelas: list[Janela] = []
    for indice, rotulo in enumerate(rotulos):
        try:
            inicio = pd.to_datetime(rotulo + "-01", format="%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rótulo de janela inválido: {rotulo!r}") from exc
        if inicio.strftime("%Y-%m") != rotulo:
            raise ValueError(f"rótulo de janela inválido: {rotulo!r}")
        fim = inicio + pd.DateOffset(months=3)
        janelas.append(
            Janela(
                indice=indice,
                treino_inicio=inicio - pd.DateOffset(months=24),
                treino_fim=inicio,
                teste_inicio=inicio,
                teste_fim=fim,
            )
        )
    return janelas


def _diagnostico_excluidos(
    todos: pd.DataFrame, aceitos: pd.DataFrame, motivo: str
) -> pd.DataFrame:
    if cvm_ipe.COLUNA_ID not in todos or cvm_ipe.COLUNA_ID not in aceitos:
        raise ValueError("ID_Documento ausente ao montar diagnóstico")
    ids = set(aceitos[cvm_ipe.COLUNA_ID])
    excluidos = todos.loc[~todos[cvm_ipe.COLUNA_ID].isin(ids)].copy()
    excluidos[eventos.COLUNA_MOTIVO_DIAGNOSTICO] = motivo
    return excluidos


def selecionar_corpus(
    fatos: pd.DataFrame,
    pares: pd.DataFrame,
    calendario: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Filtra os documentos que entram na V2."""
    etapas: list[dict[str, object]] = []
    diagnosticos: list[pd.DataFrame] = []

    originais = eventos.selecionar_apresentacoes_originais(fatos)
    diagnosticos.append(
        _diagnostico_excluidos(
            fatos, originais, "reapresentacao_ou_versao_nao_original"
        )
    )
    etapas.append({"etapa": "metadados_lideres", "documentos": len(fatos)})
    etapas.append({"etapa": "apresentacao_original", "documentos": len(originais)})

    sessoes = eventos.mapear_sessoes_disponiveis(originais, calendario)
    diagnosticos.append(sessoes.diagnosticos)
    etapas.append({"etapa": "sessao_b3_coberta", "documentos": len(sessoes.eventos)})

    por_janela = eventos.atribuir_janelas_teste(
        sessoes.eventos, _janelas_dos_pares(pares)
    )
    diagnosticos.append(por_janela.diagnosticos)
    etapas.append({"etapa": "dentro_de_janela_v1", "documentos": len(por_janela.eventos)})

    de_lideres = eventos.filtrar_eventos_lideres_top20(por_janela.eventos, pares)
    diagnosticos.append(de_lideres.diagnosticos)
    etapas.append({"etapa": "lider_top20_na_janela", "documentos": len(de_lideres.eventos)})

    selecionados = de_lideres.eventos.sort_values(
        ["Sessao_Disponivel", "emissor_id", cvm_ipe.COLUNA_ID]
    ).reset_index(drop=True)
    if selecionados[cvm_ipe.COLUNA_ID].duplicated().any():
        raise ValueError("documento duplicado após o funil de elegibilidade")

    diagnostico = pd.concat(
        [df for df in diagnosticos if not df.empty], ignore_index=True, sort=False
    ) if any(not df.empty for df in diagnosticos) else pd.DataFrame()
    return selecionados, diagnostico, etapas


def _processar_documento(
    registro: dict[str, object], *, forcar: bool
) -> dict[str, object]:
    identidade = str(registro[cvm_ipe.COLUNA_ID])
    destino = cvm_ipe.DIR_DOCUMENTOS / f"{identidade}.pdf"
    base = {
        cvm_ipe.COLUNA_ID: identidade,
        "caminho_pdf": str(destino.relative_to(RAIZ)),
    }
    try:
        artefato = cvm_ipe.baixar_documento_ipe(
            str(registro["Link_Download"]), destino, forcar=forcar
        )
        resultado = documentos.extrair_texto_pdf(artefato.caminho)
        texto_llm = (
            documentos.preparar_para_llm(
                resultado,
                limite_caracteres=documentos.LIMITE_CARACTERES_LLM,
            )
            if resultado.texto
            else ""
        )
        base.update(
            {
                "status_documento": resultado.status,
                "erro_documento": "",
                "pdf_sha256": artefato.sha256,
                "pdf_bytes": artefato.tamanho,
                "pdf_cache": artefato.de_cache,
                "paginas_total": resultado.paginas_total,
                "paginas_com_texto": resultado.paginas_com_texto,
                "caracteres_texto": resultado.caracteres,
                "texto_truncado_llm": len(texto_llm) < len(resultado.texto),
                "texto": resultado.texto,
                "texto_llm": texto_llm,
            }
        )
    except Exception as exc:  # registra a falha e continua o lote
        base.update(
            {
                "status_documento": "erro",
                "erro_documento": f"{type(exc).__name__}: {exc}",
                "pdf_sha256": "",
                "pdf_bytes": 0,
                "pdf_cache": False,
                "paginas_total": 0,
                "paginas_com_texto": 0,
                "caracteres_texto": 0,
                "texto_truncado_llm": False,
                "texto": "",
                "texto_llm": "",
            }
        )
    return base


def main() -> int:
    args = _argumentos()
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("--workers deve estar entre 1 e 16")
    if args.limite is not None and args.limite < 1:
        raise SystemExit("--limite deve ser positivo")

    run_v1 = _caminho_absoluto(args.run_v1)
    fatos_path = _caminho_absoluto(args.fatos)
    calendario_path = _caminho_absoluto(args.calendario)
    saida_path = _caminho_absoluto(args.saida)

    execucao = criar_execucao("v2_documentos")
    log = configurar_log(execucao, "v2_02_documentos")
    pares_path = run_v1 / "tabelas" / "pares_por_janela.csv"
    cfg_manifesto = {
        "run_v1": str(run_v1),
        "regra_original": "Tipo_Apresentacao=AP e Versao em {vazia,1}",
        "entrada_temporal": "primeira sessão B3 estritamente posterior",
        "limite_caracteres_llm": documentos.LIMITE_CARACTERES_LLM,
        "workers": args.workers,
        "limite_debug": args.limite,
        "somente_selecao": args.somente_selecao,
    }
    try:
        fatos = pd.read_parquet(fatos_path)
        pares = pd.read_csv(pares_path)
        calendario = _carregar_calendario(calendario_path)
        selecionados, diagnosticos, etapas = selecionar_corpus(
            fatos, pares, calendario
        )
        for etapa in etapas:
            log.info("%-28s %d", etapa["etapa"], etapa["documentos"])

        pd.DataFrame(etapas).to_csv(
            execucao.tabelas / "funil_documentos.csv", index=False, encoding="utf-8"
        )
        diagnosticos.to_csv(
            execucao.tabelas / "documentos_descartados.csv",
            index=False,
            encoding="utf-8",
        )
        selecionados.to_csv(
            execucao.tabelas / "documentos_selecionados.csv",
            index=False,
            encoding="utf-8",
        )

        if args.somente_selecao:
            gravar_manifesto(
                execucao,
                cfg_manifesto,
                arquivos_dados=[fatos_path, pares_path, calendario_path],
                status="concluida",
                extras={"documentos_selecionados": int(len(selecionados))},
            )
            return 0

        lote = selecionados if args.limite is None else selecionados.head(args.limite)
        registros = lote.to_dict("records")
        extraidos: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futuros = {
                executor.submit(_processar_documento, registro, forcar=args.forcar):
                registro[cvm_ipe.COLUNA_ID]
                for registro in registros
            }
            for numero, futuro in enumerate(as_completed(futuros), start=1):
                extraidos.append(futuro.result())
                if numero % 25 == 0 or numero == len(futuros):
                    log.info("documentos processados: %d/%d", numero, len(futuros))

        extracoes = pd.DataFrame(extraidos)
        corpus = lote.merge(
            extracoes, on=cvm_ipe.COLUNA_ID, how="left", validate="one_to_one"
        ).sort_values(["Sessao_Disponivel", "emissor_id", cvm_ipe.COLUNA_ID])
        saida_path.parent.mkdir(parents=True, exist_ok=True)
        corpus.to_parquet(saida_path, index=False)
        resumo_status = (
            corpus["status_documento"].value_counts(dropna=False).rename_axis("status")
            .reset_index(name="documentos")
        )
        resumo_status.to_csv(
            execucao.tabelas / "extracao_resumo.csv", index=False, encoding="utf-8"
        )
        falhas = corpus.loc[corpus["status_documento"].ne(documentos.STATUS_OK)]
        falhas.drop(columns=["texto", "texto_llm"], errors="ignore").to_csv(
            execucao.tabelas / "extracao_revisao.csv", index=False, encoding="utf-8"
        )
        gravar_manifesto(
            execucao,
            cfg_manifesto,
            arquivos_dados=[fatos_path, pares_path, calendario_path],
            status="concluida",
            extras={
                "arquivo_processado": str(saida_path.relative_to(RAIZ)),
                "documentos_elegiveis": int(len(selecionados)),
                "documentos_processados": int(len(corpus)),
                "documentos_ok": int(corpus["status_documento"].eq("ok").sum()),
            },
        )
        log.info("corpus salvo em %s", saida_path)
        return 0
    except Exception as exc:
        log.exception("preparação dos documentos falhou: %s", exc)
        gravar_manifesto(
            execucao,
            cfg_manifesto,
            arquivos_dados=[fatos_path, pares_path, calendario_path],
            status="falhou",
            extras={"erro": str(exc)},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
