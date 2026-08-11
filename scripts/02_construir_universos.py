"""
Constroi os universos point-in-time e os pares candidatos.

Para cada janela de treino do walk-forward:

  1. ranking de liquidez so com dados daquela janela;
  2. elegibilidade (cobertura, dias negociados, minimo de pregoes);
  3. uma classe por emissor, via parser validado de ISIN;
  4. faixas top 20/40/60/100 sobre a selecao deduplicada;
  5. setor/subsetor vigentes no ultimo pregao do treino;
  6. pares lider -> seguidora dentro de cada (setor, subsetor).

Tudo sai congelado antes da janela de teste correspondente, e nenhum retorno,
correlacao ou P&L participa de qualquer decisao.

Saidas nas tabelas da execucao:
  universo_por_janela.csv   todos os tickers de cada janela, com motivo de
                            exclusao em quem ficou fora
  pares_por_janela.csv      pares candidatos, com faixa minima
  funil_exclusoes.csv       contagem de exclusoes por janela e motivo

Uso:
    .venv\\Scripts\\python.exe scripts\\02_construir_universos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import backtest, dados, pares, qualidade_dados, setores, universo  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SETORES = RAIZ / "data" / "reference" / "setores_b3.csv"


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa2")
    log = configurar_log(execucao, "02_universos")
    T = execucao.tabelas

    tabela_setores = setores.carregar(SETORES)
    erros = setores.validar(tabela_setores)
    if erros:
        log.error("setores_b3.csv invalido - a formacao de pares nao pode usar "
                  "uma tabela reprovada:")
        for e in erros:
            log.error("  - %s", e)
        gravar_manifesto(execucao, cfg.to_dict(), status="falhou",
                         extras={"erro": "setores_b3.csv invalido"})
        return 1
    log.info("setores_b3.csv valido: %d linha(s)", len(tabela_setores))

    log.info("Carregando COTAHIST...")
    cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
    calendario = qualidade_dados.calendario_pregoes(cot)
    janelas = backtest.janelas_do_periodo(cot, cfg.periodo, cfg.walk_forward)
    log.info("%d janelas de %dm/%dm entre %s e %s", len(janelas),
             cfg.walk_forward.treino_meses, cfg.walk_forward.teste_meses,
             janelas[0].treino_inicio.date(), janelas[-1].teste_fim.date())

    universos, todos_pares = [], []
    for j in janelas:
        sel = universo.selecionar_universo(cot, j, cfg.liquidez)
        data_formacao = calendario[calendario < j.treino_fim].max()
        sel = pares.com_setor_vigente(sel, tabela_setores, data_formacao)
        p = pares.gerar_pares(sel, cfg.liquidez.faixas)

        universos.append(sel.reset_index())
        todos_pares.append(p)

        na_principal = int((sel["posicao_final"] <= cfg.liquidez.top_n).sum())
        sem_setor = int((sel["motivo_exclusao"] == pares.MOTIVO_SEM_SETOR).sum())
        log.info("%s | top%d completo%s | %3d pares (top%d: %3d)%s",
                 j.rotulo, cfg.liquidez.top_n,
                 "" if na_principal == cfg.liquidez.top_n
                 else f" nao ({na_principal})",
                 len(p), cfg.liquidez.top_n,
                 int((p["faixa_minima"] <= cfg.liquidez.top_n).sum()),
                 f" | {sem_setor} sem setor na formacao" if sem_setor else "")

    univ = pd.concat(universos, ignore_index=True)
    par = pd.concat(todos_pares, ignore_index=True)
    univ.to_csv(T / "universo_por_janela.csv", index=False, encoding="utf-8")
    par.to_csv(T / "pares_por_janela.csv", index=False, encoding="utf-8")

    # O funil e como a selecao presta contas: toda exclusao aparece com motivo e
    # contagem, janela a janela.
    funil = (univ[univ["motivo_exclusao"] != ""]
             .groupby(["janela", "motivo_exclusao"]).size()
             .rename("tickers").reset_index())
    funil.to_csv(T / "funil_exclusoes.csv", index=False, encoding="utf-8")

    log.info("--- exclusoes acumuladas por motivo ---\n%s",
             funil.groupby("motivo_exclusao")["tickers"].sum().to_string())

    pares_top_n = par[par["faixa_minima"] <= cfg.liquidez.top_n]
    log.info("--- pares por janela na faixa principal (top%d) ---",
             cfg.liquidez.top_n)
    log.info("mediana %.0f | minimo %d | maximo %d",
             pares_top_n.groupby("janela").size().median(),
             pares_top_n.groupby("janela").size().min(),
             pares_top_n.groupby("janela").size().max())

    sem_setor_total = univ[univ["motivo_exclusao"] == pares.MOTIVO_SEM_SETOR]
    if len(sem_setor_total):
        log.warning("%d exclusao(oes) por falta de classificacao confirmada, "
                    "tickers: %s", len(sem_setor_total),
                    sorted(sem_setor_total["CODNEG"].unique()))

    gravar_manifesto(
        execucao, cfg.to_dict(),
        arquivos_dados=[SETORES],
        status="concluida",
        extras={
            "janelas": len(janelas),
            "faixa_principal": f"top{cfg.liquidez.top_n}",
            "pares_total": int(len(par)),
            "pares_faixa_principal": int(len(pares_top_n)),
            "pares_faixa_principal_mediana_por_janela": float(
                pares_top_n.groupby("janela").size().median()),
            "exclusoes_por_motivo": {
                m: int(n) for m, n in
                funil.groupby("motivo_exclusao")["tickers"].sum().items()},
            "selecionados_sem_setor_na_formacao": int(len(sem_setor_total)),
            "pares_escolhidos_por_desempenho": False,
            "rede_estimada": False,
        })
    log.info("Tabelas em %s", T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
