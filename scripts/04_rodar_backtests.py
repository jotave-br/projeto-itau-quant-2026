"""
Roda o walk-forward completo e apura P&L bruto e liquido.

Para cada janela: universo -> pares -> rede (treino, com FDR) -> selecao
congelada -> sinais no teste -> posicoes em t+1 -> safras 1/k -> custos.

Saem quatro carteiras, todas do mesmo conjunto de redes congeladas:

    fdr_long_short    a afirmacao inferencial operada (pode ficar vazia)
    fdr_long_only     idem, implementabilidade
    top_k_long_short  garante serie de P&L para analise
    top_k_long_only   idem, implementabilidade

O P&L usa COTAHIST bruto com marcacao a ultimo preco, sem mascara de eventos, e
as travessias de dia ex ficam contadas no manifesto: e o tamanho observado do
vies dessa fonte.

Uso:
    .venv\\Scripts\\python.exe scripts\\04_rodar_backtests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import backtest, dados, qualidade_dados, retornos, setores  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SETORES = RAIZ / "data" / "reference" / "setores_b3.csv"


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa4")
    log = configurar_log(execucao, "04_backtests")
    T = execucao.tabelas

    tabela_setores = setores.carregar(SETORES)
    erros = setores.validar(tabela_setores)
    if erros:
        for e in erros:
            log.error("  - %s", e)
        gravar_manifesto(execucao, cfg.to_dict(), status="falhou",
                         extras={"erro": "setores_b3.csv invalido"})
        return 1

    log.info("Carregando COTAHIST e montando paineis...")
    cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
    calendario = qualidade_dados.calendario_pregoes(cot)
    volume = retornos.painel_volume_financeiro(cot).reindex(calendario)
    ret_estimacao = retornos.retornos_preco_bruto_cotahist(cot, calendario,
                                                          volume)
    ret_pnl = retornos.retornos_pnl_marcacao(cot, calendario)
    janelas = backtest.janelas_do_periodo(cot, cfg.periodo, cfg.walk_forward)
    log.info("%d janelas | fonte P&L: %s", len(janelas),
             cfg.dados.fonte_retornos_pnl)

    resultado = backtest.rodar_walk_forward(
        cot, calendario, tabela_setores, ret_estimacao, ret_pnl, janelas, cfg)

    resultado["rede"].to_csv(T / "rede_por_janela.csv", index=False,
                             encoding="utf-8")
    for regra, sel in resultado["selecionados"].items():
        sel.to_csv(T / f"pares_selecionados_{regra}.csv", index=False,
                   encoding="utf-8")
    resultado["diagnosticos"].to_csv(T / "diagnosticos_por_janela.csv",
                                     index=False, encoding="utf-8")

    resumos = []
    log.info("--- carteiras ---")
    for rotulo, r in resultado["combos"].items():
        if r.get("vazio"):
            log.warning("%s: nenhuma posicao gerada em nenhuma janela", rotulo)
            resumos.append({"carteira": rotulo, "vazia": True})
            continue
        r["pnl"].to_csv(T / f"pnl_{rotulo}.csv", encoding="utf-8")
        r["cenarios_aluguel"].to_csv(T / f"cenarios_aluguel_{rotulo}.csv",
                                     encoding="utf-8")
        r["pernas"].to_csv(T / f"pernas_{rotulo}.csv", encoding="utf-8")
        r["exposicoes"].to_csv(T / f"exposicoes_{rotulo}.csv", encoding="utf-8")
        r["posicoes"].to_csv(T / f"posicoes_{rotulo}.csv", encoding="utf-8")
        r["eventos"].to_csv(T / f"eventos_{rotulo}.csv", index=False,
                            encoding="utf-8")

        s = r["resumo"]
        resumos.append({"carteira": rotulo, "vazia": False, **s})
        log.info("%-18s | bruto %+.2f%% | liquido %+.2f%% | turnover medio "
                 "%.3f | travessias de evento %.2f%% dos posicao-dias",
                 rotulo, 100 * s["pnl_bruto_total"],
                 100 * s["pnl_liquido_total"], s["turnover_medio_diario"],
                 100 * s["frac_travessias"])

    pd.DataFrame(resumos).to_csv(T / "resumo_carteiras.csv", index=False,
                                 encoding="utf-8")

    gravar_manifesto(
        execucao, cfg.to_dict(),
        arquivos_dados=[SETORES],
        status="concluida",
        extras={
            "janelas": len(janelas),
            "fonte_pnl": cfg.dados.fonte_retornos_pnl,
            "carteiras": {r["carteira"]: {k: v for k, v in r.items()
                                          if k != "carteira"}
                          for r in resumos},
            "rede_estimada": True,
            "pnl_calculado": True,
        })
    log.info("Tabelas em %s", T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
