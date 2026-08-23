"""
Tabelas e figuras do pre-relatorio.

Consolida as saidas da estimacao e do backtest, pegando a execucao concluida
mais recente de cada uma ou as indicadas por --etapa3/--etapa4, e produz:

  - resumo_metricas.csv       sharpe, drawdown, bootstrap, NW, alfa/beta por
                              carteira, bruto e liquido
  - car_por_evento.csv        retorno anormal acumulado no nivel do sinal
  - figuras/                  as quatro figuras essenciais, mais p-valores e
                              exposicao/turnover

O benchmark (^BVSP) vem do yfinance com cache local. Sem rede e sem cache, as
metricas que dependem dele saem vazias e a ausencia fica no manifesto, em vez
de derrubar o script.

Uso:
    .venv\\Scripts\\python.exe scripts\\05_gerar_relatorio.py
    .venv\\Scripts\\python.exe scripts\\05_gerar_relatorio.py --etapa4 <pasta>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import graficos, metricas  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DIR_RUNS = RAIZ / "outputs" / "runs"
CACHE_BENCH = RAIZ / "data" / "raw" / "yfinance" / "BVSP_indice.parquet"

CARTEIRAS = ("fdr_long_short", "fdr_long_only",
             "top_k_long_short", "top_k_long_only")


def _ultima_run(sufixo: str) -> Path | None:
    candidatas = sorted(DIR_RUNS.glob(f"*_{sufixo}"), reverse=True)
    for pasta in candidatas:
        manifesto = pasta / "manifest.json"
        if not manifesto.exists():
            continue
        if json.loads(manifesto.read_text("utf-8")).get("status") == "concluida":
            return pasta
    return None


def _benchmark(log) -> pd.Series | None:
    """Retornos diarios do ^BVSP, do cache ou do yfinance; None se nao der."""
    try:
        if CACHE_BENCH.exists():
            precos = pd.read_parquet(CACHE_BENCH)["Adj Close"]
        else:
            import yfinance as yf
            df = yf.download("^BVSP", start=str(CONFIG_PADRAO.periodo.inicio),
                             auto_adjust=False, progress=False)
            if df is None or df.empty:
                raise RuntimeError("download vazio")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            CACHE_BENCH.parent.mkdir(parents=True, exist_ok=True)
            df.reset_index().to_parquet(CACHE_BENCH, index=False)
            precos = df["Adj Close"]
        precos.index = pd.to_datetime(
            precos.index if precos.index.dtype.kind == "M"
            else pd.read_parquet(CACHE_BENCH)["Date"])
        r = precos.sort_index().pct_change()
        log.info("Benchmark ^BVSP: %d pregoes", r.notna().sum())
        return r
    except Exception as e:  # noqa: BLE001 - ausencia de benchmark nao derruba
        log.warning("Benchmark indisponivel (%s): metricas de alfa/beta e CAR "
                    "ficarao vazias", e)
        return None


def main(execucao: Execucao | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa3", type=Path, default=None)
    ap.add_argument("--etapa4", type=Path, default=None)
    args, _ = ap.parse_known_args()

    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa5")
    log = configurar_log(execucao, "05_relatorio")
    T, F = execucao.tabelas, execucao.figuras

    # No pipeline as etapas gravam na mesma pasta; rodando solto, consolidamos a
    # ultima run concluida de cada uma.
    compartilhada = (execucao.tabelas / "fdr_resumo_por_faixa.csv").exists() \
        and (execucao.tabelas / "resumo_carteiras.csv").exists()
    if compartilhada:
        pasta3 = pasta4 = execucao.pasta
    else:
        pasta3 = args.etapa3 or _ultima_run("etapa3")
        pasta4 = args.etapa4 or _ultima_run("etapa4")
    if pasta3 is None or pasta4 is None:
        log.error("Nao ha execucao concluida das etapas 3 e 4 para consolidar")
        gravar_manifesto(execucao, cfg.to_dict(), status="falhou",
                         extras={"erro": "etapas 3/4 ausentes"})
        return 1
    log.info("Etapa 3: %s | Etapa 4: %s", pasta3.name, pasta4.name)
    T3, T4 = pasta3 / "tabelas", pasta4 / "tabelas"

    bench = _benchmark(log)

    linhas = []
    for carteira in CARTEIRAS:
        arq = T4 / f"pnl_{carteira}.csv"
        if not arq.exists():
            linhas.append({"carteira": carteira, "disponivel": False})
            continue
        pnl = pd.read_csv(arq, index_col=0, parse_dates=True)
        turnover = pd.read_csv(T4 / f"exposicoes_{carteira}.csv",
                               index_col=0, parse_dates=True)
        for serie_nome in ("bruto", "liquido"):
            serie = pnl[serie_nome]
            linha = {"carteira": carteira, "serie": serie_nome,
                     "disponivel": True}
            linha.update(metricas.resumo_estrategia(serie, cfg.metricas))
            boot = metricas.block_bootstrap(
                serie, cfg.metricas, seed=cfg.seed,
                holding_dias=cfg.estrategia.holding_dias)
            linha.update({f"boot_{k}": v for k, v in boot.items()})
            linha.update({f"nw_{k}": v
                          for k, v in metricas.newey_west(serie,
                                                          cfg.metricas).items()})
            if bench is not None:
                linha.update({f"ab_{k}": v
                              for k, v in metricas.alpha_beta(
                                  serie, bench, cfg.metricas).items()})
            linhas.append(linha)
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(T / "resumo_metricas.csv", index=False, encoding="utf-8")
    log.info("--- resumo (liquido) ---")
    liq = resumo[resumo.get("serie") == "liquido"]
    for _, r in liq.iterrows():
        log.info("%-18s | total %+.2f%% | sharpe %+.2f | maxDD %+.2f%% | "
                 "boot p %.3f", r["carteira"], 100 * r["retorno_total"],
                 r["sharpe"], 100 * r["max_drawdown"], r["boot_p_unilateral"])

    eventos_arq = T4 / "eventos_top_k_long_short.csv"
    if bench is not None and eventos_arq.exists():
        from src import dados, qualidade_dados, retornos as mod_ret

        log.info("Recarregando COTAHIST para o painel de retornos do CAR...")
        cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
        calendario = qualidade_dados.calendario_pregoes(cot)
        ret_pnl = mod_ret.retornos_pnl_marcacao(cot, calendario)

        eventos = pd.read_csv(eventos_arq, parse_dates=["data"])
        eventos = eventos.rename(columns={"seguidora": "ticker"})
        car = metricas.car_por_evento(
            eventos, ret_pnl, bench.reindex(calendario).fillna(0.0),
            cfg.estrategia.holding_dias,
            cfg.walk_forward.defasagem_execucao_dias)
        # evento de venda aposta em CAR negativo, dai alinhar pela direcao
        eventos["car"] = car
        eventos["car_na_direcao"] = car * eventos["direcao"]
        eventos.to_csv(T / "car_por_evento.csv", index=False, encoding="utf-8")
        log.info("CAR: %d eventos | mediana na direcao %+.4f | media %+.4f",
                 len(eventos), eventos["car_na_direcao"].median(),
                 eventos["car_na_direcao"].mean())

    rede3 = pd.read_csv(T3 / "rede_por_janela.csv")
    fdr_resumo = pd.read_csv(T3 / "fdr_resumo_por_faixa.csv")
    invertida = pd.read_csv(T3 / "direcao_invertida.csv")
    placebo_dist = pd.read_csv(T3 / "placebo_distribuicao.csv")

    graficos.fig_efeito_por_faixa(rede3, cfg.liquidez.faixas,
                                  F / "01_efeito_por_faixa.png")
    graficos.fig_real_vs_placebo(rede3[rede3["faixa_minima"] <= 40],
                                 placebo_dist, invertida,
                                 F / "03_real_vs_placebo.png")
    graficos.fig_robustez_por_janela(fdr_resumo,
                                     F / "04_robustez_fdr_por_janela.png")
    graficos.fig_histograma_pvalores(rede3, F / "05_pvalores.png")

    for carteira in CARTEIRAS:
        arq = T4 / f"pnl_{carteira}.csv"
        if not arq.exists():
            continue
        pnl = pd.read_csv(arq, index_col=0, parse_dates=True)
        cen = pd.read_csv(T4 / f"cenarios_aluguel_{carteira}.csv",
                          index_col=0, parse_dates=True)
        graficos.fig_pnl_acumulado(pnl, cen,
                                   F / f"02_pnl_{carteira}.png",
                                   f"P&L acumulado - {carteira}")
        exp = pd.read_csv(T4 / f"exposicoes_{carteira}.csv",
                          index_col=0, parse_dates=True)
        pos = pd.read_csv(T4 / f"posicoes_{carteira}.csv",
                          index_col=0, parse_dates=True)
        giro = pos.diff().abs().sum(axis=1)
        graficos.fig_exposicao_turnover(exp, giro,
                                        F / f"06_exposicao_{carteira}.png",
                                        f"Exposicao e turnover - {carteira}")

    gravar_manifesto(
        execucao, cfg.to_dict(),
        status="concluida",
        extras={
            "consolidou_etapa3": pasta3.name,
            "consolidou_etapa4": pasta4.name,
            "benchmark_disponivel": bench is not None,
            "carteiras_com_metricas": int(liq["carteira"].nunique())
            if len(liq) else 0,
        })
    log.info("Tabelas em %s | Figuras em %s", T, F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
