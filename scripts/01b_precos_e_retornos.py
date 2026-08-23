"""
Precos ajustados do yfinance e painel de retornos.

Cada fonte manda no que e dela. O COTAHIST decide existencia historica do
instrumento, calendario, liquidez, numero de negocios e universo point-in-time.
O yfinance entra so como preco ajustado e eventos corporativos: ele nao tem
autoridade sobre identidade historica, porque emenda tickers renomeados por
conta propria, de forma opaca e inconsistente.

O painel sai sem continuidade canonica nos 41 candidatos a mudanca de ticker -
aquilo depende de revisao documental.

Uso:
    .venv\\Scripts\\python.exe scripts\\01b_precos_e_retornos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import backtest, dados, qualidade_dados, retornos, universo  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
MAPA = RAIZ / "data" / "reference" / "mudancas_ticker.csv"


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa1b")
    log = configurar_log(execucao, "01b_precos")
    T = execucao.tabelas

    log.info("Carregando COTAHIST...")
    cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
    tickers = sorted(cot["CODNEG"].unique())
    calendario = qualidade_dados.calendario_pregoes(cot)
    ultima = calendario.max().date()
    log.info("%d tickers | %d pregoes | ate %s", len(tickers), len(calendario), ultima)

    bloqueados: set[str] = set()
    if MAPA.exists():
        mp = pd.read_csv(MAPA, dtype=str).fillna("")
        pend = mp[~mp["status"].isin(["confirmado"])]
        bloqueados = set(pend["ticker_antigo"]) | set(pend["ticker_novo"])
        log.info("Continuidade bloqueada (revisao pendente): %d tickers",
                 len(bloqueados & set(tickers)))

    log.info("Baixando precos ajustados (lotes de %d, ate %d tentativas)...",
             dados.YF_TAMANHO_LOTE, dados.YF_MAX_TENTATIVAS)
    status = dados.baixar_yfinance_lote(
        tickers, cfg.periodo.inicio, ultima, log=log)

    # A etapa so termina quando todo ticker tem estado documentado, e nao quando
    # a chamada retorna.
    series = dados.carregar_series_yf(tickers)
    trocas = pd.DataFrame()
    caminho_trocas = sorted(RAIZ.glob(
        "outputs/runs/*_etapa1/tabelas/candidatos_mudanca_ticker.csv"))
    if caminho_trocas:
        trocas = pd.read_csv(caminho_trocas[-1])

    terminais: dict[str, str] = {}
    if len(trocas):
        for tk in tickers:
            if tk in series:
                continue
            term = qualidade_dados.ticker_terminal_da_cadeia(trocas, tk)
            if term and term in series:
                terminais[tk] = term

    def _estado(row) -> str:
        tk = row["ticker"]
        if row["status"] == "sucesso":
            return "bloqueado" if tk in bloqueados else "sucesso"
        if tk in terminais:
            return "coberto_por_ticker_terminal"
        if row["status"] == "serie_vazia":
            return "serie_vazia"
        return "erro_persistente"

    status["estado_final"] = status.apply(_estado, axis=1)
    status["ticker_terminal"] = status["ticker"].map(terminais).fillna("")
    status.to_csv(T / "status_tickers_yfinance.csv", index=False, encoding="utf-8")

    log.info("--- estados finais ---")
    for e, n in status["estado_final"].value_counts().items():
        log.info("  %-30s %d", e, n)
    faltando = set(tickers) - set(status["ticker"])
    if faltando:
        log.error("!!! %d ticker(s) sem estado: %s", len(faltando), sorted(faltando)[:10])
        return 1
    log.info("Todos os %d tickers tem estado documentado.", len(tickers))

    status[status["estado_final"].isin(["serie_vazia", "erro_persistente"])].to_csv(
        T / "series_vazias_ou_com_falha.csv", index=False, encoding="utf-8")

    log.info("Montando paineis (%d series em cache)...", len(series))
    precos = retornos.painel_precos_ajustados(series, calendario)
    volume = retornos.painel_volume_financeiro(cot).reindex(index=calendario)
    negocios = retornos.painel_numero_negocios(cot).reindex(index=calendario)

    ret = retornos.retornos_simples(precos)
    valido = retornos.mascara_retorno_valido(ret, volume)
    ret_util = ret.where(valido)
    log.info("Painel de retornos: %d x %d | %s retornos utilizaveis",
             *ret_util.shape, f"{int(valido.sum().sum()):,}")

    precos.to_parquet(RAIZ / "data" / "processed" / "precos_ajustados.parquet")
    ret_util.to_parquet(RAIZ / "data" / "processed" / "retornos_diarios.parquet")

    # A cobertura inclui todos os tickers, nao so os que retornaram dados. Sem o
    # reindex, quem nao tem serie nem vira coluna do painel, a
    # categoria "ausente" nunca dispara e a tabela fica otima por omitir
    # justamente os casos ruins.
    precos_todos = precos.reindex(columns=tickers)
    cob = retornos.cobertura_yahoo_vs_cotahist(
        precos_todos, volume, status_download=status,
        tickers_continuidade_bloqueada=bloqueados & set(tickers),
        tickers_cobertos_por_terminal=terminais)
    cob.to_csv(T / "cobertura_yahoo_vs_cotahist.csv", encoding="utf-8")
    log.info("--- situacao de cobertura ---")
    for s, n in cob["situacao_cobertura"].value_counts().items():
        log.info("  %-32s %d", s, n)

    log.info("Procurando series duplicadas ou migradas...")
    dup = retornos.detectar_series_duplicadas(ret_util)
    dup.to_csv(T / "series_duplicadas_ou_migradas.csv", index=False, encoding="utf-8")
    log.info("  %d par(es) suspeito(s) de duplicidade", len(dup))

    liq = cot.groupby("CODNEG")["VOLTOT"].median()
    c2 = cob.join(liq.rename("voltot_mediano"))
    c2 = c2[c2["voltot_mediano"].notna()]
    c2["decil_liquidez"] = pd.qcut(
        c2["voltot_mediano"].rank(method="first"), 10,
        labels=[f"D{i}" for i in range(1, 11)])
    por_decil = c2.groupby("decil_liquidez", observed=True).agg(
        tickers=("cobertura_yahoo_vs_cotahist", "size"),
        voltot_mediano_MM=("voltot_mediano", lambda s: round(s.median() / 1e6, 3)),
        cobertura_mediana=("cobertura_yahoo_vs_cotahist", "median"),
        frac_ausente=("situacao_cobertura", lambda s: round((s == "ausente").mean(), 4)),
        frac_completa=("situacao_cobertura",
                       lambda s: round((s == "cobertura_completa").mean(), 4)))
    por_decil.to_csv(T / "cobertura_por_decil_liquidez.csv", encoding="utf-8")
    log.info("--- cobertura por decil de liquidez ---")
    log.info("\n%s", por_decil.to_string())

    log.info("Cobertura por faixa em cada janela...")
    janelas = backtest.janelas_do_periodo(cot, cfg.periodo, cfg.walk_forward)
    linhas = []
    for j in janelas:
        rk = universo.ranking_liquidez_pit(cot, j, cfg.liquidez)
        if rk.empty:
            continue
        eleg = rk[rk["elegivel"]].sort_values("posicao")
        teste = ret_util.loc[(ret_util.index >= j.teste_inicio)
                             & (ret_util.index < j.teste_fim)]
        for faixa in cfg.liquidez.faixas:
            nomes = list(eleg.head(faixa).index)
            if not nomes:
                continue
            presentes = [n for n in nomes if n in teste.columns]
            com_dado = [n for n in presentes if teste[n].notna().sum() > 0]
            linhas.append({
                "janela": j.rotulo, "faixa": f"top{faixa}",
                "n_selecionados": len(nomes),
                "com_retorno_utilizavel": len(com_dado),
                "pct_com_retorno": round(len(com_dado) / len(nomes), 4),
                "pregoes_teste": len(teste),
            })
    cob_faixa = pd.DataFrame(linhas)
    cob_faixa.to_csv(T / "cobertura_por_faixa_e_janela.csv", index=False, encoding="utf-8")
    if len(cob_faixa):
        resumo = cob_faixa.groupby("faixa")["pct_com_retorno"].agg(["min", "median", "mean"])
        log.info("--- %% do universo com retorno utilizavel, por faixa ---")
        log.info("\n%s", resumo.round(4).to_string())
        piores = cob_faixa.nsmallest(5, "pct_com_retorno")
        log.info("--- janelas com menor cobertura ---")
        log.info("\n%s", piores.to_string(index=False))

    gravar_manifesto(execucao, cfg.to_dict(), status="concluida", extras={
        "tickers": len(tickers),
        "estados": status["estado_final"].value_counts().to_dict(),
        "situacao_cobertura": cob["situacao_cobertura"].value_counts().to_dict(),
        "pares_duplicados": int(len(dup)),
        "retornos_utilizaveis": int(valido.sum().sum()),
        "continuidade_canonica_aplicada": False,
    })
    log.info("Tabelas em %s", T)
    log.info("Painel sem continuidade canonica: os 41 candidatos seguem separados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
