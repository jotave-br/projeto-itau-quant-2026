"""
Por que o Yahoo nao tem serie para 199 tickers, e se isso importa.

A cobertura do universo point-in-time ficou entre 60% e 85%, e faltava saber se
o buraco era falha de fonte, que da para recuperar, ou limitacao estrutural, que
so da para declarar.

So diagnostico: nao baixa nada, nao mexe no mapeamento de mudanca de ticker e
nao grava painel. A comparacao de cobertura entre fontes serve para decidir com
o numero na mao, e nao para trocar a fonte aqui.

Uso:
    .venv\\Scripts\\python.exe scripts\\01c_diagnostico_ausencia_yf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import backtest, dados, qualidade_dados, retornos, universo  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def _ultimo(padrao: str) -> Path | None:
    achados = sorted(RAIZ.glob(padrao))
    return achados[-1] if achados else None


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa1c")
    log = configurar_log(execucao, "01c_ausencia")
    T = execucao.tabelas

    caminho_status = _ultimo("outputs/runs/*_etapa1b/tabelas/status_tickers_yfinance.csv")
    if caminho_status is None:
        log.error("Nao encontrei status_tickers_yfinance.csv. Rode a etapa 1b antes.")
        return 1
    log.info("Status do download: %s", caminho_status.parent.parent.name)
    status = pd.read_csv(caminho_status)

    candidatos = pd.DataFrame()
    caminho_cand = _ultimo("outputs/runs/*_etapa1/tabelas/candidatos_mudanca_ticker.csv")
    if caminho_cand is not None:
        candidatos = pd.read_csv(caminho_cand)
        log.info("Candidatos a mudanca de ticker: %d", len(candidatos))

    log.info("Carregando COTAHIST...")
    cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
    calendario = qualidade_dados.calendario_pregoes(cot)
    tickers = sorted(cot["CODNEG"].unique())
    log.info("%d tickers | %d pregoes | ate %s",
             len(tickers), len(calendario), calendario.max().date())

    # A contagem de ausentes nao mede materialidade point-in-time; a melhor
    # posicao no universo mostra se o estudo foi afetado.
    janelas = backtest.janelas_do_periodo(cot, cfg.periodo, cfg.walk_forward)
    log.info("Calculando melhor posicao PIT em %d janelas...", len(janelas))
    melhor = universo.melhor_posicao_por_ticker(cot, janelas, cfg.liquidez)

    diag = qualidade_dados.classificar_ausencia_yf(
        cot, status, candidatos=candidatos, melhor_posicao=melhor, cfg=cfg.liquidez)
    diag.to_csv(T / "diagnostico_ausencia_yf.csv", index=False, encoding="utf-8")
    log.info("Tickers sem serie no Yahoo: %d", len(diag))

    log.info("--- causa provavel ---")
    for c, n in diag["causa_provavel"].value_counts().items():
        log.info("  %-40s %d", c, n)

    log.info("--- materialidade (melhor faixa alcancada) ---")
    for f, n in diag["faixa_alcancada"].value_counts().items():
        log.info("  %-40s %d", f, n)

    log.info("--- via de recuperacao ---")
    for v, n in diag["via_recuperacao"].value_counts().items():
        log.info("  %-40s %d", v, n)

    cruz = pd.crosstab(diag["causa_provavel"], diag["faixa_alcancada"])
    cruz.to_csv(T / "ausencia_causa_x_materialidade.csv", encoding="utf-8")
    log.info("--- causa x materialidade ---\n%s", cruz.to_string())

    materiais = diag[~diag["faixa_alcancada"].isin(["nunca_elegivel", "fora_das_faixas"])]
    materiais.to_csv(T / "ausencia_material.csv", index=False, encoding="utf-8")
    log.info("Materiais (entraram em alguma faixa): %d de %d", len(materiais), len(diag))
    if len(materiais):
        log.info("\n%s", materiais[[
            "ticker", "melhor_posicao_pit", "faixa_alcancada", "ultima_data",
            "causa_provavel", "via_recuperacao", "nome"]].to_string(index=False))

    # Mede o que a via COTAHIST recuperaria, sem autorizar sua adocao.
    log.info("Medindo cobertura por fonte (nenhum painel e gravado)...")
    series = dados.carregar_series_yf(tickers)
    volume = retornos.painel_volume_financeiro(cot).reindex(index=calendario)

    precos_yf = retornos.painel_precos_ajustados(series, calendario)
    ret_yf = retornos.retornos_simples(precos_yf)
    ret_yf = ret_yf.where(retornos.mascara_retorno_valido(ret_yf, volume))

    ret_cot = retornos.retornos_preco_bruto_cotahist(cot, calendario, volume)
    log.info("Retornos utilizaveis  yfinance: %s", f"{int(ret_yf.notna().sum().sum()):,}")
    log.info("Retornos utilizaveis  COTAHIST: %s", f"{int(ret_cot.notna().sum().sum()):,}")

    linhas = []
    for j in janelas:
        rk = universo.ranking_liquidez_pit(cot, j, cfg.liquidez)
        if rk.empty:
            continue
        eleg = rk[rk["elegivel"]].sort_values("posicao")
        recorte = (ret_yf.index >= j.teste_inicio) & (ret_yf.index < j.teste_fim)
        t_yf, t_cot = ret_yf.loc[recorte], ret_cot.loc[recorte]
        for faixa in cfg.liquidez.faixas:
            nomes = list(eleg.head(faixa).index)
            if not nomes:
                continue

            def _tem(painel, nomes=nomes):
                return sum(1 for n in nomes
                           if n in painel.columns and painel[n].notna().sum() > 0)

            linhas.append({
                "janela": j.rotulo, "faixa": f"top{faixa}", "n_selecionados": len(nomes),
                "pct_com_yfinance": round(_tem(t_yf) / len(nomes), 4),
                "pct_com_cotahist": round(_tem(t_cot) / len(nomes), 4),
            })
    cob = pd.DataFrame(linhas)
    cob.to_csv(T / "cobertura_por_fonte_e_janela.csv", index=False, encoding="utf-8")
    if len(cob):
        resumo = cob.groupby("faixa")[["pct_com_yfinance", "pct_com_cotahist"]].agg(
            ["min", "median", "mean"])
        log.info("--- cobertura do universo PIT por fonte ---\n%s", resumo.round(3).to_string())
        cob["ano"] = cob["janela"].str[:4]
        por_ano = cob.groupby("ano")[["pct_com_yfinance", "pct_com_cotahist"]].mean()
        por_ano.to_csv(T / "cobertura_por_fonte_e_ano.csv", encoding="utf-8")
        log.info("--- cobertura por ano (todas as faixas) ---\n%s", por_ano.round(3).to_string())
        log.info("A tendencia temporal do yfinance e a assinatura do vies de "
                 "sobrevivencia: quanto mais antiga a janela, mais nomes ja "
                 "sairam da bolsa e foram apagados da fonte.")

    gravar_manifesto(execucao, cfg.to_dict(), status="concluida", extras={
        "ausencia_versao": qualidade_dados.AUSENCIA_VERSAO,
        "tickers_sem_serie": int(len(diag)),
        "materiais": int(len(materiais)),
        "causa_provavel": diag["causa_provavel"].value_counts().to_dict(),
        "via_recuperacao": diag["via_recuperacao"].value_counts().to_dict(),
        "painel_alternativo_gravado": False,
    })
    log.info("Tabelas em %s", T)
    log.info("So diagnostico: nenhuma fonte de retorno foi trocada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
