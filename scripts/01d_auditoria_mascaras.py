"""
Quanto a mascara de eventos remove, e onde.

Com o COTAHIST como fonte principal, o preco e bruto e o retorno de estimacao
que atravessa a fronteira de um evento corporativo e descartado. Esta etapa
mede o custo desse filtro antes da estimacao. Nao estima rede, nao forma pares
e nao seleciona nada.

A fracao removida depende do denominador, e misturar os tres e o jeito mais
facil de reportar numero errado:

    ticker_dias_observados   linhas do COTAHIST (ha registro do papel no dia)
    celulas_do_painel        datas x tickers, inclui papel que nem existia
    retornos_candidatos      pares de pregoes consecutivos com preco > 0 e
                             negociacao efetiva nos dois

O terceiro e o unico que responde "quanto se perdeu", porque mede sobre o que
seria utilizavel sem a mascara. As fracoes de remocao usam esse denominador;
as medidas de cobertura identificam o proprio denominador no nome da coluna.

Uso:
    .venv\\Scripts\\python.exe scripts\\01d_auditoria_mascaras.py
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
SETORES = RAIZ / "data" / "reference" / "setores_b3.csv"

# Limiares de alerta, nao criterios de aprovacao. Servem para destacar universos
# que ficaram sem dados suficientes.
MAX_FRAC_REMOVIDA = 0.15
MIN_COBERTURA_UNIVERSO = 0.90

# Retorno diario acima disto vai para revisao manual, sem correcao automatica.
LIMIAR_EXTREMO = 0.50

# Cortes de amostra minima reportados lado a lado. O corte e convencao, e se a
# leitura muda com ele, tem que aparecer.
CORTES_SENSIBILIDADE = (10, 30, 50)


def _ultimo(padrao: str) -> Path | None:
    achados = sorted(RAIZ.glob(padrao))
    return achados[-1] if achados else None


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa1d")
    log = configurar_log(execucao, "01d_mascaras")
    T = execucao.tabelas

    log.info("Estimacao: %s | robustez: %s | P&L: %s",
             cfg.dados.fonte_retornos_estimacao,
             cfg.dados.fonte_retornos_estimacao_robustez,
             cfg.dados.fonte_retornos_pnl)

    log.info("Carregando COTAHIST...")
    cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
    calendario = qualidade_dados.calendario_pregoes(cot)
    tickers = sorted(cot["CODNEG"].unique())
    volume = retornos.painel_volume_financeiro(cot).reindex(index=calendario)

    precos = (cot.pivot_table(index="DATA", columns="CODNEG", values="PREULT",
                              aggfunc="last")
              .reindex(calendario))
    precos = precos.where(precos > 0)
    candidatos = retornos.mascara_retorno_valido(
        retornos.retornos_simples(precos), volume)

    ret_cot = retornos.retornos_preco_bruto_cotahist(
        cot, calendario, volume,
        mascarar_dia_seguinte=cfg.dados.mascarar_pregao_seguinte_ao_evento)
    removido = candidatos & ret_cot.isna()

    n_cand = int(candidatos.sum().sum())
    n_rem = int(removido.sum().sum())
    denominadores = {
        "ticker_dias_observados": int(len(cot)),
        "celulas_do_painel": int(precos.size),
        "retornos_candidatos": n_cand,
        "retornos_removidos_por_evento": n_rem,
        "frac_removida_sobre_candidatos": round(n_rem / n_cand, 6),
        "retornos_utilizaveis": int(ret_cot.notna().sum().sum()),
    }
    log.info("--- denominadores ---")
    for k, v in denominadores.items():
        log.info("  %-34s %s", k, f"{v:,}" if isinstance(v, int) else v)
    pd.Series(denominadores).to_csv(T / "mascara_denominadores.csv",
                                    header=["valor"], encoding="utf-8")

    por_ticker = pd.DataFrame({
        "candidatos": candidatos.sum(),
        "removidos": removido.sum(),
    })
    por_ticker["frac_removida"] = (por_ticker["removidos"]
                                   / por_ticker["candidatos"].replace(0, pd.NA))
    por_ticker = por_ticker[por_ticker["candidatos"] > 0].sort_values(
        "frac_removida", ascending=False)
    por_ticker.to_csv(T / "mascara_por_ticker.csv", encoding="utf-8")
    log.info("--- por ticker (%d com candidatos) ---", len(por_ticker))
    log.info("  frac_removida: mediana %.4f | p95 %.4f | max %.4f",
             por_ticker["frac_removida"].median(),
             por_ticker["frac_removida"].quantile(0.95),
             por_ticker["frac_removida"].max())
    log.info("  10 mais afetados:\n%s", por_ticker.head(10).round(4).to_string())

    janelas = backtest.janelas_do_periodo(cot, cfg.periodo, cfg.walk_forward)
    log.info("Auditando %d janelas de walk-forward...", len(janelas))

    series = dados.carregar_series_yf(tickers)
    p_yf = retornos.painel_precos_ajustados(series, calendario)
    ret_yf = retornos.retornos_simples(p_yf)
    ret_yf = ret_yf.where(retornos.mascara_retorno_valido(ret_yf, volume))

    linhas = []
    for j in janelas:
        rk = universo.ranking_liquidez_pit(cot, j, cfg.liquidez)
        if rk.empty:
            continue
        eleg = rk[rk["elegivel"]].sort_values("posicao")
        no_teste = (calendario >= j.teste_inicio) & (calendario < j.teste_fim)
        cand_j, rem_j = candidatos.loc[no_teste], removido.loc[no_teste]
        cot_j, yf_j = ret_cot.loc[no_teste], ret_yf.loc[no_teste]

        for faixa in cfg.liquidez.faixas:
            nomes = [n for n in eleg.head(faixa).index if n in cand_j.columns]
            if not nomes:
                continue
            c = int(cand_j[nomes].sum().sum())
            r = int(rem_j[nomes].sum().sum())
            com_cot = sum(1 for n in nomes if cot_j[n].notna().sum() > 0)
            com_yf = sum(1 for n in nomes
                         if n in yf_j.columns and yf_j[n].notna().sum() > 0)
            linhas.append({
                "janela": j.rotulo,
                "faixa": f"top{faixa}",
                "n_selecionados": len(eleg.head(faixa)),
                "retornos_candidatos": c,
                "retornos_removidos": r,
                "frac_removida": round(r / c, 4) if c else None,
                "tickers_com_retorno_cotahist": com_cot,
                "tickers_com_retorno_yfinance": com_yf,
                "cobertura_cotahist": round(com_cot / len(eleg.head(faixa)), 4),
                "cobertura_yfinance": round(com_yf / len(eleg.head(faixa)), 4),
            })
    aud = pd.DataFrame(linhas)
    aud.to_csv(T / "mascara_por_janela_e_faixa.csv", index=False, encoding="utf-8")

    resumo = aud.groupby("faixa").agg(
        frac_removida_media=("frac_removida", "mean"),
        frac_removida_max=("frac_removida", "max"),
        cobertura_cot_min=("cobertura_cotahist", "min"),
        cobertura_cot_mediana=("cobertura_cotahist", "median"),
        cobertura_yf_min=("cobertura_yfinance", "min"),
        cobertura_yf_mediana=("cobertura_yfinance", "median"))
    resumo.to_csv(T / "mascara_resumo_por_faixa.csv", encoding="utf-8")
    log.info("--- por faixa ---\n%s", resumo.round(4).to_string())

    por_janela = aud.groupby("janela").agg(
        frac_removida=("frac_removida", "mean"),
        cobertura_cotahist=("cobertura_cotahist", "min"),
        cobertura_yfinance=("cobertura_yfinance", "min"))
    por_janela.to_csv(T / "mascara_por_janela.csv", encoding="utf-8")
    log.info("--- 5 janelas com maior remocao ---\n%s",
             por_janela.nlargest(5, "frac_removida").round(4).to_string())

    # A comparacao yfinance x COTAHIST usa apenas o suporte comum; fora dele, a
    # diferenca seria de amostra, e nao de fonte.
    comuns = [t for t in ret_cot.columns if t in ret_yf.columns]
    ambos = ret_cot[comuns].notna() & ret_yf[comuns].notna()
    suporte = pd.DataFrame({
        "obs_cotahist": ret_cot[comuns].notna().sum(),
        "obs_yfinance": ret_yf[comuns].notna().sum(),
        "obs_suporte_comum": ambos.sum(),
    })
    suporte["frac_do_cotahist_no_comum"] = (
        suporte["obs_suporte_comum"] / suporte["obs_cotahist"].replace(0, pd.NA))
    suporte.to_csv(T / "suporte_comum_por_ticker.csv", encoding="utf-8")
    log.info("--- suporte comum ---")
    log.info("  tickers nas duas fontes: %d de %d", len(comuns), len(ret_cot.columns))
    log.info("  observacoes: cotahist %s | yfinance %s | comum %s",
             f"{int(suporte['obs_cotahist'].sum()):,}",
             f"{int(suporte['obs_yfinance'].sum()):,}",
             f"{int(suporte['obs_suporte_comum'].sum()):,}")

    log.info("Validando a fronteira por token...")
    val = retornos.validar_fronteiras_por_token(cot, calendario, ret_yf, volume)
    val.to_csv(T / "validacao_fronteira_por_token.csv", index=False, encoding="utf-8")
    log.info("--- fronteiras removidas, por token ---\n%s", val[[
        "token", "fronteira_n_total_cotahist", "fronteira_n_suporte_comum",
        "fronteira_cobertura_referencia", "fronteira_erro_mediano",
        "fronteira_erro_p90", "fronteira_erro_p99", "fronteira_erro_max",
        "fronteira_frac_acima_1bp", "fronteira_frac_acima_10bps",
        "fronteira_frac_acima_50bps", "classificacao_evidencia",
    ]].head(20).round(6).to_string(index=False))
    log.info("--- dias marcados mantidos ---\n%s", val[[
        "token", "mantido_n_total_cotahist", "mantido_n_suporte_comum",
        "mantido_erro_mediano", "mantido_erro_p95", "mantido_erro_p99",
        "mantido_erro_max", "mantido_frac_acima_10bps",
    ]].head(20).round(6).to_string(index=False))
    log.info("--- classificacao da evidencia ---")
    for c, n in val["classificacao_evidencia"].value_counts().items():
        log.info("  %-32s %d token(s)", c, n)

    # Se a leitura muda com o corte de amostra minima, isso tem que aparecer em
    # vez de ficar implicito.
    sens = pd.DataFrame({
        f"min_{n}": retornos.reclassificar_evidencia(val, min_fronteiras=n)
        .value_counts() for n in CORTES_SENSIBILIDADE}).fillna(0).astype(int)
    sens.to_csv(T / "evidencia_sensibilidade_amostra.csv", encoding="utf-8")
    log.info("--- sensibilidade ao corte de amostra minima ---\n%s",
             sens.to_string())
    log.info("A R3 permanece integral por conservadorismo, em qualquer corte. "
             "Evidencia fraca nao autoriza excecao: abrir uma olhando estes "
             "numeros seria ajustar a regra ao resultado.")
    log.info("A referencia so existe onde o yfinance tem serie: nada aqui vale "
             "para quem saiu da bolsa.")

    candidatos_mudanca = pd.DataFrame()
    caminho_cand = _ultimo("outputs/runs/*_etapa1/tabelas/candidatos_mudanca_ticker.csv")
    if caminho_cand is not None:
        candidatos_mudanca = pd.read_csv(caminho_cand)

    ext = retornos.auditar_retornos_extremos(
        ret_cot, cot, calendario, candidatos_mudanca=candidatos_mudanca,
        limiar=LIMIAR_EXTREMO)
    log.info("--- triagem: |r| > %.0f%% remanescentes: %d ---",
             LIMIAR_EXTREMO * 100, len(ext))

    if len(ext):
        # Sem o cruzamento com os universos, a fila de revisao sairia ordenada
        # por tamanho do retorno, que e o criterio errado.
        alc = universo.alcance_pit(ext[["ticker", "data"]], cot, janelas, cfg.liquidez)
        ext = ext.merge(alc, on=["ticker", "data"], how="left")
        ext.to_csv(T / "retornos_extremos_triagem.csv", index=False, encoding="utf-8")

        log.info("causa provavel (indicio, nao confirmacao):\n%s",
                 ext["causa_provavel"].value_counts().to_string())

        # A melhor posicao global responde "o papel chegou a importar?", e nao
        # "este evento importou?". Quem decide efeito e o estado na data.
        estados = pd.DataFrame({
            "treino": ext["status_treino_na_data"].value_counts(),
            "teste": ext["status_teste_na_data"].value_counts(),
        }).fillna(0).astype(int)
        estados.to_csv(T / "extremos_estado_exclusivo_na_data.csv",
                       index_label="estado", encoding="utf-8")
        log.info("--- estado exclusivo na data (cada caso conta uma vez) ---\n%s",
                 estados.to_string())
        log.info("--- melhor estado em qualquer janela (so descritivo) ---\n%s",
                 ext["faixa_melhor_qualquer_janela"].value_counts().to_string())

        # Leitura diferente da anterior: aqui top100 contem top20, top40 e
        # top60, entao um caso na 58a posicao aparece em top60 e em top100.
        cum = pd.DataFrame({
            "treino": universo.alcance_cumulativo_por_faixa(
                ext, "melhor_posicao_treino_na_data", cfg.liquidez.faixas),
            "teste": universo.alcance_cumulativo_por_faixa(
                ext, "melhor_posicao_teste_na_data", cfg.liquidez.faixas),
        })
        cum.to_csv(T / "extremos_alcance_cumulativo_por_faixa.csv",
                   index_label="faixa", encoding="utf-8")
        log.info("--- alcance_cumulativo_por_faixa (top100 contem os menores) ---\n%s",
                 cum.to_string())

        log.info("pode afetar a estimacao (data no treino, papel em faixa): %d",
                 int(ext["pode_afetar_estimacao"].sum()))
        log.info("pode afetar sinal/P&L (data no teste, universo congelado): %d",
                 int(ext["pode_afetar_sinal"].sum()))
        log.info("Estes numeros sao limite superior: nao houve deduplicacao por "
                 "emissor, filtro setorial nem formacao de pares, e cada um dos "
                 "tres so reduz o conjunto.")

        pri = ext[ext["pode_afetar_estimacao"] | ext["pode_afetar_sinal"]]
        pri.to_csv(T / "retornos_extremos_prioridade.csv", index=False, encoding="utf-8")
        log.info("--- fila de revisao documental (%d casos) ---\n%s", len(pri),
                 pri.sort_values("melhor_posicao_treino_na_data").head(20)[
                     ["ticker", "data", "retorno", "status_treino_na_data",
                      "melhor_posicao_treino_na_data", "status_teste_na_data",
                      "melhor_posicao_teste_na_data", "faixa_melhor_qualquer_janela",
                      "fator_redondo", "causa_provavel"]].to_string(index=False))
    else:
        ext.to_csv(T / "retornos_extremos_triagem.csv", index=False, encoding="utf-8")
    log.info("Isto e triagem, nao auditoria concluida. Fator redondo e indicio "
             "de grupamento, nao confirmacao. Nada foi winsorizado nem "
             "corrigido.")

    pendentes = [
        "efeito da mascara por setor/subsetor",
        "efeito da mascara por papel de lider/seguidora",
    ]
    if not SETORES.exists():
        pendentes.append("classificacao setorial indisponivel: "
                         "data/reference/setores_b3.csv nao existe")
    for p in pendentes:
        log.warning("dimensao nao auditada -> %s", p)

    alertas = []
    ruins = aud[aud["frac_removida"] > MAX_FRAC_REMOVIDA]
    if len(ruins):
        alertas.append(f"{len(ruins)} (janela, faixa) com remocao acima de "
                       f"{MAX_FRAC_REMOVIDA:.0%}")
    baixa = aud[aud["cobertura_cotahist"] < MIN_COBERTURA_UNIVERSO]
    if len(baixa):
        alertas.append(f"{len(baixa)} (janela, faixa) com cobertura COTAHIST "
                       f"abaixo de {MIN_COBERTURA_UNIVERSO:.0%}")

    if alertas:
        log.warning("=" * 68)
        for a in alertas:
            log.warning("ALERTA: %s", a)
        if len(baixa):
            log.warning("piores coberturas:\n%s",
                        baixa.nsmallest(10, "cobertura_cotahist")[
                            ["janela", "faixa", "n_selecionados",
                             "tickers_com_retorno_cotahist",
                             "cobertura_cotahist"]].to_string(index=False))
        log.warning("Nao estimar a rede antes de revisar estes casos.")
        log.warning("=" * 68)
    else:
        log.info("Nenhum universo perdeu cobertura material por faixa top-N, "
                 "globalmente. Setor e lider/seguidora seguem nao auditados.")

    gravar_manifesto(execucao, cfg.to_dict(), status="concluida", extras={
        "fonte_retornos_estimacao": cfg.dados.fonte_retornos_estimacao,
        "fonte_retornos_pnl": cfg.dados.fonte_retornos_pnl,
        "mascarar_pregao_seguinte": cfg.dados.mascarar_pregao_seguinte_ao_evento,
        "denominadores": denominadores,
        "alertas": alertas,
        "dimensoes_nao_auditadas": pendentes,
        "escopo_da_afirmacao_de_cobertura": ("por faixa top-N, global; nao "
                                             "verificada por setor nem por "
                                             "papel de lider/seguidora"),
        "retornos_extremos_triagem": int(len(ext)),
        "extremos_com_prioridade_documental": (
            int((ext["pode_afetar_estimacao"] | ext["pode_afetar_sinal"]).sum())
            if len(ext) else 0),
        "extremos_tratados_automaticamente": False,
        "evidencia_por_token": val["classificacao_evidencia"].value_counts().to_dict(),
        "corte_amostra_minima": retornos.MIN_FRONTEIRAS_COM_REFERENCIA,
        "escopo_dos_extremos": ("estado EXCLUSIVO na data decide efeito; a "
                                "tabela alcance_cumulativo_por_faixa e leitura "
                                "cumulativa, onde top100 contem os menores. "
                                "Limite superior: sem dedupe por emissor, sem "
                                "filtro setorial e sem formacao de pares"),
        "evidencia_zero_nao_sustentados": (
            "nenhum token foi classificado GLOBALMENTE como nao sustentado. "
            "Isso nao implica que toda fronteira removida seja materialmente "
            "relevante - ES tem mediana proxima de zero e ~42% das fronteiras "
            "acima de 10 bps"),
        "rede_estimada": False,
    })
    log.info("Tabelas em %s", T)
    log.info("AUDITORIA APENAS: nenhuma rede foi estimada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
