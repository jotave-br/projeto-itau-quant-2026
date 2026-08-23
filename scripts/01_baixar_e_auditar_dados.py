"""
Baixa, cacheia e audita os dados.

Sequencia: COTAHIST anual da B3 de 2015 em diante, validado e cacheado; filtro
do universo candidato (a vista, lote padrao, allowlist de papel); calendario de
pregoes derivado dos proprios dados; auditoria por ticker gravada na pasta da
execucao. Falha explicitamente em inconsistencia de layout ou tipo de
instrumento nao classificado.

As tabelas de frequencia de ESPECI, CODBDI e TPMERC sao regeradas a cada
execucao em vez de congeladas em data/reference/, porque sao dado derivado e um
arquivo desatualizado seria pior que arquivo nenhum. Ja a allowlist e a denylist
ficam versionadas em data/reference/classificacao_instrumentos.csv: sao decisao,
nao resultado.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import backtest, dados, qualidade_dados, universo  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao  # noqa: E402


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa1")
    log = configurar_log(execucao, "01_dados")

    log.info("Carregando COTAHIST de %s ate hoje", cfg.periodo.inicio)
    bruto, relatorios = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=False)
    log.info("Registros brutos: %s", f"{len(bruto):,}")

    pool = bruto[
        bruto["TPMERC"].isin(cfg.dados.cotahist_tipo_mercado)
        & bruto["CODBDI"].isin(cfg.dados.cotahist_codbdi)
    ]
    for nome, tabela in dados.inventario_categorias(pool).items():
        destino = execucao.tabelas / f"frequencias_{nome.lower()}.csv"
        tabela.to_csv(destino, encoding="utf-8")
        log.info("Frequencias de %-11s -> %d categorias", nome, len(tabela))

    acoes = dados.filtrar_acoes_a_vista(bruto)
    log.info(
        "Filtro de instrumento: %s tickers no pool -> %s acoes (%s removidos)",
        f"{pool['CODNEG'].nunique():,}",
        f"{acoes['CODNEG'].nunique():,}",
        f"{pool['CODNEG'].nunique() - acoes['CODNEG'].nunique():,}",
    )

    divergencias = dados.conferir_especi_contra_isin(pool)
    log.info(
        "Checagem redundante ESPECI x ISIN: %s linhas divergentes (%s tickers)",
        f"{len(divergencias):,}", divergencias["CODNEG"].nunique(),
    )
    if len(divergencias):
        divergencias.drop_duplicates("CODNEG").to_csv(
            execucao.tabelas / "divergencias_especi_isin.csv", index=False,
            encoding="utf-8",
        )

    calendario = qualidade_dados.calendario_pregoes(acoes)
    log.info(
        "Calendario derivado dos dados: %s pregoes, de %s a %s",
        f"{len(calendario):,}",
        calendario.min().date(), calendario.max().date(),
    )
    pd.Series(calendario, name="pregao").to_csv(
        execucao.tabelas / "calendario_pregoes.csv", index=False, encoding="utf-8"
    )

    log.info("Auditando %s tickers...", f"{acoes['CODNEG'].nunique():,}")
    aud = qualidade_dados.auditar_tickers(
        acoes, calendario, cfg.nao_sincronia, cfg.liquidez
    )
    aud.to_csv(execucao.tabelas / "auditoria_tickers.csv", encoding="utf-8")

    resumo = qualidade_dados.resumo_auditoria(aud)
    for chave, valor in resumo.items():
        log.info("  %-28s %s", chave, f"{valor:,.4f}" if isinstance(valor, float)
                 else f"{valor:,}")

    janelas = backtest.janelas_do_periodo(acoes, cfg.periodo, cfg.walk_forward)
    log.info("Janelas de walk-forward: %d (treino %dm / teste %dm), de %s a %s",
             len(janelas), cfg.walk_forward.treino_meses,
             cfg.walk_forward.teste_meses,
             janelas[0].rotulo, janelas[-1].rotulo)

    melhor_pos = universo.melhor_posicao_por_ticker(acoes, janelas, cfg.liquidez)
    melhor_pos.to_csv(execucao.tabelas / "melhor_posicao_pit.csv", encoding="utf-8")
    log.info("Tickers que entraram no top-100 em alguma janela: %d",
             int((melhor_pos["melhor_posicao"] <= 100).sum()))

    trocas = qualidade_dados.candidatos_mudanca_ticker(acoes)
    log.info("Candidatos a mudanca de ticker (brutos): %d", len(trocas))
    if len(trocas):
        trocas.to_csv(execucao.tabelas / "candidatos_mudanca_ticker.csv",
                      index=False, encoding="utf-8")

    revisao = qualidade_dados.tabela_revisao_mudanca_ticker(
        acoes, trocas, melhor_pos, calendario
    )
    log.info("Candidatos materialmente relevantes para revisao: %d", len(revisao))

    # O yfinance emenda tickers renomeados por conta propria. Como o COTAHIST e
    # o registro oficial da bolsa, da para auditar a emenda: no periodo do
    # codigo antigo, o preco
    # reportado sob o codigo novo deveria reproduzir o registro da B3. Daqui sai
    # evidencia e prioridade, nao autorizacao de continuidade.
    if len(revisao):
        log.info("Validando %d series do yfinance contra o COTAHIST...",
                 revisao["ticker_novo"].nunique())
        cache_yf: dict[str, tuple] = {}

        def _yf(tk: str):
            if tk not in cache_yf:
                cache_yf[tk] = dados.baixar_yfinance(tk, cfg.periodo.inicio)
            return cache_yf[tk]

        validacoes, por_cadeia = [], 0
        for _, r in revisao.iterrows():
            ant, nov = r["ticker_antigo"], r["ticker_novo"]
            yf_df, meta_yf = _yf(nov)
            met = qualidade_dados.metricas_validacao_yf(acoes, ant, nov, yf_df, meta_yf)
            cls = qualidade_dados.classificar_validacao_yf(met)

            # Segunda tentativa pelo terminal da cadeia: o Yahoo costuma migrar
            # o historico inteiro para o ultimo codigo.
            if cls["resultado_validacao_yf"] in (qualidade_dados.CAT_AUSENTE,
                                                 qualidade_dados.CAT_INCONCLUSIVO):
                terminal = qualidade_dados.ticker_terminal_da_cadeia(trocas, ant)
                if terminal and terminal != nov:
                    yf_t, meta_t = _yf(terminal)
                    met_t = qualidade_dados.metricas_validacao_yf(
                        acoes, ant, terminal, yf_t, meta_t)
                    cls_t = qualidade_dados.classificar_validacao_yf(met_t)
                    if cls_t["resultado_validacao_yf"] not in (
                            qualidade_dados.CAT_AUSENTE,
                            qualidade_dados.CAT_INCONCLUSIVO):
                        met, cls = met_t, cls_t
                        cls["motivo_validacao_yf"] = (
                            f"validado pelo terminal da cadeia ({terminal}), pois "
                            f"{nov} nao tem serie util no Yahoo; "
                            + cls["motivo_validacao_yf"])
                        por_cadeia += 1

            met.pop("_situacao", None)   # interno, nao vai para a tabela
            validacoes.append({**met, **cls})

        revisao = pd.concat(
            [revisao.reset_index(drop=True), pd.DataFrame(validacoes)], axis=1)

        if por_cadeia:
            log.info("    %d resolvido(s) pelo terminal da cadeia", por_cadeia)
        for cat, n in revisao["resultado_validacao_yf"].value_counts().items():
            log.info("    %-42s %d", cat, n)

        revisao.to_csv(execucao.tabelas / "revisao_mudanca_ticker.csv",
                       index=False, encoding="utf-8")
        for _, r in revisao.iterrows():
            log.info("    %-7s -> %-7s | %-13s -> %-13s | pos %s->%s | %s",
                     r["ticker_antigo"], r["ticker_novo"],
                     str(r["nome_antigo"])[:13], str(r["nome_novo"])[:13],
                     r["melhor_posicao_antigo"], r["melhor_posicao_novo"],
                     r["criterio_relevancia"])

    pd.DataFrame([
        {
            "ano": r["ano"],
            "registros": r["arquivo"]["cotacoes"],
            "trailer_convencao": r["arquivo"]["trailer_convencao"],
            "duplicatas_pool": r["valores"]["duplicatas_no_pool_a_vista"],
            "voltot_erro_mediano": r["valores"].get("voltot_liquidos_erro_mediano"),
            "preco_max_pool": r["valores"].get("preco_max_pool"),
            "sha256": r["meta_download"]["sha256"][:16],
        }
        for r in relatorios
    ]).to_csv(execucao.tabelas / "validacao_por_ano.csv", index=False, encoding="utf-8")

    log.info("Tabelas gravadas em %s", execucao.tabelas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
