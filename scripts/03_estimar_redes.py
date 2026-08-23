"""
Estima as redes lead-lag, janela a janela.

Para cada janela do walk-forward, so com o treino dela:

  1. universo final e pares point-in-time (mesma logica da etapa 2);
  2. regressao lider -> seguidora no lag principal, com HAC, para todos os
     pares ate o top100 (a coluna faixa_minima da a estratificacao);
  3. Benjamini-Hochberg dentro da janela, por faixa;
  4. selecao pelas duas regras, fdr e top_k, mantidas separadas;
  5. na faixa principal: lags alternativos, direcao invertida, reestimativa sem
     dias de preco velho, subconjunto sempre negociado e placebo de seguidoras
     embaralhadas.

So estimacao: nenhum sinal, posicao, custo ou P&L sai daqui.

Uso:
    .venv\\Scripts\\python.exe scripts\\03_estimar_redes.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import (backtest, dados, leadlag, multiplos_testes, nao_sincronia,  # noqa: E402
                 pares, qualidade_dados, retornos, setores, universo)
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SETORES = RAIZ / "data" / "reference" / "setores_b3.csv"


def main(execucao: Execucao | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino-meses", type=int, default=None,
                    help="robustez: sobrescreve os 24m de treino (ex.: 12, 36)")
    ap.add_argument("--sem-placebos", action="store_true",
                    help="pula os embaralhamentos (robustez fica mais rapida)")
    args, _ = ap.parse_known_args()

    cfg = CONFIG_PADRAO
    rotulo = ("etapa3" if args.treino_meses is None
              else f"etapa3_treino{args.treino_meses}")
    execucao = execucao or criar_execucao(rotulo)
    log = configurar_log(execucao, "03_redes")
    T = execucao.tabelas
    if args.treino_meses is not None:
        log.info("Robustez: treino de %dm no lugar dos %dm padrao",
                 args.treino_meses, cfg.walk_forward.treino_meses)

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
    negocios = retornos.painel_numero_negocios(cot).reindex(calendario)
    precos = (cot.pivot_table(index="DATA", columns="CODNEG", values="PREULT",
                              aggfunc="last").reindex(calendario))
    ret = retornos.retornos_preco_bruto_cotahist(cot, calendario, volume)
    janelas = backtest.janelas_do_periodo(cot, cfg.periodo, cfg.walk_forward,
                                          treino_meses=args.treino_meses)
    log.info("%d janelas | painel de retornos %d x %d", len(janelas),
             *ret.shape)

    top_n = cfg.liquidez.top_n
    redes, alternativos, invertidas, sem_velho, sempre_neg = [], [], [], [], []
    fdr_resumos, placebo_dists, placebo_resumos = [], [], []
    sel_fdr, sel_topk = [], []

    for j in janelas:
        sel = universo.selecionar_universo(cot, j, cfg.liquidez)
        data_formacao = calendario[calendario < j.treino_fim].max()
        sel = pares.com_setor_vigente(sel, tabela_setores, data_formacao)
        p = pares.gerar_pares(sel, cfg.liquidez.faixas)
        p_principal = p[p["faixa_minima"] <= top_n].reset_index(drop=True)

        rede = leadlag.estimar_rede(ret, p, j, cfg.leadlag)

        # BH por faixa, sempre dentro da janela. A tabela principal fica com a
        # correcao da faixa principal, e as demais vao para o resumo por faixa.
        com_fdr = {}
        for f in cfg.liquidez.faixas:
            sub = rede[rede["faixa_minima"] <= f].reset_index(drop=True)
            if sub.empty:
                continue
            corrigida, resumo = multiplos_testes.aplicar_fdr_janela(
                sub, cfg.multiplos_testes)
            resumo["faixa"] = f"top{f}"
            fdr_resumos.append(resumo)
            com_fdr[f] = corrigida
        rede_principal = com_fdr[top_n]
        redes.append(rede_principal)

        sel_fdr.append(multiplos_testes.selecionar_pares(
            rede_principal, "fdr", cfg.multiplos_testes))
        sel_topk.append(multiplos_testes.selecionar_pares(
            rede_principal, "top_k", cfg.multiplos_testes))

        alternativos.append(leadlag.estimar_lags(ret, p_principal, j,
                                                 cfg.leadlag))
        invertidas.append(leadlag.estimar_direcao_invertida(
            ret, p_principal, j, cfg.leadlag))

        mascara = nao_sincronia.filtrar_dias_preco_velho(
            volume, negocios, precos, j, cfg.nao_sincronia)
        sem_velho.append(nao_sincronia.reestimar_sem_preco_velho(
            ret, mascara, p_principal, j, cfg.leadlag))

        limpos = set(nao_sincronia.subconjunto_sempre_negociado(volume, j))
        p_limpo = p_principal[
            p_principal["lider"].isin(limpos)
            & p_principal["seguidora"].isin(limpos)].reset_index(drop=True)
        rede_limpa = leadlag.estimar_rede(ret, p_limpo, j, cfg.leadlag)
        rede_limpa["reestimativa"] = "sempre_negociado"
        sempre_neg.append(rede_limpa)

        if cfg.placebo.ativar and not args.sem_placebos and len(p_principal):
            dist, resumo = nao_sincronia.rodar_placebos(
                ret, p_principal, j, cfg.placebo, cfg.leadlag,
                seed=cfg.seed + j.indice)
            dist["janela"] = j.rotulo
            placebo_dists.append(dist)
            placebo_resumos.append(resumo)

        n_fdr = int(rede_principal["aprovado_fdr"].sum())
        log.info(
            "%s | %3d pares top%d | beta mediano %+.4f | FDR aprova %2d | "
            "placebo p=%.3f | sempre_negociado %d pares",
            j.rotulo, len(p_principal), top_n,
            rede_principal["beta"].median(), n_fdr,
            placebo_resumos[-1]["p_empirico_mediana"] if placebo_resumos else float("nan"),
            len(p_limpo))

    tabelas = {
        "rede_por_janela.csv": pd.concat(redes, ignore_index=True),
        "lags_alternativos.csv": pd.concat(alternativos, ignore_index=True),
        "direcao_invertida.csv": pd.concat(invertidas, ignore_index=True),
        "reestimativa_sem_preco_velho.csv": pd.concat(sem_velho,
                                                      ignore_index=True),
        "rede_sempre_negociado.csv": pd.concat(sempre_neg, ignore_index=True),
        "fdr_resumo_por_faixa.csv": pd.DataFrame(fdr_resumos),
        "pares_selecionados_fdr.csv": pd.concat(sel_fdr, ignore_index=True),
        "pares_selecionados_top_k.csv": pd.concat(sel_topk, ignore_index=True),
    }
    if placebo_resumos:
        tabelas["placebo_resumo.csv"] = pd.DataFrame(placebo_resumos)
        tabelas["placebo_distribuicao.csv"] = pd.concat(placebo_dists,
                                                        ignore_index=True)
    for nome, df in tabelas.items():
        df.to_csv(T / nome, index=False, encoding="utf-8")

    rede_toda = tabelas["rede_por_janela.csv"]
    resumo_fdr = tabelas["fdr_resumo_por_faixa.csv"]
    aprovados_top_n = resumo_fdr[resumo_fdr["faixa"] == f"top{top_n}"]
    placebo = tabelas.get("placebo_resumo.csv", pd.DataFrame())

    log.info("--- sintese da estimacao (faixa principal top%d) ---", top_n)
    log.info("beta mediano geral %+.4f | janelas com FDR>0: %d de %d | "
             "aprovados FDR mediana %.0f",
             rede_toda["beta"].median(),
             int((aprovados_top_n["aprovados_fdr"] > 0).sum()), len(janelas),
             aprovados_top_n["aprovados_fdr"].median())
    if len(placebo):
        log.info("placebo: p empirico mediano %.3f | janelas com p<0.05: %d",
                 placebo["p_empirico_mediana"].median(),
                 int((placebo["p_empirico_mediana"] < 0.05).sum()))

    gravar_manifesto(
        execucao, cfg.to_dict(),
        arquivos_dados=[SETORES],
        status="concluida",
        extras={
            "janelas": len(janelas),
            "treino_meses": args.treino_meses or cfg.walk_forward.treino_meses,
            "placebos_executados": bool(placebo_resumos),
            "faixa_principal": f"top{top_n}",
            "pares_estimados_faixa_principal": int(
                rede_toda["p_valor"].notna().sum()),
            "beta_mediano_faixa_principal": float(rede_toda["beta"].median()),
            "janelas_com_aprovacao_fdr": int(
                (aprovados_top_n["aprovados_fdr"] > 0).sum()),
            "placebo_p_empirico_mediano": float(
                placebo["p_empirico_mediana"].median()) if len(placebo) else None,
            "regras_de_selecao_separadas": ["fdr", "top_k"],
            "rede_estimada": True,
            "pnl_calculado": False,
        })
    log.info("Tabelas em %s", T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
