"""
Orquestrador temporal do walk-forward.

E o unico modulo que sabe qual data e "hoje" em cada momento da simulacao. Os
outros recebem dados ja recortados e por isso nao tem como espiar o futuro por
conta propria.

O laco de cada janela:

    universo = selecionar_universo(treino)     # point-in-time
    pares    = gerar_pares(universo, treino)
    rede     = estimar_leadlag(treino, pares)
    rede     = aplicar_fdr(rede)               # so os testes da janela

    sinais   = gerar_sinais(rede, teste)       # rede congelada
    posicoes = construir_posicoes(sinais)
    pnl      = aplicar_custos(posicoes, teste)

A rede nunca e estimada sobre a amostra inteira e depois entregue ao backtest -
seria descobrir o melhor par com o gabarito na mao. Nenhuma funcao de estimacao
recebe DataFrame com data posterior ao fim do treino da janela, e
tests/test_anti_lookahead.py cobra isso.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.config import PeriodoConfig, WalkForwardConfig


@dataclass(frozen=True)
class Janela:
    """
    Uma janela do walk-forward.

    Treino fechado a esquerda e aberto a direita, teste comecando exatamente
    onde o treino termina. Nenhum pregao cai nas duas janelas ao mesmo tempo,
    que seria um vazamento direto de informacao futura.
    """

    indice: int
    treino_inicio: pd.Timestamp
    treino_fim: pd.Timestamp      # exclusivo
    teste_inicio: pd.Timestamp    # == treino_fim
    teste_fim: pd.Timestamp       # exclusivo

    @property
    def rotulo(self) -> str:
        return f"{self.teste_inicio:%Y-%m}"


def gerar_janelas(
    inicio: date,
    fim: date,
    cfg: WalkForwardConfig | None = None,
    treino_meses: int | None = None,
) -> list[Janela]:
    """
    Janelas deslizantes de treino e teste.

    Com os padroes (24 meses de treino, 3 de teste) e a amostra de 2015 ate
    hoje saem cerca de 38 janelas fora da amostra.

    `treino_meses` sobrescreve o padrao para as robustezas de 12 e 36 meses,
    sem precisar mexer no config.
    """
    cfg = cfg or WalkForwardConfig()
    treino = treino_meses or cfg.treino_meses

    janelas: list[Janela] = []
    t_ini = pd.Timestamp(inicio)
    limite = pd.Timestamp(fim)
    i = 0
    while True:
        t_fim = t_ini + pd.DateOffset(months=treino)
        teste_fim = t_fim + pd.DateOffset(months=cfg.teste_meses)
        if teste_fim > limite:
            break
        janelas.append(Janela(i, t_ini, t_fim, t_fim, teste_fim))
        t_ini = t_ini + pd.DateOffset(months=cfg.teste_meses)
        i += 1
    return janelas


def janelas_do_periodo(
    painel: pd.DataFrame,
    cfg_periodo: PeriodoConfig | None = None,
    cfg_wf: WalkForwardConfig | None = None,
    treino_meses: int | None = None,
) -> list[Janela]:
    """Janelas cobrindo o periodo efetivamente disponivel no painel."""
    cfg_periodo = cfg_periodo or PeriodoConfig()
    fim = cfg_periodo.fim or painel["DATA"].max().date()
    return gerar_janelas(cfg_periodo.inicio, fim, cfg_wf, treino_meses)


def rodar_walk_forward(
    cot: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    tabela_setores: pd.DataFrame,
    ret_estimacao: pd.DataFrame,
    ret_pnl: pd.DataFrame,
    janelas: list[Janela],
    cfg,
    combos: list[tuple[str, str]] | None = None,
) -> dict:
    """
    O laco completo: para cada janela, estima no treino e opera no teste.

    A rede sai uma vez por janela e as combinacoes (regra de selecao x modo de
    carteira) derivam posicoes do mesmo resultado congelado. Reestimar por
    combinacao multiplicaria tempo sem gerar informacao nova.

    Nao ha aleatoriedade aqui - os placebos ficam na etapa de estimacao -,
    entao duas chamadas com as mesmas entradas dao exatamente as mesmas series.

    Devolve, por combinacao: posicoes, P&L bruto e liquido, cenarios de
    aluguel, pernas, exposicoes e resumo. E, uma vez so: a rede por janela com
    FDR, os pares selecionados por regra e os diagnosticos.
    """
    # Import local para nao fechar ciclo: estes modulos importam Janela daqui.
    from src import custos, estrategia, multiplos_testes, pares, retornos, universo
    from src.leadlag import estimar_rede

    combos = combos or [(regra, modo)
                        for regra in cfg.multiplos_testes.regras_selecao
                        for modo in cfg.estrategia.modos]
    vol = retornos.volatilidade_realizada(ret_pnl,
                                          cfg.estrategia.janela_vol_dias)

    posicoes_por_combo: dict[tuple[str, str], pd.DataFrame] = {}
    eventos_por_combo: dict[tuple[str, str], list] = {}
    redes, selecionados, diagnosticos = [], {r: [] for r, _ in combos}, []

    for j in janelas:
        sel = universo.selecionar_universo(cot, j, cfg.liquidez)
        data_formacao = calendario[calendario < j.treino_fim].max()
        sel = pares.com_setor_vigente(sel, tabela_setores, data_formacao)
        p = pares.gerar_pares(sel, cfg.liquidez.faixas)
        p = p[p["faixa_minima"] <= cfg.liquidez.top_n].reset_index(drop=True)

        rede = estimar_rede(ret_estimacao, p, j, cfg.leadlag)
        rede, _ = multiplos_testes.aplicar_fdr_janela(rede, cfg.multiplos_testes)
        redes.append(rede)

        escolhidos: dict[str, pd.DataFrame] = {}
        for regra in {r for r, _ in combos}:
            escolhidos[regra] = multiplos_testes.selecionar_pares(
                rede, regra, cfg.multiplos_testes)
            selecionados[regra].append(escolhidos[regra])

        for regra, modo in combos:
            pos, diag = estrategia.construir_posicoes_janela(
                escolhidos[regra], ret_estimacao, vol, j, calendario,
                cfg.estrategia, cfg.walk_forward, modo)
            eventos_por_combo.setdefault((regra, modo), []).extend(
                diag.pop("eventos", []))
            diag.update({"regra": regra,
                         "pares_selecionados": len(escolhidos[regra])})
            diagnosticos.append(diag)
            if pos.empty:
                continue
            atual = posicoes_por_combo.get((regra, modo))
            posicoes_por_combo[(regra, modo)] = (
                pos if atual is None else atual.add(pos, fill_value=0.0))

    fronteiras = retornos.painel_fronteiras_evento(cot, calendario)
    resultados: dict[str, dict] = {}
    for combo in combos:
        rotulo = f"{combo[0]}_{combo[1]}"
        pos = posicoes_por_combo.get(combo)
        if pos is None:
            resultados[rotulo] = {"vazio": True}
            continue
        pos = pos.reindex(calendario).fillna(0.0)
        bruto, diag_pnl = estrategia.pnl_bruto(pos, ret_pnl)
        giro = estrategia.turnover_diario(pos)
        liquido = custos.aplicar_custos(bruto, pos, giro, cfg.custos)
        resultados[rotulo] = {
            "posicoes": pos,
            "eventos": pd.DataFrame(eventos_por_combo.get(combo, [])),
            "pnl": pd.DataFrame({"bruto": bruto, "liquido": liquido}),
            "cenarios_aluguel": custos.cenarios_aluguel(bruto, pos, giro,
                                                        cfg.custos),
            "pernas": estrategia.separar_pernas(pos, ret_pnl),
            "exposicoes": estrategia.exposicoes(pos),
            "turnover": giro,
            "resumo": {
                "regra": combo[0], "modo": combo[1],
                "pnl_bruto_total": float(bruto.sum()),
                "pnl_liquido_total": float(liquido.sum()),
                "turnover_medio_diario": float(giro.mean()),
                **diag_pnl,
                **retornos.medir_travessias_de_evento(pos, fronteiras),
            },
        }

    return {
        "combos": resultados,
        "rede": pd.concat(redes, ignore_index=True),
        "selecionados": {r: pd.concat(v, ignore_index=True)
                         for r, v in selecionados.items()},
        "diagnosticos": pd.DataFrame(diagnosticos),
    }
