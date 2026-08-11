"""
Custos de transacao: meio-spread, slippage, corretagem, emolumentos e aluguel.

O que decide se a difusao encontrada e negociavel e o P&L liquido daqui. Um
efeito que existe no bruto e morre nos custos e resultado valido, nao fracasso.

O aluguel (BTC) e o custo que costuma matar long-short no Brasil, e incide so
na perna vendida. A taxa vai de ~1% a.a. em blue chip a 20%+ a.a. em small cap,
e em muitos papeis nem ha doador - como a seguidora e, por construcao, a ponta
menos liquida do par, a venda cai justamente onde e caro ou impossivel. Sem
serie point-in-time de taxa, rodamos cenarios em vez de custo observado.

O modulo so cobra a conta: nao gera sinal nem dimensiona posicao. A invariante
que os testes cobram e P&L liquido <= P&L bruto, sempre.
"""

from __future__ import annotations

import pandas as pd

from src.config import CustosConfig


def custo_por_ponta_bps(cfg: CustosConfig | None = None) -> float:
    """Custo de uma ponta de execucao, em bps sobre o valor negociado."""
    cfg = cfg or CustosConfig()
    return (cfg.meio_spread_bps + cfg.slippage_bps
            + cfg.corretagem_bps + cfg.emolumentos_bps)


def custo_giro(
    turnover: pd.Series,
    cfg: CustosConfig | None = None,
) -> pd.Series:
    """
    Custo diario do giro: cada unidade de turnover paga uma ponta.

    `turnover` e a fracao do capital negociada no dia (soma dos |deltas| de
    posicao), entao abrir e fechar uma posicao inteira ja aparece como duas
    pontas.
    """
    cfg = cfg or CustosConfig()
    c = turnover.abs() * custo_por_ponta_bps(cfg) / 1e4
    c.name = "custo_giro"
    return c


def custo_aluguel(
    posicoes: pd.DataFrame,
    taxa_anual: float,
    cfg: CustosConfig | None = None,
) -> pd.Series:
    """
    Aluguel pro-rata por dia util, so sobre a exposicao vendida.

    5% a.a. sobre 3 pregoes custa 5% x 3/252, nao 5%. Posicao comprada nao toma
    acao emprestada e nao paga nada.
    """
    cfg = cfg or CustosConfig()
    vendida = posicoes.clip(upper=0.0).abs().sum(axis=1)
    c = vendida * taxa_anual / cfg.dias_uteis_ano
    c.name = "custo_aluguel"
    return c


def aplicar_custos(
    pnl_bruto: pd.Series,
    posicoes: pd.DataFrame,
    turnover: pd.Series,
    cfg: CustosConfig | None = None,
    taxa_aluguel_anual: float | None = None,
) -> pd.Series:
    """
    Serie diaria liquida de execucao: bruto menos giro menos aluguel.

    Os dois custos sao nao-negativos por construcao, o que garante a invariante
    de que o liquido nunca supera o bruto.
    """
    cfg = cfg or CustosConfig()
    taxa = (cfg.aluguel_cenario_base if taxa_aluguel_anual is None
            else taxa_aluguel_anual)
    liquido = (pnl_bruto
               - custo_giro(turnover, cfg).reindex(pnl_bruto.index, fill_value=0.0)
               - custo_aluguel(posicoes, taxa, cfg).reindex(pnl_bruto.index,
                                                            fill_value=0.0))
    liquido.name = "pnl_liquido"
    return liquido


def cenarios_aluguel(
    pnl_bruto: pd.Series,
    posicoes: pd.DataFrame,
    turnover: pd.Series,
    cfg: CustosConfig | None = None,
) -> pd.DataFrame:
    """Uma serie liquida por cenario de aluguel, rotulada pela taxa anual."""
    cfg = cfg or CustosConfig()
    return pd.DataFrame({
        f"aluguel_{taxa:.0%}": aplicar_custos(pnl_bruto, posicoes, turnover,
                                              cfg, taxa)
        for taxa in cfg.aluguel_cenarios_anual
    })


def ir_ilustrativo(pnl_anual: pd.Series, cfg: CustosConfig | None = None) -> pd.Series:
    """
    Aproximacao de IR: aliquota sobre o lucro dos anos positivos.

    Nao compensa prejuizo entre anos nem separa swing de day trade. E um teto
    de imposto, e nao o custo tributario de verdade.
    """
    cfg = cfg or CustosConfig()
    imposto = pnl_anual.clip(lower=0.0) * cfg.ir_aliquota_ilustrativa
    liquido = pnl_anual - imposto
    liquido.name = "pnl_apos_ir_ilustrativo"
    return liquido
