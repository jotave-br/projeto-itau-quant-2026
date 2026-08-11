"""
Correcao de multiplas comparacoes.

Testando dezenas de pares, alguns parecem ter lead-lag forte so por sorte -
como quem joga uma moeda 8 vezes num grupo de 500 pessoas e tira cara nas 8.
Duas defesas: restringir os pares ao mesmo setor (feito em pares.py), o que ja
derruba o numero de hipoteses de centenas para dezenas, e Benjamini-Hochberg,
que sobe a barra proporcionalmente a quantidade de testes.

BH roda so sobre os testes de uma janela de treino. Corrigir sobre a amostra
inteira e depois operar seria selecionar pares com informacao do futuro, entao
`aplicar_fdr_janela` recusa tabela com mais de uma janela em vez de corrigir em
silencio.

As duas regras de selecao respondem perguntas diferentes. Para dizer se existe
difusao, o FDR e obrigatorio. Para decidir onde por dinheiro, ele e apenas uma
escolha conservadora, e pode zerar a carteira em varias janelas - por isso
tambem rodamos top_k por estatistica t. FDR vazio ja e um resultado, e o top_k
garante serie de P&L para analisar.

A hipotese e direcional: difusao implica beta positivo, entao as duas regras so
aceitam beta positivo. Beta negativo significativo vira diagnostico, nunca
posicao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MultiplosTestesConfig


def benjamini_hochberg(pvalores: pd.Series, q: float) -> pd.DataFrame:
    """
    Correcao BH: devolve `p_ajustado` e `aprovado` alinhados a entrada.

    p-valor NaN (par nao estimado) nao conta como hipotese - contar inflaria
    `m` e puniria os pares estimados pela existencia dos que nem rodaram. Sai
    com aprovado=False e p_ajustado NaN.
    """
    p = pvalores.dropna()
    m = len(p)
    out = pd.DataFrame({"p_ajustado": np.nan, "aprovado": False},
                       index=pvalores.index)
    if m == 0:
        return out

    ordem = p.sort_values()
    posicao = np.arange(1, m + 1)
    ajustado = ordem.to_numpy() * m / posicao
    # step-up: cada ajustado e o minimo dos ajustados dali para cima, para um p
    # menor nunca terminar com ajuste maior que o de um p maior.
    ajustado = np.minimum.accumulate(ajustado[::-1])[::-1]
    ajustado = np.minimum(ajustado, 1.0)

    aprovado = ordem.to_numpy() <= posicao * q / m
    # BH aprova todos os p ate o maior que passa, mesmo que algum intermediario
    # falhe na propria comparacao.
    if aprovado.any():
        corte = np.max(np.where(aprovado)[0])
        aprovado[: corte + 1] = True

    out.loc[ordem.index, "p_ajustado"] = ajustado
    out.loc[ordem.index, "aprovado"] = aprovado
    return out


def aplicar_fdr_janela(
    rede: pd.DataFrame,
    cfg: MultiplosTestesConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    BH sobre os testes de uma janela. Mais de uma janela na tabela e erro.

    Devolve a rede com `p_ajustado_bh` e `aprovado_fdr`, mais o resumo com
    hipoteses testadas, q e aprovados antes e depois da correcao.
    """
    cfg = cfg or MultiplosTestesConfig()
    janelas = rede["janela"].unique()
    if len(janelas) > 1:
        raise ValueError(
            f"aplicar_fdr_janela recebeu {len(janelas)} janelas "
            f"({sorted(janelas)[:3]}...): BH atravessando janelas seleciona "
            "pares com informacao do futuro")

    bh = benjamini_hochberg(rede["p_valor"], cfg.q_fdr)
    saida = rede.copy()
    saida["p_ajustado_bh"] = bh["p_ajustado"]
    saida["aprovado_fdr"] = bh["aprovado"]

    resumo = {
        "janela": janelas[0] if len(janelas) else None,
        "hipoteses_testadas": int(rede["p_valor"].notna().sum()),
        "pares_nao_estimados": int(rede["p_valor"].isna().sum()),
        "q_fdr": cfg.q_fdr,
        "aprovados_sem_correcao": int((rede["p_valor"] <= cfg.q_fdr).sum()),
        "aprovados_fdr": int(saida["aprovado_fdr"].sum()),
        "aprovados_fdr_beta_positivo": int(
            (saida["aprovado_fdr"] & (saida["beta"] > 0)).sum()),
    }
    return saida, resumo


def selecionar_pares(
    rede: pd.DataFrame,
    regra: str,
    cfg: MultiplosTestesConfig | None = None,
) -> pd.DataFrame:
    """
    Pares que operam na janela de teste, congelados pelo treino.

    "fdr"    aprovados no BH da propria janela, com beta positivo.
    "top_k"  os k maiores por estatistica t do treino, com beta positivo.

    A estatistica t vem da regressao de treino, nunca de desempenho posterior.
    Beta negativo fica de fora nas duas regras: operar o sinal contrario ao
    mecanismo declarado depois de observar os resultados seria uma escolha
    post hoc.
    """
    cfg = cfg or MultiplosTestesConfig()
    if regra not in cfg.regras_selecao:
        raise ValueError(f"regra desconhecida: {regra!r} "
                         f"(validas: {cfg.regras_selecao})")

    candidatos = rede[rede["beta"] > 0]
    if regra == "fdr":
        if "aprovado_fdr" not in candidatos.columns:
            raise ValueError("rede sem colunas de FDR: rode aplicar_fdr_janela "
                             "antes de selecionar pela regra 'fdr'")
        sel = candidatos[candidatos["aprovado_fdr"]]
        return sel.sort_values("estat_t", ascending=False).reset_index(drop=True)

    return (candidatos.sort_values("estat_t", ascending=False)
            .head(cfg.top_k_pares).reset_index(drop=True))
