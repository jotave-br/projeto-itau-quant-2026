"""
Transformacao da rede em posicoes.

Lider subiu, compra a seguidora; lider caiu, vende a seguidora. O sinal e
observado no fechamento de t e a posicao so abre no fechamento de
t + `defasagem_execucao_dias`, senao estariamos negociando a um preco que ainda
nao existia quando o sinal saiu.

Convencao de tempo, fixada aqui para nao virar bug la na frente:

  posicoes.loc[d] = exposicao detida no fechamento de d
  pnl(d)          = posicoes(d-1) x retorno(d)

  sinal em t -> abre no fechamento de t+1 -> ativo em t+1..t+k -> P&L nos
  retornos de t+2..t+k+1.

Com holding de k pregoes e sinal diario, a operacao de segunda se sobrepoe as
de terca e quarta. Cada safra entra com peso 1/k, entao a exposicao total nao
multiplica capital e sai uma unica serie diaria de P&L. Operacao sobreposta nao
e observacao independente, e e por isso que metricas.py usa bootstrap por
blocos.

Os pesos sao proporcionais ao inverso da volatilidade da seguidora (rolling, so
passado), normalizados para exposicao bruta 1 por safra, com o teto por posicao
aplicado por corte e sem renormalizar para cima: com poucos pares, safra
subinvestida e melhor que safra concentrada. O `vol_alvo_anual` cancela na
normalizacao (vol_alvo/vol_i normalizado e igual a 1/vol_i normalizado) e fica
guardado para uma versao com alavancagem explicita.

Os dois paineis de retorno sao propositais: o sinal usa o painel mascarado da
estimacao, porque o degrau do dia ex da lider nao e noticia e dispararia venda
falsa da seguidora. O P&L usa PREULT bruto sem mascara e, portanto, mede retorno
de preco, sem creditar proventos na ponta comprada nem debita-los na vendida.
As travessias de evento registram a incidencia dessa limitacao.

long_short e o teste limpo do sinal. long_only descarta as vendas e renormaliza
entre as compras - nao e "a long-short sem a metade cara", e outra hipotese, e
por isso as pernas da long-short tambem saem separadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import Janela
from src.config import EstrategiaConfig, WalkForwardConfig

MODOS = ("long_short", "long_only")


def construir_posicoes_janela(
    pares_selecionados: pd.DataFrame,
    ret_sinal: pd.DataFrame,
    vol_seguidora: pd.DataFrame,
    janela: Janela,
    calendario: pd.DatetimeIndex,
    cfg: EstrategiaConfig | None = None,
    cfg_wf: WalkForwardConfig | None = None,
    modo: str = "long_short",
) -> tuple[pd.DataFrame, dict]:
    """
    Posicoes por (data, ticker) geradas pelos sinais da janela de teste.

    A rede chega congelada, com os pares ja selecionados no treino. A ultima
    safra pode passar alguns pregoes de `teste_fim` porque trade aberto no fim
    do teste ainda precisa fechar; nenhuma decisao usa dado posterior ao sinal.

    Devolve tambem os diagnosticos: sinais gerados, descartados por falta de
    volatilidade e pares sem retorno de sinal.
    """
    cfg = cfg or EstrategiaConfig()
    cfg_wf = cfg_wf or WalkForwardConfig()
    if modo not in MODOS:
        raise ValueError(f"modo desconhecido: {modo!r} (validos: {MODOS})")

    cal = pd.DatetimeIndex(calendario)
    k = cfg.holding_dias
    defasagem = cfg_wf.defasagem_execucao_dias
    dias_sinal = cal[(cal >= janela.teste_inicio) & (cal < janela.teste_fim)]

    posicoes: dict[pd.Timestamp, dict[str, float]] = {}
    eventos: list[dict] = []
    diag = {"janela": janela.rotulo, "modo": modo, "sinais": 0,
            "descartados_sem_vol": 0, "descartados_sem_retorno_sinal": 0,
            "eventos": eventos}

    for t in dias_sinal:
        pos_t = cal.get_loc(t)
        pesos: dict[str, float] = {}
        for _, par in pares_selecionados.iterrows():
            lider, seguidora = par["lider"], par["seguidora"]
            r = (ret_sinal.at[t, lider]
                 if lider in ret_sinal.columns else np.nan)
            if pd.isna(r) or r == 0:
                diag["descartados_sem_retorno_sinal"] += pd.isna(r)
                continue
            direcao = float(np.sign(r))
            if modo == "long_only" and direcao < 0:
                continue
            v = (vol_seguidora.at[t, seguidora]
                 if seguidora in vol_seguidora.columns else np.nan)
            if pd.isna(v) or v <= 0:
                diag["descartados_sem_vol"] += 1
                continue
            # o mesmo ticker pode ser seguidora de mais de um par no dia
            pesos[seguidora] = pesos.get(seguidora, 0.0) + direcao / v
            # registro no nivel do sinal, para calcular o CAR depois
            eventos.append({"data": t, "lider": lider, "seguidora": seguidora,
                            "direcao": direcao, "janela": janela.rotulo})

        if not pesos:
            continue
        diag["sinais"] += len(pesos)
        bruto = sum(abs(w) for w in pesos.values())
        finais = {tk: np.sign(w) * min(abs(w) / bruto,
                                       cfg.peso_maximo_por_posicao)
                  for tk, w in pesos.items()}

        # safra 1/k, do fechamento de t+defasagem ate t+defasagem+k-1
        for passo in range(k):
            idx = pos_t + defasagem + passo
            if idx >= len(cal):
                break
            dia = cal[idx]
            destino = posicoes.setdefault(dia, {})
            for tk, w in finais.items():
                destino[tk] = destino.get(tk, 0.0) + w / k

    if not posicoes:
        return pd.DataFrame(index=cal[0:0]), diag
    pos = pd.DataFrame.from_dict(posicoes, orient="index").sort_index()
    return pos.fillna(0.0), diag


def pnl_bruto(
    posicoes: pd.DataFrame,
    ret_pnl: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    """
    P&L diario bruto: posicao do fechamento anterior vezes o retorno do dia.

    Posicao-dia sem retorno entra como zero e vai para o diagnostico. Com a
    marcacao a ultimo preco isso nao deveria acontecer, entao contador acima de
    zero e caso para investigar.
    """
    r = ret_pnl.reindex(index=posicoes.index, columns=posicoes.columns)
    exposta = posicoes.shift(1).fillna(0.0)
    sem_retorno = int(((exposta != 0) & r.isna()).sum().sum())
    pnl = (exposta * r.fillna(0.0)).sum(axis=1)
    pnl.name = "pnl_bruto"
    return pnl, {"posicao_dias_sem_retorno": sem_retorno}


def separar_pernas(
    posicoes: pd.DataFrame,
    ret_pnl: pd.DataFrame,
) -> pd.DataFrame:
    """
    P&L da ponta comprada e da vendida, separadas.

    Noticia ruim se difunde diferente de noticia boa. Se o lead-lag for
    assimetrico, a long-only deixa de ser proxy da long-short, e isso so
    aparece olhando as pernas.
    """
    r = (ret_pnl.reindex(index=posicoes.index, columns=posicoes.columns)
         .fillna(0.0))
    exposta = posicoes.shift(1).fillna(0.0)
    return pd.DataFrame({
        "perna_comprada": (exposta.clip(lower=0.0) * r).sum(axis=1),
        "perna_vendida": (exposta.clip(upper=0.0) * r).sum(axis=1),
    })


def turnover_diario(posicoes: pd.DataFrame) -> pd.Series:
    """
    Fracao do capital negociada por dia: soma dos |deltas| de posicao.

    Base do custo de giro, ja que cada unidade de turnover paga uma ponta. A
    primeira linha conta como abertura integral.
    """
    delta = posicoes.diff()
    delta.iloc[0] = posicoes.iloc[0]
    t = delta.abs().sum(axis=1)
    t.name = "turnover"
    return t


def exposicoes(posicoes: pd.DataFrame) -> pd.DataFrame:
    """Exposicao bruta e liquida por dia."""
    return pd.DataFrame({
        "bruta": posicoes.abs().sum(axis=1),
        "liquida": posicoes.sum(axis=1),
    })
