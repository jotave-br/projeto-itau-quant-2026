"""
Formacao dos pares candidatos lider -> seguidora.

Tres regras formam um par:

  liquidez  a lider e a mais negociada da dupla. Vem da hipotese: a informacao
            chega primeiro onde tem mais gente olhando. A liquidez e sempre a
            da janela de treino, com desempate deterministico, nunca
            desempenho - escolher par por desempenho e o lookahead classico.

  setor     as duas pontas no mesmo (setor, subsetor) vigente na data de
            formacao. A razao economica e obvia (petroleo antecipa petroleo,
            nao antecipa banco) e a estatistica tambem: restringir ao grupo
            derruba as hipoteses testadas de centenas para dezenas.

  emissor   emissores diferentes nas duas pontas. A selecao de universo ja
            garante uma classe por emissor, mas checamos de novo porque um par
            PETR3 x PETR4 nao seria resultado ruim, seria defeito de
            construcao.

A tabela setorial e curada, versionada e temporal (data/reference/
setores_b3.csv), consultada sempre por data. Ticker sem classificacao
confirmada vigente na formacao fica fora dos pares da janela, sem herdar de
sucessor e sem cair para a linha mais proxima - aconteceu com ITSA4 entre
2026-07-24 e 2026-07-27.

O placebo de seguidoras embaralhadas mora aqui porque e manipulacao de pares;
quem roda em serie e nao_sincronia.rodar_placebos.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src import setores

MOTIVO_SEM_SETOR = "sem_classificacao_confirmada_na_formacao"

COLUNAS_PARES = (
    "janela", "setor", "subsetor", "lider", "seguidora",
    "emissor_lider", "emissor_seguidora",
    "posicao_lider", "posicao_seguidora",
    "liquidez_lider", "liquidez_seguidora", "faixa_minima",
)


def com_setor_vigente(
    selecao: pd.DataFrame,
    tabela_setores: pd.DataFrame,
    data_formacao,
) -> pd.DataFrame:
    """
    Anota em cada ticker selecionado o (setor, subsetor) vigente na formacao.

    `data_formacao` e o ultimo pregao do treino, a ultima data cuja informacao
    pode congelar universo e pares. Consultar o setor mais adiante usaria uma
    reclassificacao que ainda nao tinha acontecido na hora da decisao.

    Quem nao tem classificacao confirmada vigente recebe MOTIVO_SEM_SETOR e sai
    da formacao, e a exclusao fica no funil como qualquer outra.

    So consultamos quem alcancou alguma faixa: fora do top100 o setor nao
    decide nada, e a tabela curada so cobre quem alcanca faixa. Marcar o resto
    como "sem classificacao" encheria o funil de exclusao irrelevante.
    """
    r = selecao.copy()
    r["setor"] = ""
    r["subsetor"] = ""
    r["data_formacao"] = pd.Timestamp(data_formacao)

    em_faixa = (r["motivo_exclusao"] == "") & r["faixa"].str.startswith("top")
    for ticker in r.index[em_faixa]:
        linha = setores.setor_vigente(tabela_setores, ticker, data_formacao)
        if linha is None:
            r.loc[ticker, "motivo_exclusao"] = MOTIVO_SEM_SETOR
        else:
            r.loc[ticker, ["setor", "subsetor"]] = [linha["setor"],
                                                    linha["subsetor"]]
    return r


def gerar_pares(
    selecao_com_setor: pd.DataFrame,
    faixas: tuple[int, ...],
) -> pd.DataFrame:
    """
    Todos os pares candidatos lider -> seguidora de uma janela.

    Dentro de cada (setor, subsetor), toda dupla vira par com a mais liquida na
    frente. `posicao_final` ja carrega a liquidez do treino com desempate
    deterministico, entao a direcao sai de uma comparacao de posicoes, nunca de
    retornos.

    `faixa_minima` e a menor faixa que contem as duas pontas: lider na posicao
    15 com seguidora na 55 e um par top60. Com isso, as faixas viram filtro por
    coluna e ninguem precisa regenerar pares.

    A checagem de emissor e defensiva - a deduplicacao do universo ja deveria
    ter resolvido, e se nao resolveu e melhor parar.
    """
    faixa_max = max(faixas)
    c = selecao_com_setor
    candidatos = c[(c["motivo_exclusao"] == "") & (c["posicao_final"] <= faixa_max)]

    linhas = []
    for (setor, subsetor), grupo in candidatos.groupby(["setor", "subsetor"]):
        g = grupo.sort_values("posicao_final")
        for (tk_a, a), (tk_b, b) in itertools.combinations(g.iterrows(), 2):
            if a["emissor_id"] == b["emissor_id"]:
                raise ValueError(
                    f"par com o mesmo emissor nas duas pontas: {tk_a} x {tk_b} "
                    f"({a['emissor_id']}) - a selecao de universo nao foi "
                    "deduplicada")
            pos_pior = max(a["posicao_final"], b["posicao_final"])
            linhas.append({
                "janela": a["janela"],
                "setor": setor,
                "subsetor": subsetor,
                "lider": tk_a,
                "seguidora": tk_b,
                "emissor_lider": a["emissor_id"],
                "emissor_seguidora": b["emissor_id"],
                "posicao_lider": a["posicao_final"],
                "posicao_seguidora": b["posicao_final"],
                "liquidez_lider": a["liquidez"],
                "liquidez_seguidora": b["liquidez"],
                "faixa_minima": min(f for f in faixas if pos_pior <= f),
            })
    pares = pd.DataFrame(linhas, columns=COLUNAS_PARES)
    return pares.sort_values(
        ["setor", "subsetor", "posicao_lider", "posicao_seguidora"]
    ).reset_index(drop=True)


# Colunas que viajam junto com a ponta seguidora no embaralhamento.
_LADO_SEGUIDORA = ("seguidora", "emissor_seguidora", "posicao_seguidora",
                   "liquidez_seguidora")

_MAX_TENTATIVAS_EMBARALHAMENTO = 50


def pares_placebo_embaralhados(
    pares: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Placebo: sorteia de novo quem e seguidora de quem, dentro do grupo.

    O contraste verifica se a mediana observada se destaca quando a atribuicao
    de seguidoras e embaralhada dentro do grupo.

    O sorteio preserva o grupo (setor, subsetor), a proibicao de lider ==
    seguidora e a de mesmo emissor. Nao preserva a ordem de liquidez, de
    proposito: o placebo existe justamente para quebrar o alinhamento lider ->
    seguidora.

    Grupo em que nenhuma permutacao valida saiu (normalmente grupo de um par
    so) fica com a atribuicao original - limitacao conhecida em grupos
    pequenos.

    `faixa_minima` perde o sentido depois do embaralhamento, entao filtre a
    faixa antes.
    """
    saida = pares.copy()
    for _, grupo in pares.groupby(["setor", "subsetor"], sort=False):
        if len(grupo) < 2:
            continue
        idx = grupo.index.to_numpy()
        for _ in range(_MAX_TENTATIVAS_EMBARALHAMENTO):
            perm = rng.permutation(len(idx))
            candidato = grupo[list(_LADO_SEGUIDORA)].iloc[perm].reset_index(drop=True)
            valido = (
                (candidato["seguidora"].to_numpy() != grupo["lider"].to_numpy())
                & (candidato["emissor_seguidora"].to_numpy()
                   != grupo["emissor_lider"].to_numpy()))
            if valido.all():
                saida.loc[idx, list(_LADO_SEGUIDORA)] = candidato.to_numpy()
                break
    return saida
