"""
Selecao point-in-time do universo operavel.

Este e um dos pontos em que lookahead pode entrar (o outro e escolher pares por
desempenho). "As 40 mais liquidas" calculado sobre a amostra inteira usa o
futuro para decidir o que operar no passado. Por isso o ranking e reconstruido
dentro de cada janela de treino e congelado durante o teste seguinte. As
funcoes carregam a data de corte na assinatura para permitir essa verificacao
nos testes automatizados.

Por janela: cobertura minima, pregoes validos minimos, dias sem negociacao,
volume financeiro mediano, top N por liquidez e, por fim, uma classe de acao
por emissor, a mais liquida. O ultimo passo e o que impede par PETR3 x PETR4 ou
acao x unit do mesmo emissor. Sem ele, a tabela poderia destacar duas classes
da mesma empresa, com correlacao quase perfeita e defasagem mecanica da classe
menos liquida, exatamente o artefato que o projeto procura separar.

O ranking dinamico elimina o lookahead da selecao por liquidez, mas nao resolve
sozinho toda a historia societaria. O COTAHIST inclui papeis deslistados; a
classificacao setorial temporal e o mapeamento de tickers tratam separadamente
as mudancas de identidade e de grupo economico.
"""

from __future__ import annotations

import pandas as pd

from src.backtest import Janela
from src.config import LiquidezConfig
from src.setores import emissor_do_isin


def ranking_liquidez_pit(
    painel: pd.DataFrame,
    janela: Janela,
    cfg: LiquidezConfig | None = None,
) -> pd.DataFrame:
    """
    Ranking de liquidez point-in-time de uma janela de treino.

    So os pregoes entre `janela.treino_inicio` (inclusive) e `janela.treino_fim`
    (exclusivo). Nenhum dado posterior entra, nem para calcular nem para
    filtrar.

    Liquidez e a mediana do volume financeiro diario, nao a media: um unico dia
    de leilao gigante nao deve promover papel iliquido para o topo.

    Cada codigo e rankeado com os proprios dados da janela. Volume de tickers
    distintos nunca e somado aqui, mesmo quando se sabe que sao a mesma
    empresa: corrigir identidade de ativo e retrospectivo e legitimo, mas
    deixar a liquidez futura do codigo novo ajudar a selecionar o antigo seria
    vazamento.
    """
    cfg = cfg or LiquidezConfig()
    treino = painel[
        (painel["DATA"] >= janela.treino_inicio) & (painel["DATA"] < janela.treino_fim)
    ]
    if treino.empty:
        return pd.DataFrame()

    pregoes_janela = treino["DATA"].nunique()
    agg = "median" if cfg.estatistica_liquidez == "mediana" else "mean"

    r = treino.groupby("CODNEG").agg(
        liquidez=("VOLTOT", agg),
        totneg_mediano=("TOTNEG", "median"),
        pregoes=("DATA", "nunique"),
        dias_volume_zero=("VOLTOT", lambda s: int((s <= 0).sum())),
        emissor_id=("CODISI", lambda s: emissor_do_isin(s.iloc[0])),
        tipo_papel=("TIPO_PAPEL", "first"),
        nome=("NOMRES", "first"),
    )
    r["cobertura"] = r["pregoes"] / pregoes_janela
    r["prop_dias_negociados"] = (r["pregoes"] - r["dias_volume_zero"]) / pregoes_janela

    # A mediana de 24 meses rankeia bem um papel que morreu semanas antes do fim
    # do treino: VALE5, ja convertida em VALE3, chegou a 2o lugar numa janela em
    # que nao negociava mais. Quem decide e o pregao de formacao - sem negocio
    # nele, o papel nao e operavel no teste.
    ultima_com_negocio = (treino[treino["VOLTOT"] > 0]
                          .groupby("CODNEG")["DATA"].max())
    r["negociou_no_pregao_final"] = (
        ultima_com_negocio.reindex(r.index) == treino["DATA"].max())

    # Os limiares do config valem dentro da janela: foram calibrados para 24
    # meses, nao para a amostra inteira.
    r["elegivel"] = (
        (r["cobertura"] >= cfg.cobertura_minima)
        & (r["prop_dias_negociados"] >= cfg.proporcao_minima_dias_negociados)
        & (r["pregoes"] >= min(cfg.min_pregoes_treino, pregoes_janela))
    )

    # Desempate deterministico: liquidez igual, vence a ordem alfabetica. O sort
    # padrao do pandas nao e estavel, e posicao que depende do algoritmo de
    # ordenacao nao e reproduzivel.
    r = r.sort_index().sort_values("liquidez", ascending=False, kind="mergesort")
    # Posicao so entre os elegiveis: papel com serie furada nao deve empurrar os
    # outros para baixo.
    r["posicao"] = r["elegivel"].cumsum().where(r["elegivel"])
    r["janela"] = janela.rotulo
    r["pregoes_na_janela"] = pregoes_janela
    return r


# Motivos de exclusao da selecao final. Um value_counts sobre eles da o funil
# completo, que e como a selecao presta contas do que deixou de fora.
MOTIVO_NAO_ELEGIVEL = "nao_elegivel"
MOTIVO_SEM_NEGOCIACAO_NA_FORMACAO = "sem_negociacao_no_pregao_de_formacao"
MOTIVO_ISIN_INVALIDO = "isin_invalido"
MOTIVO_CLASSE_MENOS_LIQUIDA = "classe_menos_liquida_do_emissor"


def _motivos_base(r: pd.DataFrame) -> pd.DataFrame:
    """Exclusoes que nao dependem de emissor: qualidade e existencia."""
    r["motivo_exclusao"] = ""
    r.loc[~r["elegivel"], "motivo_exclusao"] = MOTIVO_NAO_ELEGIVEL
    vivo = r["motivo_exclusao"] == ""
    r.loc[vivo & ~r["negociou_no_pregao_final"],
          "motivo_exclusao"] = MOTIVO_SEM_NEGOCIACAO_NA_FORMACAO
    return r


def um_por_emissor(ranking: pd.DataFrame) -> pd.DataFrame:
    """
    Uma classe por emissor: sobrevive a mais liquida entre as elegiveis.

    PETR3 e PETR4 sao a mesma empresa, e um par entre elas seria preco velho na
    forma mais pura.

    Duas decisoes que nao sao obvias:

    - a disputa e so entre elegiveis e vivos na formacao. Classe reprovada na
      cobertura, ou que ja parou de negociar, nao elimina a irma viva. Foi o
      caso VALE5/VALE3 em 2018: a extinta ainda rankeava pela mediana e, sem
      esta ordem, teria eliminado a viva na deduplicacao;
    - ticker sem ISIN valido fica fora. Sem emissor identificavel nao da para
      garantir a regra, e incluir seria apostar que nao existe classe irma.

    Devolve o ranking com `motivo_exclusao` (vazio = selecionado) e
    `posicao_final`, recontada so entre os selecionados.
    """
    r = _motivos_base(ranking.copy())

    sem_emissor = (r["motivo_exclusao"] == "") & r["emissor_id"].isna()
    r.loc[sem_emissor, "motivo_exclusao"] = MOTIVO_ISIN_INVALIDO

    # O ranking ja vem ordenado por liquidez com desempate deterministico,
    # entao "manter a primeira ocorrencia do emissor" e manter a mais liquida.
    disputa = r["motivo_exclusao"] == ""
    repetida = disputa & r.loc[disputa, "emissor_id"].duplicated().reindex(
        r.index, fill_value=False)
    r.loc[repetida, "motivo_exclusao"] = MOTIVO_CLASSE_MENOS_LIQUIDA

    selecionada = r["motivo_exclusao"] == ""
    r["posicao_final"] = selecionada.cumsum().where(selecionada)
    return r


def selecionar_universo(
    painel: pd.DataFrame,
    janela: Janela,
    cfg: LiquidezConfig | None = None,
) -> pd.DataFrame:
    """
    Selecao final do universo operavel de uma janela de treino.

    ranking point-in-time -> elegibilidade -> uma classe por emissor -> faixas.
    A deduplicacao vem antes do corte das faixas, entao "top 40" sao os 40
    emissores mais liquidos, cada um pela melhor classe. Cortar primeiro e
    deduplicar depois encolheria o universo a cada empresa com duas classes no
    topo, e o tamanho da faixa deixaria de ser comparavel entre janelas.

    Devolve todos os tickers da janela, com `motivo_exclusao` nos que ficaram
    fora e `faixa` nos selecionados. Quem consome uma faixa filtra por
    `posicao_final <= n`.
    """
    cfg = cfg or LiquidezConfig()
    r = ranking_liquidez_pit(painel, janela, cfg)
    if r.empty:
        return r

    if cfg.um_ticker_por_emissor:
        r = um_por_emissor(r)
    else:
        r = _motivos_base(r.copy())
        selecionada = r["motivo_exclusao"] == ""
        r["posicao_final"] = selecionada.cumsum().where(selecionada)

    r["faixa"] = ""
    com_posicao = r["posicao_final"].notna()
    r.loc[com_posicao, "faixa"] = r.loc[com_posicao, "posicao_final"].map(
        lambda p: _faixa(p, cfg.faixas))
    return r


def melhor_posicao_por_ticker(
    painel: pd.DataFrame,
    janelas: list[Janela],
    cfg: LiquidezConfig | None = None,
) -> pd.DataFrame:
    """
    Para cada ticker, a melhor posicao de liquidez alcancada em alguma janela.

    Responde "este papel chegou a importar em algum momento?" sem recorrer a
    liquidez media dos 12 anos, que mistura epocas e favorece quem listou
    recentemente num periodo de volume alto.
    """
    registros = []
    for j in janelas:
        r = ranking_liquidez_pit(painel, j, cfg)
        if r.empty:
            continue
        registros.append(
            r.loc[r["elegivel"], ["posicao", "liquidez", "pregoes"]]
            .assign(janela=j.rotulo)
        )
    if not registros:
        return pd.DataFrame()

    # Ordenado por posicao, a melhor janela de cada ticker sai de um
    # drop_duplicates, sem lambda de agregacao.
    todos = pd.concat(registros).reset_index().sort_values("posicao")

    melhor = todos.groupby("CODNEG").agg(
        melhor_posicao=("posicao", "min"),
        janelas_elegivel=("posicao", "size"),
        liquidez_max=("liquidez", "max"),
    )
    melhor["janela_melhor"] = (
        todos.drop_duplicates("CODNEG", keep="first")
        .set_index("CODNEG")["janela"]
    )
    return melhor.sort_values("melhor_posicao")


def alcance_pit(
    ocorrencias: pd.DataFrame,
    painel: pd.DataFrame,
    janelas: list[Janela],
    cfg: LiquidezConfig | None = None,
) -> pd.DataFrame:
    """
    Onde cada par (ticker, data) cai em relacao aos universos point-in-time.

    Serve para priorizar revisao documental: retorno absurdo num papel que nunca
    chegou perto de uma faixa nao afeta o estudo, o mesmo retorno na 8a posicao
    do top20 afeta tudo. Sem este cruzamento, a fila de revisao sairia ordenada
    por tamanho do retorno, que e o criterio errado.

    Sao tres posicoes diferentes, e juntar as tres estragaria a fila. A melhor
    posicao em qualquer janela e so descritiva: responde "este papel chegou a
    importar?", nao "este evento importou?". GOLL4 ja esteve em 25o lugar, mas
    no treino que contem 2016-02-01 estava em 58o, entao o evento alcanca
    top60. Usar a melhor global promoveria evento antigo por liquidez que so
    veio depois.

    Quem decide efeito e a posicao na janela que contem a data:

      pode_afetar_estimacao  a data cai no treino de uma janela em que o papel
                             esta numa faixa, contaminando a rede estimada.
      pode_afetar_sinal      a data cai no teste, contaminando sinal e P&L com o
                             universo ja congelado pelo treino anterior.

    As duas bandeiras sao limite superior: nao passaram por deduplicacao de
    emissor, filtro setorial nem formacao de pares, e os tres so reduzem o
    conjunto.
    """
    cfg = cfg or LiquidezConfig()
    faixa_max = max(cfg.faixas)

    rankings = {j.rotulo: ranking_liquidez_pit(painel, j, cfg) for j in janelas}

    def _status_e_posicao(rotulos, ticker) -> tuple[str, float]:
        """
        Estado do ticker nas janelas que contem a data, e a melhor posicao.

        A prioridade importa quando a data cai em varias janelas: entrou em
        faixa em alguma, e isso que vale; depois elegivel sem faixa; so entao
        nao-elegivel.
        """
        if not rotulos:
            return STATUS_SEM_JANELA, float("nan")

        melhor, elegivel_em_alguma, presente_em_alguma = float("nan"), False, False
        for rot in rotulos:
            r = rankings.get(rot)
            if r is None or r.empty or ticker not in r.index:
                continue
            presente_em_alguma = True
            if not bool(r.at[ticker, "elegivel"]):
                continue
            elegivel_em_alguma = True
            p = float(r.at[ticker, "posicao"])
            if p == p and (melhor != melhor or p < melhor):
                melhor = p

        if melhor == melhor and melhor <= faixa_max:
            return _faixa(melhor, cfg.faixas), melhor
        if elegivel_em_alguma:
            return STATUS_ELEGIVEL_FORA, melhor
        if presente_em_alguma:
            return STATUS_NAO_ELEGIVEL, float("nan")
        return STATUS_AUSENTE, float("nan")

    linhas = []
    for _, oc in ocorrencias.iterrows():
        tk = oc["ticker"]
        data = pd.Timestamp(oc["data"])

        rot_treino = [j.rotulo for j in janelas
                      if j.treino_inicio <= data < j.treino_fim]
        rot_teste = [j.rotulo for j in janelas
                     if j.teste_inicio <= data < j.teste_fim]

        st_treino, p_treino = _status_e_posicao(rot_treino, tk)
        st_teste, p_teste = _status_e_posicao(rot_teste, tk)
        st_global, p_global = _status_e_posicao(list(rankings), tk)

        linhas.append({
            "ticker": tk,
            "data": data.date(),
            # descritivo: nao ordena fila nem decide efeito
            "melhor_posicao_qualquer_janela": p_global,
            "faixa_melhor_qualquer_janela": st_global,
            # e isto que decide
            "status_treino_na_data": st_treino,
            "melhor_posicao_treino_na_data": p_treino,
            "status_teste_na_data": st_teste,
            "melhor_posicao_teste_na_data": p_teste,
            "janelas_treino_contendo_a_data": len(rot_treino),
            "janelas_teste_contendo_a_data": len(rot_teste),
            "pode_afetar_estimacao": p_treino == p_treino and p_treino <= faixa_max,
            "pode_afetar_sinal": p_teste == p_teste and p_teste <= faixa_max,
        })
    return pd.DataFrame(linhas)


# Estados de um (ticker, data) em relacao aos universos, exclusivos entre si (o
# alcance cumulativo por faixa e outra leitura). Um "fora do universo" so
# juntaria quatro situacoes diferentes: nao existir na janela nao e o mesmo que
# existir e nao qualificar.
STATUS_SEM_JANELA = "sem_janela_na_data"
STATUS_AUSENTE = "ausente_da_janela"
STATUS_NAO_ELEGIVEL = "nao_elegivel_na_janela"
STATUS_ELEGIVEL_FORA = "elegivel_fora_top100"


def _faixa(posicao: float, faixas: tuple[int, ...]) -> str:
    """Menor faixa que a posicao alcanca (58 -> top60, nao top100)."""
    if posicao != posicao:
        return STATUS_ELEGIVEL_FORA
    for f in sorted(faixas):
        if posicao <= f:
            return f"top{f}"
    return STATUS_ELEGIVEL_FORA


def alcance_cumulativo_por_faixa(
    alcance: pd.DataFrame, coluna_posicao: str, faixas: tuple[int, ...]
) -> pd.Series:
    """
    Quantas ocorrencias alcancam cada faixa, de forma cumulativa.

    top40 contem top20, top60 contem os dois, e assim por diante. Aqui um caso
    na 58a posicao conta em top60 e em top100; no estado exclusivo ele e so
    top60.
    """
    pos = alcance[coluna_posicao]
    return pd.Series({f"top{f}": int((pos <= f).sum()) for f in sorted(faixas)})
