"""
Auditoria dos dados: nenhum ticker entra no backtest sem passar por aqui.

A ordem e marcar ausencia, calcular as flags sobre o observado e so entao
decidir tratamento. Alinhar ao calendario e preencher antes das flags daria, a
um papel que ficou cinco pregoes sem negociar, cinco fechamentos identicos e
cinco retornos zero fabricados pelo proprio pre-processamento - o detector de
preco velho passaria a detectar a si mesmo. Por isso toda flag sai da sequencia
de registros efetivamente observados, nunca de serie preenchida.

Sao tres situacoes diferentes, e confundir as duas primeiras destruiria a
metrica de cobertura (empresa que abriu capital em 2020 apareceria com 5 anos
de ausencia):

    nao_listado      a data esta fora da janela de vida do papel; ele nao
                     existia ou ja tinha saido. Nao e dado faltante.
    sem_negociacao   o papel existia, houve pregao e ele nao negociou. E aqui
                     que mora o preco velho.
    negociado        ha registro no arquivo.

Sobre o TOTNEG: "mediana de um negocio por dia" e evidencia fortissima de
iliquidez, mas nao prova fechamento obsoleto - aquele negocio unico pode ter
saido perto do fim do pregao, com preco atual. O COTAHIST diario e um resumo e
nao traz o horario do ultimo negocio, entao obsolescencia e sempre inferencia.
Dai o diagnostico ser por convergencia de sinais, nunca por indicador isolado.

Aqui so medimos, marcamos e reportamos. Nada e corrigido, preenchido ou
descartado: a exclusao e decisao de universo.py, com criterio no config.
"""

from __future__ import annotations

import pandas as pd

from src.config import LiquidezConfig, NaoSincroniaConfig


def calendario_pregoes(painel: pd.DataFrame, min_tickers: int = 20) -> pd.DatetimeIndex:
    """
    Dias em que houve pregao na B3, derivados dos proprios dados.

    Uma data e pregao se um numero minimo de papeis negociou nela. Nao usamos
    lista de feriados nacional porque o calendario da B3 tem particularidades
    (feriados estaduais de Sao Paulo, quarta-feira de cinzas ate meio-dia,
    pregoes especiais) que uma lista generica erra; o proprio arquivo da bolsa e
    a fonte mais confiavel de quando ela abriu.

    O piso de `min_tickers` evita que uma unica linha com data corrompida vire
    um pregao inteiro.
    """
    por_data = painel.groupby("DATA")["CODNEG"].nunique()
    return pd.DatetimeIndex(sorted(por_data[por_data >= min_tickers].index))


def janela_listagem(painel: pd.DataFrame) -> pd.DataFrame:
    """
    Primeira e ultima data em que cada papel aparece no arquivo.

    Delimita a janela de vida do instrumento. Fora dela, ausencia significa
    "nao existia", nao "nao negociou".
    """
    g = painel.groupby("CODNEG")["DATA"]
    return pd.DataFrame({"primeira_data": g.min(), "ultima_data": g.max()})


def marcar_dias_suspeitos(
    painel: pd.DataFrame, cfg: NaoSincroniaConfig | None = None
) -> pd.DataFrame:
    """
    Flags por (data, ticker), calculadas sobre os registros observados.

    `fechamento_repetido` e `retorno_zero` comparam registros consecutivos no
    arquivo, e nao dias consecutivos do calendario: se o papel pulou tres
    pregoes, a comparacao e entre o negocio de antes e o de depois do buraco.

    Flags produzidas:
        volume_zero          VOLTOT == 0 com registro existindo
        poucos_negocios      TOTNEG abaixo do minimo configurado
        volume_baixo         VOLTOT no decil inferior do proprio papel
        fechamento_repetido  PREULT igual ao do registro anterior
        retorno_zero         variacao zero entre registros consecutivos
        suspeito_preco_velho convergencia de dois ou mais sinais acima
    """
    cfg = cfg or NaoSincroniaConfig()
    df = painel.sort_values(["CODNEG", "DATA"]).copy()
    g = df.groupby("CODNEG", sort=False)

    df["volume_zero"] = df["VOLTOT"].fillna(0) <= 0
    df["poucos_negocios"] = df["TOTNEG"].fillna(0) < cfg.min_negocios_dia

    # Limiar relativo ao proprio papel: papel pequeno nao deve ser marcado por
    # ser pequeno, e sim por estar abaixo do padrao dele mesmo.
    limiar = g["VOLTOT"].transform(lambda s: s.quantile(cfg.percentil_volume_baixo))
    df["volume_baixo"] = df["VOLTOT"] < limiar

    anterior = g["PREULT"].shift(1)
    df["fechamento_repetido"] = (df["PREULT"] == anterior) & anterior.notna()
    df["retorno_zero"] = df["fechamento_repetido"]  # mesma coisa, outro nome

    # Nenhum sinal isolado basta (ver docstring do modulo).
    sinais = df[["volume_zero", "poucos_negocios", "volume_baixo",
                 "fechamento_repetido"]].sum(axis=1)
    df["n_sinais"] = sinais
    df["suspeito_preco_velho"] = sinais >= 2
    return df


def sequencias_fechamento_repetido(painel: pd.DataFrame) -> pd.Series:
    """
    Maior sequencia de fechamentos identicos consecutivos, por ticker.

    Preco travado por varios registros seguidos e um dos sinais mais claros de
    papel que praticamente nao negocia.
    """
    df = painel.sort_values(["CODNEG", "DATA"])
    mudou = df.groupby("CODNEG", sort=False)["PREULT"].transform(
        lambda s: (s != s.shift(1)).cumsum()
    )
    tamanhos = df.groupby(["CODNEG", mudou]).size()
    return tamanhos.groupby(level="CODNEG").max().rename("maior_sequencia_preco_igual")


def auditar_tickers(
    painel: pd.DataFrame,
    calendario: pd.DatetimeIndex | None = None,
    cfg_ns: NaoSincroniaConfig | None = None,
    cfg_liq: LiquidezConfig | None = None,
) -> pd.DataFrame:
    """
    Uma linha por ticker, com tudo que decide se ele e confiavel.

    A cobertura e medida dentro da janela de listagem do papel, e nao contra o
    periodo inteiro: caso contrario toda empresa que abriu capital no meio da
    amostra apareceria com dados faltando.
    """
    cfg_ns = cfg_ns or NaoSincroniaConfig()
    cfg_liq = cfg_liq or LiquidezConfig()
    calendario = calendario if calendario is not None else calendario_pregoes(painel)
    cal = pd.Series(1, index=pd.DatetimeIndex(calendario)).sort_index()

    flags = marcar_dias_suspeitos(painel, cfg_ns)
    janelas = janela_listagem(painel)

    pos_ini = cal.index.searchsorted(janelas["primeira_data"].values, side="left")
    pos_fim = cal.index.searchsorted(janelas["ultima_data"].values, side="right")
    janelas["pregoes_esperados"] = pos_fim - pos_ini

    g = flags.groupby("CODNEG")
    aud = pd.DataFrame({
        "primeira_data": janelas["primeira_data"],
        "ultima_data": janelas["ultima_data"],
        "pregoes_esperados": janelas["pregoes_esperados"],
        "pregoes_observados": g.size(),
        "tipo_papel": g["TIPO_PAPEL"].first(),
        "emissor_isin": g["CODISI"].first().str.slice(2, 6),
        "nome": g["NOMRES"].first(),
        "voltot_mediano": g["VOLTOT"].median(),
        "voltot_medio": g["VOLTOT"].mean(),
        "totneg_mediano": g["TOTNEG"].median(),
        "totneg_p10": g["TOTNEG"].quantile(0.10),
        "preco_mediano": g["PREULT"].median(),
        "dias_volume_zero": g["volume_zero"].sum(),
        "dias_poucos_negocios": g["poucos_negocios"].sum(),
        "dias_fechamento_repetido": g["fechamento_repetido"].sum(),
        "dias_retorno_zero": g["retorno_zero"].sum(),
        "dias_suspeitos": g["suspeito_preco_velho"].sum(),
    })
    aud["maior_sequencia_preco_igual"] = sequencias_fechamento_repetido(painel)

    # Pregoes em que o papel existia e nao negociou: ausencia so conta dentro da
    # janela de listagem.
    aud["pregoes_sem_negociacao"] = (
        aud["pregoes_esperados"] - aud["pregoes_observados"]
    ).clip(lower=0)

    aud["cobertura"] = (aud["pregoes_observados"] / aud["pregoes_esperados"]).where(
        aud["pregoes_esperados"] > 0
    )
    aud["prop_dias_negociados"] = (
        (aud["pregoes_observados"] - aud["dias_volume_zero"])
        / aud["pregoes_esperados"]
    ).where(aud["pregoes_esperados"] > 0)

    aud["frac_dias_suspeitos"] = aud["dias_suspeitos"] / aud["pregoes_observados"]
    aud["frac_fechamento_repetido"] = (
        aud["dias_fechamento_repetido"] / aud["pregoes_observados"]
    )

    # Elegibilidade so marca; quem exclui e universo.py, janela a janela.
    #
    # Rodando sobre o periodo inteiro, isto aqui e descritivo: os limiares de
    # cobertura e de dias negociados foram calibrados para 24 meses, e exigi-los
    # sobre 12 anos reprovaria qualquer empresa que abriu ou fechou capital no
    # meio da amostra - o que nao e defeito de dado.
    aud["elegivel_cobertura"] = aud["cobertura"] >= cfg_liq.cobertura_minima
    aud["elegivel_negociacao"] = (
        aud["prop_dias_negociados"] >= cfg_liq.proporcao_minima_dias_negociados
    )

    # A taxa, em vez do máximo absoluto, separa séries quebradas de repetições
    # ocasionais em papéis líquidos.
    aud["elegivel_serie_integra"] = (
        aud["frac_fechamento_repetido"] <= cfg_ns.max_frac_fechamento_repetido
    )

    aud["elegivel"] = (
        aud["elegivel_cobertura"] & aud["elegivel_negociacao"]
        & aud["elegivel_serie_integra"]
    )

    return aud.sort_values("voltot_mediano", ascending=False)


# Versao do detector de mudanca de ticker, congelada: os casos relevantes ja
# apareceram e a proxima melhoria vem de confirmacao documental, nao de mais
# heuristica. Mexer na deteccao ou nos diagnosticos exige subir esta versao, o
# que marca needs_revalidation nas linhas ja revisadas - decisao tomada olhando
# diagnostico antigo pode nao valer para o novo.
DETECTOR_VERSAO = "1.0.0"


def candidatos_mudanca_ticker(
    painel: pd.DataFrame, folga_pregoes: int = 15, janela_liquidez: int = 60
) -> pd.DataFrame:
    """
    Pares de tickers que parecem ser a mesma empresa antes e depois de uma troca
    de codigo.

    Caso real da amostra: a Embraer negocia como EMBR3 ate 31/10/2025 e como
    EMBJ3 a partir de 03/11/2025, com ISIN novo (BREMBRACNOR4 -> BREMBJACNOR1),
    e a Eletrobras vira AXIA ENERGIA (AXIA3) na mesma epoca. Para o pipeline sao
    empresas diferentes: a serie quebra em dois pedacos curtos, e os dois podem
    reprovar na cobertura. O codigo de emissor do ISIN, que resolve PETR3 x
    PETR4, nao ajuda aqui - EMBR e EMBJ sao emissores distintos para ele.

    O criterio e conservador de proposito: mesmo NOMRES, tickers diferentes,
    janelas de negociacao praticamente sem sobreposicao e o novo comecando logo
    depois de o antigo terminar.

    Devolve candidatos para conferencia humana. Nao emenda nada sozinho:
    detectar evento corporativo so por preco e nome e fragil, e um encadeamento
    errado inventaria uma serie que nunca existiu.
    """
    janelas = janela_listagem(painel)
    nomes = painel.groupby("CODNEG")["NOMRES"].agg(
        lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""
    )
    info = janelas.join(nomes)

    linhas = []

    # Razão de preço distante de 1 com o mesmo nome sugere desdobramento ou
    # conversão de classe, não uma simples troca de código.
    ordenado_por_data = painel.sort_values("DATA")
    ultimo_preco = ordenado_por_data.groupby("CODNEG")["PREULT"].last()
    primeiro_preco = ordenado_por_data.groupby("CODNEG")["PREULT"].first()

    def _precos(tk_ant: str, tk_nov: str) -> dict:
        p_a = float(ultimo_preco.get(tk_ant, float("nan")))
        p_n = float(primeiro_preco.get(tk_nov, float("nan")))
        salto = abs(p_n / p_a - 1) if p_a and p_a == p_a and p_n == p_n else float("nan")
        return {"preco_antigo": round(p_a, 2) if p_a == p_a else None,
                "preco_novo": round(p_n, 2) if p_n == p_n else None,
                "salto_preco": round(salto, 4) if salto == salto else None}

    vistos = set()
    for nome, grupo in info[info["NOMRES"] != ""].groupby("NOMRES"):
        if len(grupo) < 2:
            continue
        g = grupo.sort_values("primeira_data")
        for i in range(len(g) - 1):
            ant, nov = g.iloc[i], g.iloc[i + 1]
            if nov["primeira_data"] <= ant["ultima_data"]:
                continue
            intervalo = (nov["primeira_data"] - ant["ultima_data"]).days
            if intervalo > folga_pregoes * 3:
                continue
            vistos.add((g.index[i], g.index[i + 1]))
            linhas.append({
                "empresa_antiga": nome, "empresa_nova": nome,
                "ticker_antigo": g.index[i], "fim_antigo": ant["ultima_data"].date(),
                "ticker_novo": g.index[i + 1], "inicio_novo": nov["primeira_data"].date(),
                "dias_de_intervalo": intervalo, "confianca": "alta",
                "motivo": "mesmo nome resumido",
                **_precos(g.index[i], g.index[i + 1]),
            })

    # Rebrandings exigem continuidade de preço e liquidez próxima à transição.
    # Adjacência temporal ou liquidez histórica isoladas geram falsos pares.
    ultimos = (painel.sort_values("DATA").groupby("CODNEG")
               .tail(janela_liquidez).groupby("CODNEG")["VOLTOT"].median())
    primeiros = (painel.sort_values("DATA").groupby("CODNEG")
                 .head(janela_liquidez).groupby("CODNEG")["VOLTOT"].median())

    for tk_ant, ant in info.sort_values("ultima_data").iterrows():
        adjacentes = info[
            (info["primeira_data"] > ant["ultima_data"])
            & ((info["primeira_data"] - ant["ultima_data"]).dt.days <= folga_pregoes)
            & (info.index != tk_ant)
        ]
        for tk_nov, nov in adjacentes.iterrows():
            if (tk_ant, tk_nov) in vistos:
                continue
            a, b = ultimos.get(tk_ant, 0), primeiros.get(tk_nov, 0)
            if a <= 0 or b <= 0 or not (1 / 3 <= b / a <= 3.0):
                continue
            p_ant, p_nov = ultimo_preco.get(tk_ant, 0), primeiro_preco.get(tk_nov, 0)
            if p_ant <= 0 or p_nov <= 0:
                continue
            salto = abs(p_nov / p_ant - 1)
            if salto > 0.15:
                continue
            linhas.append({
                "empresa_antiga": ant["NOMRES"], "empresa_nova": nov["NOMRES"],
                "ticker_antigo": tk_ant, "fim_antigo": ant["ultima_data"].date(),
                "ticker_novo": tk_nov, "inicio_novo": nov["primeira_data"].date(),
                "dias_de_intervalo": (nov["primeira_data"] - ant["ultima_data"]).days,
                **_precos(tk_ant, tk_nov),
                "confianca": "media",
                "motivo": "series adjacentes com preco continuo, nomes diferentes",
            })

    if not linhas:
        return pd.DataFrame()
    return (pd.DataFrame(linhas)
            .sort_values(["confianca", "fim_antigo"])
            .reset_index(drop=True))


# Versao da camada de validacao da serie do Yahoo contra o registro oficial.
# Independente de DETECTOR_VERSAO: uma muda a deteccao de candidatos, a outra a
# conferencia numerica.
VALIDACAO_VERSAO = "1.0.0"


def metricas_validacao_yf(
    painel: pd.DataFrame,
    ticker_antigo: str,
    ticker_novo: str,
    yf_df: pd.DataFrame,
    meta_yf: dict | None = None,
    folga_transicao: int = 3,
) -> dict:
    """
    Compara a serie que o Yahoo entrega sob o ticker novo com o registro oficial
    da B3 para o ticker antigo.

    O yfinance emenda tickers renomeados por conta propria: COGN3.SA devolve
    historico desde 2014, embora COGN3 so exista no COTAHIST desde out/2019. A
    emenda e conveniente e opaca, mas auditavel - no periodo em que a empresa
    negociava sob o codigo antigo, o fechamento bruto reportado sob o codigo
    novo deveria reproduzir o PREULT do antigo.

    Dois testes, separados de proposito:

      preco bruto  Close nao ajustado do Yahoo contra PREULT do COTAHIST. Razao
                   constante diferente de 1 e reajuste retroativo por split,
                   bonificacao ou conversao: a emenda esta certa, so reescalada.
                   Razao erratica indica series sem relacao. Por isso o que
                   importa e a dispersao da razao, nao a distancia dela para 1.

      retornos     e o que o projeto usa, e onde fator constante some. Exclui
                   transicao, evento corporativo e dias sem negociacao, pontos
                   em que a divergencia e esperada e nao diz nada sobre a
                   identidade da serie.

    Devolve so metricas. A classificacao vem depois, com limiares calibrados em
    casos conhecidos, e nunca autoriza continuidade sozinha.
    """
    m: dict = {
        "yf_ticker_consultado": ticker_novo,
        "validation_version": VALIDACAO_VERSAO,
        "yf_primeira_data": None, "yf_ultima_data": None,
        "n_datas_sobrepostas": 0, "n_datas_eventos_excluidas": 0,
        "corr_retornos": None, "erro_retorno_mediano": None,
        "erro_retorno_p95": None, "razao_precos_mediana": None,
        "dispersao_razao_precos": None,
    }

    status_meta = (meta_yf or {}).get("status")
    if status_meta == "erro_download":
        m["_situacao"] = "erro_download"
        return m
    if yf_df is None or yf_df.empty or status_meta == "serie_ausente":
        m["_situacao"] = "serie_yahoo_ausente"
        return m

    yf = yf_df.copy()
    yf["Date"] = pd.to_datetime(yf["Date"])
    yf = yf.set_index("Date").sort_index()
    m["yf_primeira_data"] = yf.index.min().date()
    m["yf_ultima_data"] = yf.index.max().date()

    cot = painel[painel["CODNEG"] == ticker_antigo].set_index("DATA").sort_index()
    if cot.empty:
        m["_situacao"] = "sem_cotahist"
        return m

    ini, fim = cot.index.min(), cot.index.max()
    j = pd.DataFrame({
        "yf_close": yf.loc[(yf.index >= ini) & (yf.index <= fim), "Close"],
        "cot_preult": cot["PREULT"],
        "cot_voltot": cot["VOLTOT"],
        "cot_totneg": cot["TOTNEG"],
    })
    for col in ("Dividends", "Stock Splits"):
        j[col] = yf[col] if col in yf.columns else 0.0
    j = j.dropna(subset=["yf_close", "cot_preult"]).sort_index()
    m["n_datas_sobrepostas"] = int(len(j))
    if len(j) < 30:
        m["_situacao"] = "inconclusivo_pouca_amostra"
        return m

    razao = j["yf_close"] / j["cot_preult"]
    mediana = float(razao.median())
    m["razao_precos_mediana"] = round(mediana, 6)
    m["dispersao_razao_precos"] = round(float(razao.std() / mediana), 6) if mediana else None

    r_yf = j["yf_close"].pct_change()
    r_cot = j["cot_preult"].pct_change()

    evento = (j["Dividends"].fillna(0) != 0) | (j["Stock Splits"].fillna(0) != 0)
    # O retorno do dia seguinte a um evento tambem e contaminado.
    evento = evento | evento.shift(1, fill_value=False)
    sem_negocio = (j["cot_voltot"].fillna(0) <= 0) | (j["cot_totneg"].fillna(0) <= 0)
    sem_negocio = sem_negocio | sem_negocio.shift(1, fill_value=False)
    transicao = pd.Series(False, index=j.index)
    transicao.iloc[:folga_transicao] = True
    transicao.iloc[-folga_transicao:] = True

    excluir = evento | sem_negocio | transicao
    m["n_datas_eventos_excluidas"] = int(excluir.sum())

    ok = r_yf.notna() & r_cot.notna() & ~excluir
    if int(ok.sum()) < 30:
        m["_situacao"] = "inconclusivo_pouca_amostra"
        return m

    a, b = r_yf[ok], r_cot[ok]
    dif = (a - b).abs()
    m["corr_retornos"] = round(float(a.corr(b)), 6)
    m["erro_retorno_mediano"] = round(float(dif.median()), 8)
    m["erro_retorno_p95"] = round(float(dif.quantile(0.95)), 8)
    m["_situacao"] = "calculado"
    return m


# Os controles foram classificados por evidência societária antes da calibração
# numérica; detalhes em data/reference/calibracao_validacao_yf.md.
CALIBRACAO_VERSAO = "1.0.0"   # ver data/reference/calibracao_validacao_yf.md

LIMIAR_ERRO_RETORNO_MEDIANO = 1e-05
LIMIAR_ERRO_RETORNO_P95 = 0.02
LIMIAR_CORR_RETORNOS = 0.95
# Acima disto a razao de precos tem degrau, o que nao invalida a serie mas
# indica evento societario no meio do periodo. Fixado em ~3x a dispersao maxima
# dos controles positivos (0,0151).
LIMIAR_DISPERSAO_DEGRAU = 0.05

CONTROLES_POSITIVOS = (
    ("ESTC3", "YDUQ3"), ("EMBR3", "EMBJ3"), ("CCRO3", "MOTV3"),
    ("TIMP3", "TIMS3"), ("ARZZ3", "AZZA3"), ("KROT3", "COGN3"),
    ("BRDT3", "VBBR3"), ("DTEX3", "DXCO3"), ("BVMF3", "B3SA3"),
    ("ELET3", "AXIA3"), ("BTOW3", "AMER3"),
)
CONTROLES_NEGATIVOS = (
    ("BRFS3", "MBRF3"),   # fusao: MBRF3 carrega o historico da Marfrig
    ("SUZB5", "SUZB3"),   # conversao de classe PNA -> ON
    ("RUMO3", "RAIL3"),   # reestruturacao com troca de acoes
)

# Categorias do resultado. Evitamos "aprovado": a palavra carrega autorizacao
# mesmo quando o texto diz que nao, e RUMO3 -> RAIL3 mostra o risco - os
# retornos batem, mas nao foi troca de codigo.
CAT_CONSISTENTE = "retornos_consistentes_sem_quebra_escala"
CAT_CONSISTENTE_QUEBRA = "retornos_consistentes_com_quebra_escala"
CAT_DIVERGENTE = "retornos_divergentes"
CAT_INCONCLUSIVO = "inconclusivo"
CAT_AUSENTE = "serie_ausente"


def ticker_terminal_da_cadeia(
    candidatos: pd.DataFrame, ticker: str, max_saltos: int = 5
) -> str | None:
    """
    Segue a cadeia de sucessores ate o ultimo codigo.

    O Yahoo nao ter mais o codigo intermediario nao quer dizer que o historico
    sumiu: ele costuma migrar tudo para o terminal. Na cadeia VVAR3 -> VIIA3 ->
    BHIA3, o VIIA3 nao existe mais no Yahoo, mas o BHIA3 pode carregar o
    historico inteiro, periodo do VVAR3 incluido.

    Para em bifurcacao. Com mais de um sucessor candidato - o ESTC3 aparece indo
    para YDUQ3 e para ALSO3 - nao ha cadeia unica a seguir, e adivinhar seria
    pior que nao responder.

    Continua sendo auditoria numerica: o Yahoo ter emendado a cadeia inteira nao
    diz que cada salto foi renomeacao societaria.
    """
    atual, visitados = ticker, {ticker}
    for _ in range(max_saltos):
        sucessores = candidatos.loc[
            candidatos["ticker_antigo"] == atual, "ticker_novo"
        ].unique()
        sucessores = [s for s in sucessores if s not in visitados]
        if len(sucessores) != 1:
            break            # fim da cadeia, ou bifurcacao ambigua
        atual = sucessores[0]
        visitados.add(atual)
    return atual if atual != ticker else None


def classificar_validacao_yf(m: dict) -> dict:
    """
    Traduz as metricas em categoria e motivo legivel.

    O criterio principal e o retorno, e nao o preco: e o retorno que o projeto
    usa, e fator de escala constante (split, bonificacao) some nele. A dispersao
    da razao de precos entra como contexto, sinalizando degrau de escala, nunca
    como reprovacao isolada.

    Consistencia numerica nao e identidade economica. Em RUMO3 -> RAIL3 os
    retornos batem (erro mediano 3e-08, correlacao 0,997), mas societariamente
    foi reestruturacao com troca de acoes, e nao renomeacao. Daqui sai evidencia
    e prioridade; autorizar continuidade depende de documento.
    """
    situacao = m.get("_situacao")
    if situacao == "erro_download":
        return {"resultado_validacao_yf": "erro_download",
                "motivo_validacao_yf": "falha ao consultar o yfinance"}
    if situacao == "serie_yahoo_ausente":
        return {"resultado_validacao_yf": CAT_AUSENTE,
                "motivo_validacao_yf": "yfinance nao tem serie para o ticker novo"}
    if situacao == "sem_cotahist":
        return {"resultado_validacao_yf": CAT_INCONCLUSIVO,
                "motivo_validacao_yf": "ticker antigo ausente do COTAHIST no periodo"}
    if situacao == "inconclusivo_pouca_amostra":
        n = m.get("n_datas_sobrepostas", 0)
        motivo = ("serie do Yahoo nao cobre o periodo do ticker antigo: "
                  "nao houve emenda" if n == 0
                  else f"apenas {n} datas sobrepostas, insuficiente")
        return {"resultado_validacao_yf": CAT_INCONCLUSIVO,
                "motivo_validacao_yf": motivo}

    erro = m.get("erro_retorno_mediano")
    p95 = m.get("erro_retorno_p95")
    corr = m.get("corr_retornos")
    disp = m.get("dispersao_razao_precos")
    razao = m.get("razao_precos_mediana")

    if erro is None or corr is None:
        return {"resultado_validacao_yf": CAT_INCONCLUSIVO,
                "motivo_validacao_yf": "metricas de retorno indisponiveis"}

    falhas = []
    if erro > LIMIAR_ERRO_RETORNO_MEDIANO:
        falhas.append(f"erro mediano de retorno {erro:.2e} acima de "
                      f"{LIMIAR_ERRO_RETORNO_MEDIANO:.0e}")
    if p95 is not None and p95 > LIMIAR_ERRO_RETORNO_P95:
        falhas.append(f"p95 do erro {p95:.4f} acima de {LIMIAR_ERRO_RETORNO_P95}")
    if corr < LIMIAR_CORR_RETORNOS:
        falhas.append(f"correlacao de retornos {corr:.4f} abaixo de "
                      f"{LIMIAR_CORR_RETORNOS}")

    if falhas:
        return {"resultado_validacao_yf": CAT_DIVERGENTE,
                "motivo_validacao_yf": "; ".join(falhas)}

    partes = [f"retornos reproduzem o COTAHIST (erro mediano {erro:.1e}, "
              f"correlacao {corr:.4f}, {m.get('n_datas_sobrepostas')} datas)"]
    if razao is not None and abs(razao - 1) > 0.01:
        partes.append(f"escala reajustada por fator {razao:.4f}")

    # Quebra de escala vira categoria propria em vez de aviso: retorno
    # consistente com razao de precos instavel quer dizer evento societario no
    # meio do periodo, que e o que separa troca de codigo de reestruturacao.
    if disp is not None and disp > LIMIAR_DISPERSAO_DEGRAU:
        partes.append(f"razao de precos INSTAVEL (dispersao {disp:.3f}): houve "
                      "evento societario no periodo, a serie nao e uma simples "
                      "renomeacao com fator unico")
        return {"resultado_validacao_yf": CAT_CONSISTENTE_QUEBRA,
                "motivo_validacao_yf": "; ".join(partes)}

    return {"resultado_validacao_yf": CAT_CONSISTENTE,
            "motivo_validacao_yf": "; ".join(partes)}


def tabela_revisao_mudanca_ticker(
    painel: pd.DataFrame,
    candidatos: pd.DataFrame,
    melhor_posicao: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    top_universo: int = 100,
    top_margem: int = 120,
    pregoes_curto: int = 400,
    janela_comparacao: int = 60,
) -> pd.DataFrame:
    """
    Reduz os candidatos aos materialmente relevantes e monta a planilha de
    conferencia humana.

    A relevancia e point-in-time, nunca por liquidez media dos 12 anos, que
    mistura epocas e favorece quem listou recentemente num periodo de volume
    alto. Um candidato entra na revisao se:

      a) alguma das pontas entrou no top-`top_universo` de alguma janela; ou
      b) alguma das pontas chegou ao top-`top_margem`, folga para quem fica
         perto do corte; ou
      c) uma das pontas tem historico curto demais para ser rankeada e a outra e
         relevante, ou seja, provavelmente seria elegivel com a serie
         continuada. Este e o caso mais perigoso: o papel novo com poucos
         pregoes reprova por cobertura e some do universo sem ninguem notar que
         a empresa continua ali.

    A tabela nao decide nada e sai com as colunas de decisao em branco:
    continuidade de preco prioriza candidato, mas nao substitui confirmacao
    societaria.
    """
    if candidatos.empty:
        return pd.DataFrame()

    cal = pd.DatetimeIndex(calendario)
    ordenado = painel.sort_values("DATA")
    por_ticker = ordenado.groupby("CODNEG")

    isin = por_ticker["CODISI"].first()
    especie = por_ticker["TIPO_PAPEL"].first()
    n_pregoes = por_ticker["DATA"].nunique()

    def _pos(tk: str) -> float:
        if tk in melhor_posicao.index:
            v = melhor_posicao.loc[tk, "melhor_posicao"]
            return float(v) if pd.notna(v) else float("inf")
        return float("inf")

    def _janela(tk: str) -> str:
        if tk in melhor_posicao.index:
            return str(melhor_posicao.loc[tk, "janela_melhor"])
        return ""

    def _janelas_relevantes(tk: str) -> int:
        if tk in melhor_posicao.index:
            return int(melhor_posicao.loc[tk, "janelas_elegivel"])
        return 0

    def _perfil(tk: str, ultimos: bool) -> tuple[float, float]:
        """Volume financeiro e numero de negocios medianos junto a transicao."""
        s = por_ticker.get_group(tk) if tk in por_ticker.groups else None
        if s is None or s.empty:
            return float("nan"), float("nan")
        fatia = s.tail(janela_comparacao) if ultimos else s.head(janela_comparacao)
        return float(fatia["VOLTOT"].median()), float(fatia["TOTNEG"].median())

    linhas = []
    for _, c in candidatos.iterrows():
        ant, nov = c["ticker_antigo"], c["ticker_novo"]
        pos_ant, pos_nov = _pos(ant), _pos(nov)
        preg_ant = int(n_pregoes.get(ant, 0))
        preg_nov = int(n_pregoes.get(nov, 0))

        relevante_a = min(pos_ant, pos_nov) <= top_universo
        relevante_b = min(pos_ant, pos_nov) <= top_margem
        curto_ant = preg_ant < pregoes_curto and pos_nov <= top_margem
        curto_nov = preg_nov < pregoes_curto and pos_ant <= top_margem
        if not (relevante_a or relevante_b or curto_ant or curto_nov):
            continue

        criterios = []
        if relevante_a:
            criterios.append(f"top{top_universo}")
        elif relevante_b:
            criterios.append(f"top{top_margem}")
        if curto_ant or curto_nov:
            criterios.append("ponta_curta")

        d_ant = pd.Timestamp(c["fim_antigo"])
        d_nov = pd.Timestamp(c["inicio_novo"])
        gap = int(cal.searchsorted(d_nov) - cal.searchsorted(d_ant))

        vol_ant, neg_ant = _perfil(ant, ultimos=True)
        vol_nov, neg_nov = _perfil(nov, ultimos=False)

        linhas.append({
            "ticker_antigo": ant,
            "ticker_novo": nov,
            "nome_antigo": c["empresa_antiga"],
            "nome_novo": c["empresa_nova"],
            "isin_antigo": isin.get(ant, ""),
            "isin_novo": isin.get(nov, ""),
            "especie_antiga": especie.get(ant, ""),
            "especie_nova": especie.get(nov, ""),
            "ultimo_pregao_antigo": c["fim_antigo"],
            "primeiro_pregao_novo": c["inicio_novo"],
            "intervalo_pregoes": gap,
            "preco_ultimo_antigo": c.get("preco_antigo"),
            "preco_primeiro_novo": c.get("preco_novo"),
            "razao_precos": (round(c["preco_novo"] / c["preco_antigo"], 4)
                             if c.get("preco_antigo") else None),
            "voltot_mediano_antes": round(vol_ant, 2) if pd.notna(vol_ant) else None,
            "voltot_mediano_depois": round(vol_nov, 2) if pd.notna(vol_nov) else None,
            "totneg_mediano_antes": neg_ant if pd.notna(neg_ant) else None,
            "totneg_mediano_depois": neg_nov if pd.notna(neg_nov) else None,
            "pregoes_antigo": preg_ant,
            "pregoes_novo": preg_nov,
            "melhor_posicao_antigo": None if pos_ant == float("inf") else int(pos_ant),
            "melhor_posicao_novo": None if pos_nov == float("inf") else int(pos_nov),
            "janela_melhor_antigo": _janela(ant),
            "janela_melhor_novo": _janela(nov),
            "janelas_elegivel_antigo": _janelas_relevantes(ant),
            "janelas_elegivel_novo": _janelas_relevantes(nov),
            "criterio_relevancia": "+".join(criterios),
            "metodo_deteccao": c["motivo"],
            "confianca_deteccao": c["confianca"],
            "tipo_evento": "",
            "proporcao_conversao": "",
            "fonte": "",
            "data_confirmacao": "",
            "revisor": "",
            "decisao": "",
            "observacoes": "",
        })

    if not linhas:
        return pd.DataFrame()
    return (pd.DataFrame(linhas)
            .sort_values(["melhor_posicao_antigo", "melhor_posicao_novo"],
                         na_position="last")
            .reset_index(drop=True))


def resumo_auditoria(aud: pd.DataFrame) -> dict:
    """Numeros de cabecalho da auditoria, para o log e para o manifesto."""
    return {
        "tickers": int(len(aud)),
        "elegiveis": int(aud["elegivel"].sum()),
        "reprovados_cobertura": int((~aud["elegivel_cobertura"]).sum()),
        "reprovados_negociacao": int((~aud["elegivel_negociacao"]).sum()),
        "reprovados_serie_integra": int((~aud["elegivel_serie_integra"]).sum()),
        "cobertura_mediana": float(aud["cobertura"].median()),
        "totneg_mediano_mediana": float(aud["totneg_mediano"].median()),
        "frac_suspeitos_mediana": float(aud["frac_dias_suspeitos"].median()),
    }


# Ausencia no Yahoo, em camada separada do detector de mudanca de ticker. O
# detector so enxerga sucessao com adjacencia temporal, entao nao ve conversao
# para classe que ja existia (VIVT4 -> VIVT3), fechamento de capital sem
# sucessor negociado (CIEL3) nem sucessao para instrumento fora do universo de
# acoes, como BDR (JBSS3).
AUSENCIA_VERSAO = "1.0.0"

CAUSA_RECENTE = "instrumento_recente_sem_serie_ainda"
CAUSA_ATIVO = "ativo_no_fim_sem_serie"
CAUSA_SUCESSOR_DETECTADO = "encerrado_com_sucessor_no_detector"
CAUSA_CLASSE_MIGRADA = "encerrado_com_outra_classe_do_emissor"
CAUSA_SEM_SUCESSOR = "encerrado_sem_sucessor_no_universo"

VIA_DOCUMENTAL = "revisao_documental_mudanca_ticker"
VIA_CONVERSAO = "conversao_classe_documentada"
VIA_COTAHIST = "preco_bruto_cotahist"
VIA_NAO_MATERIAL = "nao_material"


def _faixa_alcancada(posicao: float, faixas: tuple[int, ...]) -> str:
    """Menor faixa de liquidez que o ticker alcancou em alguma janela."""
    if pd.isna(posicao):
        return "nunca_elegivel"
    for f in sorted(faixas):
        if posicao <= f:
            return f"top{f}"
    return "fora_das_faixas"


def classificar_ausencia_yf(
    painel: pd.DataFrame,
    status_download: pd.DataFrame,
    candidatos: pd.DataFrame | None = None,
    melhor_posicao: pd.DataFrame | None = None,
    cfg: LiquidezConfig | None = None,
    dias_ativo_no_fim: int = 30,
    meses_instrumento_recente: int = 12,
) -> pd.DataFrame:
    """
    Por que o Yahoo nao tem serie para cada ticker, e se isso importa.

    Cruza a causa com a melhor posicao point-in-time porque so as duas juntas
    decidem: papel que nunca chegou perto de uma faixa pode faltar sem
    consequencia, papel que passou trinta janelas no top-10 nao.

    Produz diagnostico e prioridade. Nao baixa nada, nao altera o mapeamento de
    mudanca de ticker e nao autoriza continuidade.
    """
    cfg = cfg or LiquidezConfig()
    g = painel.groupby("CODNEG")
    perfil = pd.DataFrame({
        "primeira_data": g["DATA"].min(),
        "ultima_data": g["DATA"].max(),
        "pregoes": g["DATA"].nunique(),
        "voltot_mediano": g["VOLTOT"].median(),
        "totneg_mediano": g["TOTNEG"].median(),
        "tipo_papel": g["TIPO_PAPEL"].first(),
        "nome": g["NOMRES"].first(),
        "emissor_isin": g["CODISI"].first().str[2:6],
    })

    sem_serie = sorted(status_download.loc[
        status_download["estado_final"] == "serie_vazia", "ticker"])
    com_serie = set(status_download.loc[
        status_download["estado_final"].isin(["sucesso", "bloqueado"]), "ticker"])

    no_detector: set[str] = set()
    if candidatos is not None and len(candidatos):
        no_detector = set(candidatos["ticker_antigo"]) | set(candidatos["ticker_novo"])

    fim_amostra = painel["DATA"].max()
    corte_ativo = fim_amostra - pd.Timedelta(days=dias_ativo_no_fim)
    corte_recente = fim_amostra - pd.DateOffset(months=meses_instrumento_recente)

    # Irmao e outra classe do mesmo emissor com serie: e a pista de conversao de
    # classe, que o detector nao ve por falta de adjacencia.
    por_emissor: dict[str, list[str]] = {}
    for tk, em in perfil["emissor_isin"].items():
        if tk in com_serie:
            por_emissor.setdefault(em, []).append(tk)

    linhas = []
    for tk in sem_serie:
        if tk not in perfil.index:
            continue
        p = perfil.loc[tk]
        irmaos = [t for t in por_emissor.get(p["emissor_isin"], []) if t != tk]
        ativo = p["ultima_data"] >= corte_ativo
        recente = p["primeira_data"] >= corte_recente

        if ativo:
            causa = CAUSA_RECENTE if recente else CAUSA_ATIVO
        elif tk in no_detector:
            causa = CAUSA_SUCESSOR_DETECTADO
        elif irmaos:
            causa = CAUSA_CLASSE_MIGRADA
        else:
            causa = CAUSA_SEM_SUCESSOR

        pos = float("nan")
        if melhor_posicao is not None and tk in melhor_posicao.index:
            pos = melhor_posicao.at[tk, "melhor_posicao"]
        faixa = _faixa_alcancada(pos, cfg.faixas)

        if faixa in ("nunca_elegivel", "fora_das_faixas"):
            via = VIA_NAO_MATERIAL
        elif causa == CAUSA_SUCESSOR_DETECTADO:
            via = VIA_DOCUMENTAL
        elif causa == CAUSA_CLASSE_MIGRADA:
            via = VIA_CONVERSAO
        else:
            via = VIA_COTAHIST

        linhas.append({
            "ticker": tk,
            "causa_provavel": causa,
            "via_recuperacao": via,
            "melhor_posicao_pit": pos,
            "faixa_alcancada": faixa,
            "primeira_data": p["primeira_data"].date(),
            "ultima_data": p["ultima_data"].date(),
            "pregoes": int(p["pregoes"]),
            "voltot_mediano": float(p["voltot_mediano"]),
            "totneg_mediano": float(p["totneg_mediano"]),
            "tipo_papel": p["tipo_papel"],
            "nome": p["nome"],
            "emissor_isin": p["emissor_isin"],
            "no_detector_mudanca_ticker": tk in no_detector,
            "irmaos_mesmo_emissor_com_serie": ",".join(sorted(irmaos)),
            "ausencia_versao": AUSENCIA_VERSAO,
        })

    if not linhas:
        return pd.DataFrame()
    return (pd.DataFrame(linhas)
            .sort_values(["melhor_posicao_pit", "voltot_mediano"],
                         ascending=[True, False], na_position="last")
            .reset_index(drop=True))
