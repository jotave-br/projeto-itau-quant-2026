"""
Retornos e indicadores de liquidez.

As fontes cumprem papeis diferentes. A estimacao e o P&L usam PREULT bruto do
COTAHIST, com tratamentos distintos para eventos corporativos. O preco ajustado
do yfinance fica restrito a auditorias e robustez em suporte comum. O volume
financeiro vem do VOLTOT do COTAHIST, ja calculado em reais pela bolsa.

Vale aqui a mesma ordem da auditoria - marcar ausencia, calcular sobre o
observado, so entao decidir tratamento. Retorno diario tem que ser de um pregao
so: se o papel ficou tres pregoes sem negociar, a variacao entre um negocio e
outro e soma de tres dias concentrada num ponto, e trata-la como retorno diario
inflaria a volatilidade e criaria correlacao espuria justamente nos iliquidos,
que sao os que precisamos examinar com mais cuidado. Por isso o retorno so sai
quando os dois pregoes consecutivos do calendario tem observacao; fora disso,
NaN explicito.

Cada ticker e uma serie independente. Emendar codigos diferentes so pode
acontecer depois da revisao documental registrada em
data/reference/mudancas_ticker.csv, e nao aqui.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def painel_precos_ajustados(
    series_yf: dict[str, pd.DataFrame],
    calendario: pd.DatetimeIndex,
    coluna: str = "Adj Close",
) -> pd.DataFrame:
    """
    Monta o painel (datas x tickers) de precos, alinhado ao calendario da B3.

    Alinhar nao preenche: data sem observacao fica NaN. Preencher, se um dia
    for o caso, e decisao explicita de quem chama, nunca efeito colateral do
    alinhamento.
    """
    colunas = {}
    for ticker, df in series_yf.items():
        if df is None or df.empty or coluna not in df.columns:
            continue
        s = df.copy()
        s["Date"] = pd.to_datetime(s["Date"])
        s = s.set_index("Date")[coluna].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        colunas[ticker] = s
    if not colunas:
        return pd.DataFrame(index=pd.DatetimeIndex(calendario))
    return pd.DataFrame(colunas).reindex(pd.DatetimeIndex(calendario))


def retornos_simples(precos: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Variacao percentual entre pregoes consecutivos do calendario.

    Nao usamos `pct_change`: ele pula NaN por padrao, emenda por cima dos
    buracos e devolve retorno de varios dias disfarcado de diario. Aqui, dia
    anterior sem preco no calendario vira NaN.
    """
    anterior = precos.shift(1)
    r = precos / anterior - 1.0
    return r.where(precos.notna() & anterior.notna())


def retornos_log(precos: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Retorno logaritmico, que soma de forma limpa ao longo do tempo.

    Mesma disciplina de buraco do `retornos_simples`. Tanto faz qual das duas
    formas usar, desde que seja a mesma para todos os ativos.
    """
    anterior = precos.shift(1)
    r = np.log(precos / anterior)
    return r.where(precos.notna() & anterior.notna())


def mascara_retorno_valido(
    retornos: pd.DataFrame,
    painel_cotahist: pd.DataFrame | None = None,
    exigir_negociacao: bool = True,
) -> pd.DataFrame:
    """
    Onde o retorno pode ser usado.

    So vale com negociacao efetiva nos dois pregoes que o formam. Papel que nao
    negociou carrega preco herdado do dia anterior, entao a variacao ali e zero
    por falta de negocio, e nao por decisao de mercado - e o caminho mais curto
    para o preco velho entrar na regressao.
    """
    valido = retornos.notna()
    if painel_cotahist is None or not exigir_negociacao:
        return valido

    negociou = (painel_cotahist.reindex(index=retornos.index,
                                        columns=retornos.columns) > 0)
    negociou = negociou.fillna(False)
    return valido & negociou & negociou.shift(1, fill_value=False)


# Retorno a partir do preco bruto do COTAHIST, que nao tem o vies de
# sobrevivencia do yfinance. O PREULT nao e ajustado, mas entre dois eventos o
# ajuste e fator constante e cancela na razao, entao o retorno bruto iguala o
# ajustado - so o dia ex fica contaminado, e o COTAHIST marca esse dia no
# ESPECI. A serie mede variacao de preco entre eventos, e nao retorno total ao
# acionista: o dia do provento e descartado, nunca reconstruido.
SEGMENTOS_GOVERNANCA = frozenset({"NM", "N1", "N2", "MA", "MB", ""})


def token_evento_especi(especi: pd.Series) -> pd.Series:
    """
    Token de evento do ESPECI, ou "" ("ON  ED  NM" -> "ED").

    Detectamos pelo prefixo "E" em vez de listar os marcadores conhecidos
    porque a B3 usa mais de vinte e cria novos. Lista fechada erraria calada no
    marcador inedito, deixando passar degrau de provento como retorno; o
    prefixo erra descartando um dia a mais.
    """
    tokens = especi.astype(str).str.split()
    out = pd.Series("", index=especi.index, dtype=object)
    for i in (2, 1):                      # 0 e o tipo do papel, tratado em dados.py
        tok = tokens.str[i].fillna("")
        eh_evento = ((tok.str.startswith("E") | tok.str.startswith("*"))
                     & ~tok.isin(SEGMENTOS_GOVERNANCA))
        out = out.mask(eh_evento, tok)    # posicao 1 prevalece sobre a 2
    return out


def marcar_eventos_especi(especi: pd.Series) -> pd.Series:
    """Dias em que o ESPECI carrega algum marcador de evento."""
    return token_evento_especi(especi) != ""


def retornos_preco_bruto_cotahist(
    painel_cotahist: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    painel_volume: pd.DataFrame | None = None,
    mascarar_dia_seguinte: bool = False,
) -> pd.DataFrame:
    """
    Retornos do PREULT com as fronteiras de evento removidas.

    O marcador do ESPECI identifica um periodo, e nao o dia ex isolado: a B3
    mantem o marcador por varios pregoes. Descartar todo dia marcado tirava
    metade da serie de ITUB4 e BBDC4, pagadores frequentes, sem ganhar precisao
    contra o yfinance. O degrau de preco esta na transicao, entao removemos o
    retorno em que o token muda.

    Comparar o token, e nao so a presenca dele, pega dois eventos colados (ED
    seguido de EJ), que uma regra de "inicio de sequencia" leria como um so.

    A remocao e por ticker: o mesmo pregao continua valido para quem nao teve
    evento.

    `mascarar_dia_seguinte` foi medido e desligado - custa observacao sem mudar
    erro mediano nem p95. Fica disponivel para robustez.
    """
    precos = painel_cotahist.pivot_table(
        index="DATA", columns="CODNEG", values="PREULT", aggfunc="last"
    ).reindex(pd.DatetimeIndex(calendario))
    precos = precos.where(precos > 0)     # PREULT zero em papel que nao negociou

    token = (painel_cotahist
             .assign(_tok=token_evento_especi(painel_cotahist["ESPECI"]))
             .pivot_table(index="DATA", columns="CODNEG", values="_tok",
                          aggfunc="last")
             .reindex(pd.DatetimeIndex(calendario))
             .reindex(columns=precos.columns)
             .fillna(""))
    fronteira = (token != "") & (token != token.shift(1).fillna(""))
    if mascarar_dia_seguinte:
        fronteira = fronteira | fronteira.shift(1, fill_value=False)

    r = retornos_simples(precos).where(~fronteira)
    return r.where(mascara_retorno_valido(r, painel_volume))


def painel_token_evento(
    painel_cotahist: pd.DataFrame, calendario: pd.DatetimeIndex
) -> pd.DataFrame:
    """Painel (datas x tickers) do token de evento, "" onde nao ha."""
    return (painel_cotahist
            .assign(_tok=token_evento_especi(painel_cotahist["ESPECI"]))
            .pivot_table(index="DATA", columns="CODNEG", values="_tok",
                         aggfunc="last")
            .reindex(pd.DatetimeIndex(calendario))
            .fillna(""))


def painel_fronteiras_evento(
    painel_cotahist: pd.DataFrame, calendario: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Painel booleano das fronteiras de evento: o dia em que o token muda.

    Mesma regra da mascara da estimacao, exposta sozinha para contar quantos
    posicao-dias atravessam uma fronteira. A contagem mede incidencia, nao a
    magnitude nem o sinal do efeito sobre o P&L.
    """
    token = painel_token_evento(painel_cotahist, calendario)
    return (token != "") & (token != token.shift(1).fillna(""))


# O P&L nao herda a mascara da estimacao. Como a fonte abaixo usa apenas PREULT,
# sem fluxo de dividendos ou JCP, o resultado e retorno de preco bruto, nao
# retorno economico total. O degrau do dia ex aparece como perda na ponta
# comprada e ganho na vendida; painel_fronteiras_evento mede quantas posicoes
# ficaram expostas a essa limitacao.
def retornos_pnl_marcacao(
    painel_cotahist: pd.DataFrame,
    calendario: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Retornos de P&L com marcacao a ultimo preco negociado.

    Excecao localizada a regra de nao fazer forward-fill, que vale para a
    estimacao, onde preco herdado fabricaria co-movimento. Aqui o problema e
    outro: carteira precisa de um valor por dia. Papel que nao negociou fica
    marcado no ultimo preco (retorno zero) e o salto aparece inteiro no pregao
    em que volta a negociar, entao a soma ao longo do holding e a variacao
    entre precos efetivamente observados.

    Nunca preenchemos para tras - antes da primeira observacao o papel nao
    existe. Depois da ultima, em caso de deslistagem, a marcacao congela no
    ultimo preco e o desfecho societario posterior fica de fora.
    """
    precos = painel_cotahist.pivot_table(
        index="DATA", columns="CODNEG", values="PREULT", aggfunc="last"
    ).reindex(pd.DatetimeIndex(calendario))
    marcado = precos.where(precos > 0).ffill()
    anterior = marcado.shift(1)
    r = marcado / anterior - 1.0
    return r.where(marcado.notna() & anterior.notna())


def medir_travessias_de_evento(
    posicoes: pd.DataFrame,
    fronteiras: pd.DataFrame,
) -> dict:
    """
    Quantos posicao-dias do backtest caem numa fronteira de evento.

    Cada travessia identifica exposicao ao degrau de um evento. A contagem mede
    frequencia, nao o tamanho monetario nem a direcao do vies.
    """
    f = fronteiras.reindex(index=posicoes.index,
                           columns=posicoes.columns).fillna(False)
    ativa = posicoes != 0
    dias_ativos = int(ativa.sum().sum())
    travessias = int((ativa & f).sum().sum())
    return {
        "posicao_dias": dias_ativos,
        "travessias_de_evento": travessias,
        "frac_travessias": travessias / dias_ativos if dias_ativos else 0.0,
    }


# Rotulos sobre a qualidade da prova de cada token, e nao veredito sobre
# manter ou remover o token da mascara.
EVID_SUSTENTADA = "sustentado_pela_referencia"
EVID_MISTA = "evidencia_mista"
EVID_NAO_SUSTENTADA = "nao_sustentado_pela_referencia"
EVID_AMOSTRA = "amostra_insuficiente"
EVID_SEM_REF = "sem_referencia_externa"

# Convencoes, nao conclusoes estatisticas. Ficam nomeadas para dar para medir a
# sensibilidade a elas.
MIN_FRONTEIRAS_COM_REFERENCIA = 30
LIMIAR_ERRO_MATERIAL = 0.0010          # 10 bps
LIMIAR_FRAC_MISTA = 0.10               # fracao de fronteiras que se move


def _classificar_evidencia(
    n_ref: int,
    mediana: float | None,
    frac_acima_10bps: float | None,
    min_fronteiras: int = MIN_FRONTEIRAS_COM_REFERENCIA,
    limiar_erro: float = LIMIAR_ERRO_MATERIAL,
    limiar_frac: float = LIMIAR_FRAC_MISTA,
) -> str:
    """
    Que forca tem a prova para este token, e nao se ele deve sair da mascara.

    A mediana sozinha nao separa: token cuja remocao tipica nao move nada mas
    cuja cauda move muito e diferente de token em que nada se move nunca. O
    primeiro e evidencia mista, o segundo e ausencia de sustentacao - dai a
    fracao de fronteiras acima de 10 bps entrar na regra.

    Token com evidencia fraca continua mascarado por conservadorismo. Abrir
    excecao olhando estes numeros seria ajustar a regra ao resultado.
    """
    if not n_ref:
        return EVID_SEM_REF
    if n_ref < min_fronteiras:
        return EVID_AMOSTRA
    if mediana is not None and mediana > limiar_erro:
        return EVID_SUSTENTADA
    if frac_acima_10bps is not None and frac_acima_10bps >= limiar_frac:
        return EVID_MISTA
    return EVID_NAO_SUSTENTADA


def reclassificar_evidencia(
    tabela: pd.DataFrame, min_fronteiras: int, **limiares
) -> pd.Series:
    """
    Reclassifica a tabela ja calculada com outro corte de amostra minima.

    Mede a sensibilidade ao corte sem refazer a validacao inteira. O corte e
    convencao, e se a leitura muda quando ele muda, isso tem que aparecer.
    """
    return tabela.apply(
        lambda r: _classificar_evidencia(
            int(r.get("fronteira_n_suporte_comum", 0) or 0),
            r.get("fronteira_erro_mediano"),
            r.get("fronteira_frac_acima_10bps"),
            min_fronteiras=min_fronteiras, **limiares),
        axis=1)


def validar_fronteiras_por_token(
    painel_cotahist: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    retornos_referencia: pd.DataFrame,
    painel_volume: pd.DataFrame | None = None,
    min_fronteiras_com_referencia: int = MIN_FRONTEIRAS_COM_REFERENCIA,
    limiar_erro_material: float = LIMIAR_ERRO_MATERIAL,
    limiar_frac_mista: float = LIMIAR_FRAC_MISTA,
) -> pd.DataFrame:
    """
    Que evidencia existe a favor da remocao, em cada tipo de evento.

    Nos pontos que a regra remove, compara o retorno bruto com o de referencia.
    Onde a remocao esta certa os dois discordam, e a discordancia e o degrau do
    provento; onde o erro fica no ruido, a remocao pode estar custando
    observacao a toa. `mantido_*` faz a pergunta simetrica nos dias marcados
    que a regra mantem.

    Sao dois denominadores, que nunca se misturam:

        n_total_cotahist   fronteiras do token no painel inteiro, ou seja, o
                           que a regra efetivamente remove.
        n_suporte_comum    subconjunto com valor de referencia, o unico lugar
                           onde da para medir erro.

    A referencia so existe onde o yfinance tem serie, isto e, entre
    sobreviventes. Token com cobertura de referencia baixa nao esta validado
    nem invalidado: esta sem prova.
    """
    precos = (painel_cotahist.pivot_table(index="DATA", columns="CODNEG",
                                          values="PREULT", aggfunc="last")
              .reindex(pd.DatetimeIndex(calendario)))
    precos = precos.where(precos > 0)
    bruto = retornos_simples(precos)
    candidatos = mascara_retorno_valido(bruto, painel_volume)

    token = (painel_token_evento(painel_cotahist, calendario)
             .reindex(columns=precos.columns).fillna(""))
    marcado = token != ""
    fronteira = marcado & (token != token.shift(1).fillna(""))

    ref = retornos_referencia.reindex(index=bruto.index, columns=bruto.columns)
    erro = (bruto - ref).abs()

    def _stats(sel: pd.DataFrame, prefixo: str) -> dict:
        sel = sel & candidatos
        n_total = int(sel.sum().sum())
        e = erro.where(sel).stack().dropna()
        d = {f"{prefixo}_n_total_cotahist": n_total,
             f"{prefixo}_n_suporte_comum": int(len(e)),
             f"{prefixo}_cobertura_referencia": (
                 round(len(e) / n_total, 4) if n_total else None)}
        if e.empty:
            return d
        d.update({
            f"{prefixo}_erro_mediano": float(e.median()),
            f"{prefixo}_erro_p90": float(e.quantile(0.90)),
            f"{prefixo}_erro_p95": float(e.quantile(0.95)),
            f"{prefixo}_erro_p99": float(e.quantile(0.99)),
            f"{prefixo}_erro_max": float(e.max()),
            f"{prefixo}_frac_acima_1bp": float((e > 0.0001).mean()),
            f"{prefixo}_frac_acima_10bps": float((e > 0.0010).mean()),
            f"{prefixo}_frac_acima_50bps": float((e > 0.0050).mean()),
        })
        return d

    linhas = []
    for tok in sorted(t for t in pd.unique(token.values.ravel()) if t):
        eh_tok = token == tok
        linha = {"token": tok,
                 "dias_marcados_n_total_cotahist": int((eh_tok & candidatos).sum().sum())}
        linha.update(_stats(eh_tok & fronteira, "fronteira"))
        linha.update(_stats(eh_tok & ~fronteira, "mantido"))
        linha["classificacao_evidencia"] = _classificar_evidencia(
            linha.get("fronteira_n_suporte_comum", 0),
            linha.get("fronteira_erro_mediano"),
            linha.get("fronteira_frac_acima_10bps"),
            min_fronteiras=min_fronteiras_com_referencia,
            limiar_erro=limiar_erro_material,
            limiar_frac=limiar_frac_mista)
        linhas.append(linha)

    if not linhas:
        return pd.DataFrame()
    return (pd.DataFrame(linhas)
            .sort_values("fronteira_n_total_cotahist", ascending=False)
            .reset_index(drop=True))


FATORES_GRUPAMENTO = (2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100)


def _razao_de_grupamento(
    preco_anterior, preco_atual, tolerancia: float = 0.02
) -> tuple[float | None, str | None]:
    """
    A razao de precos e proxima de um grupamento ou desdobramento redondo?

    MAPT4 foi de R$ 10,00 para R$ 40,00 num pregao, sem marcador no ESPECI.
    Quatro vezes exatos nao e movimento de mercado, e grupamento 1:4 que a B3
    nao sinalizou. Sem esta checagem o caso cai em "sem causa", a gaveta onde
    erro de dado se esconde. Ainda assim e indicio: papel de centavos pode
    dobrar de verdade.
    """
    if preco_anterior is None or preco_atual is None:
        return None, None
    if pd.isna(preco_anterior) or pd.isna(preco_atual) or preco_anterior <= 0:
        return None, None
    razao = float(preco_atual) / float(preco_anterior)
    for f in FATORES_GRUPAMENTO:
        if abs(razao / f - 1.0) <= tolerancia:
            return razao, f"1:{f}"
        if abs(razao * f - 1.0) <= tolerancia:
            return razao, f"{f}:1"
    return razao, None


def auditar_retornos_extremos(
    retornos_painel: pd.DataFrame,
    painel_cotahist: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    candidatos_mudanca: pd.DataFrame | None = None,
    limiar: float = 0.5,
    janela_proximidade_dias: int = 10,
) -> pd.DataFrame:
    """
    Retornos extremos que sobraram no painel, com o contexto para julga-los.

    Nao winsoriza, nao corta e nao corrige. Salto de 60% num papel iliquido
    pode ser real; num liquido, quase sempre e evento nao capturado. A
    diferenca so aparece olhando volume, numero de negocios, ESPECI do dia e
    proximidade de mudanca de ticker, e por isso tudo isso sai na tabela.

    `causa_provavel` e hipotese ordenada por forca do indicio, nunca veredito.
    """
    extremos = retornos_painel.abs() > limiar
    if not extremos.any().any():
        return pd.DataFrame()

    precos = (painel_cotahist.pivot_table(index="DATA", columns="CODNEG",
                                          values="PREULT", aggfunc="last")
              .reindex(pd.DatetimeIndex(calendario)))
    volume = painel_volume_financeiro(painel_cotahist).reindex(
        index=calendario, columns=retornos_painel.columns)
    negocios = painel_numero_negocios(painel_cotahist).reindex(
        index=calendario, columns=retornos_painel.columns)
    especi = (painel_cotahist.pivot_table(index="DATA", columns="CODNEG",
                                          values="ESPECI", aggfunc="last")
              .reindex(pd.DatetimeIndex(calendario)))
    token = painel_token_evento(painel_cotahist, calendario)

    bordas: dict[str, list[pd.Timestamp]] = {}
    if candidatos_mudanca is not None and len(candidatos_mudanca):
        for col_tk, col_dt in (("ticker_antigo", "fim_antigo"),
                               ("ticker_novo", "inicio_novo")):
            if col_tk in candidatos_mudanca and col_dt in candidatos_mudanca:
                for tk, dt in zip(candidatos_mudanca[col_tk],
                                  pd.to_datetime(candidatos_mudanca[col_dt],
                                                 errors="coerce")):
                    if pd.notna(dt):
                        bordas.setdefault(tk, []).append(dt)

    linhas = []
    marcados = extremos.stack()
    for (data, ticker) in marcados[marcados].index:
        r = retornos_painel.at[data, ticker]
        p_ant = precos[ticker].shift(1).get(data)
        p_atu = precos.at[data, ticker] if ticker in precos.columns else None
        tok_hoje = token.at[data, ticker] if ticker in token.columns else ""
        tok_ontem = (token[ticker].shift(1).get(data)
                     if ticker in token.columns else "")
        neg = negocios.at[data, ticker] if ticker in negocios.columns else None
        vol = volume.at[data, ticker] if ticker in volume.columns else None

        dias_da_borda = None
        for dt in bordas.get(ticker, []):
            d = abs((data - dt).days)
            dias_da_borda = d if dias_da_borda is None else min(dias_da_borda, d)

        razao, fator = _razao_de_grupamento(p_ant, p_atu)
        if dias_da_borda is not None and dias_da_borda <= janela_proximidade_dias:
            causa = "proximo_de_mudanca_de_ticker"
        elif tok_hoje or tok_ontem:
            causa = "evento_marcado_na_vizinhanca"
        elif fator is not None:
            causa = "possivel_grupamento_nao_marcado"
        elif p_ant is not None and pd.notna(p_ant) and p_ant < 1.0:
            causa = "preco_baixo_efeito_de_tick"
        elif neg is not None and pd.notna(neg) and neg <= 10:
            causa = "papel_iliquido"
        else:
            causa = "sem_causa_identificada"

        linhas.append({
            "ticker": ticker,
            "data": data.date(),
            "retorno": float(r),
            "preco_anterior": None if p_ant is None or pd.isna(p_ant) else float(p_ant),
            "preco_atual": None if p_atu is None or pd.isna(p_atu) else float(p_atu),
            "volume_financeiro": None if vol is None or pd.isna(vol) else float(vol),
            "numero_negocios": None if neg is None or pd.isna(neg) else float(neg),
            "especi": especi.at[data, ticker] if ticker in especi.columns else "",
            "token_evento_hoje": tok_hoje,
            "token_evento_ontem": tok_ontem or "",
            "dias_ate_mudanca_ticker": dias_da_borda,
            "razao_precos": None if razao is None else round(razao, 4),
            "fator_redondo": fator or "",
            "causa_provavel": causa,
        })
    return (pd.DataFrame(linhas)
            .sort_values("retorno", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True))


def painel_volume_financeiro(painel_cotahist: pd.DataFrame) -> pd.DataFrame:
    """
    Painel (datas x tickers) do volume financeiro em reais.

    Vem do VOLTOT do COTAHIST, calculado pela bolsa. Nao recalculamos por preco
    vezes quantidade: escolher qual preco usar traria de volta a ambiguidade de
    ajuste que a fonte oficial ja resolveu.
    """
    return painel_cotahist.pivot_table(
        index="DATA", columns="CODNEG", values="VOLTOT", aggfunc="sum"
    )


def painel_numero_negocios(painel_cotahist: pd.DataFrame) -> pd.DataFrame:
    """Painel do numero de negocios por dia, o detector direto de iliquidez."""
    return painel_cotahist.pivot_table(
        index="DATA", columns="CODNEG", values="TOTNEG", aggfunc="sum"
    )


def liquidez_mediana(
    volume: pd.DataFrame, janela: int, min_observacoes: int | None = None
) -> pd.DataFrame:
    """
    Mediana movel do volume financeiro.

    Mediana e nao media: um dia de leilao gigante nao deve promover papel
    iliquido no ranking.
    """
    min_obs = min_observacoes or max(1, janela // 2)
    return volume.rolling(janela, min_periods=min_obs).median()


def proporcao_dias_negociados(volume: pd.DataFrame, janela: int) -> pd.DataFrame:
    """Fracao dos ultimos `janela` pregoes em que o papel efetivamente negociou."""
    negociou = (volume > 0).astype(float).where(volume.notna())
    return negociou.rolling(janela, min_periods=1).mean()


def volatilidade_realizada(
    retornos: pd.DataFrame, janela: int, anualizar: bool = True,
    dias_uteis_ano: int = 252, min_observacoes: int | None = None,
) -> pd.DataFrame:
    """
    Desvio-padrao movel dos retornos, para o dimensionamento de posicao.

    O `rolling` so olha para tras, entao da para calcular sobre o painel
    inteiro sem vazamento: o valor em t depende de t e do passado.
    """
    min_obs = min_observacoes or max(2, janela // 2)
    vol = retornos.rolling(janela, min_periods=min_obs).std()
    return vol * np.sqrt(dias_uteis_ano) if anualizar else vol


def cobertura_yahoo_vs_cotahist(
    painel_precos: pd.DataFrame,
    painel_cotahist_volume: pd.DataFrame,
    status_download: pd.DataFrame | None = None,
    tickers_continuidade_bloqueada: set[str] | None = None,
    tickers_cobertos_por_terminal: dict[str, str] | None = None,
    limiar_completa: float = 0.98,
    limiar_parcial: float = 0.50,
) -> pd.DataFrame:
    """
    Cobertura do Yahoo em relacao aos pregoes observados no COTAHIST.

    Mede cobertura de fonte: em quantos dos pregoes com negociacao registrada
    na B3 o yfinance tem preco ajustado. Nao e o tamanho do vies de
    sobrevivencia - ausencia no Yahoo tambem vem de migracao de ticker, falha
    de download ou comportamento da API, nada disso relacionado a empresa ter
    deixado de existir. Serve para dimensionar o risco, e ler o numero como "o
    survivorship bias e de X%" seria exagero.

    Classificacao em `situacao_cobertura`:
        cobertura_completa              >= limiar_completa
        cobertura_parcial               entre os dois limiares
        historico_sob_codigo_posterior  o Yahoo migrou a serie para outro codigo
        ausente                         sem serie e sem substituto conhecido
        erro_temporario                 falha de download, nao ausencia real
        continuidade_bloqueada          candidato a mudanca de ticker pendente
                                        de revisao documental
    """
    vol = painel_cotahist_volume.reindex(
        index=painel_precos.index, columns=painel_precos.columns)
    negociou = (vol > 0).fillna(False)
    tem_preco = painel_precos.notna()

    d = pd.DataFrame({
        "pregoes_com_negociacao": negociou.sum(),
        "pregoes_com_preco_yf": tem_preco.sum(),
        "negociou_e_tem_preco": (negociou & tem_preco).sum(),
        "negociou_sem_preco": (negociou & ~tem_preco).sum(),
        "preco_sem_negociacao": (~negociou & tem_preco).sum(),
    })
    # Cobertura e a intersecao sobre os dias negociados, nunca a razao entre os
    # totais: dividir totais da valor acima de 1 em papel iliquido, porque o
    # Yahoo reporta preco em muito mais dias do que a acao negociou - sintoma de
    # preco velho, e nao cobertura excedente.
    d["cobertura_yahoo_vs_cotahist"] = (
        d["negociou_e_tem_preco"] / d["pregoes_com_negociacao"].replace(0, np.nan))
    d["gap_cobertura_yahoo"] = 1.0 - d["cobertura_yahoo_vs_cotahist"]
    # O inverso: dias com cotacao no Yahoo e sem negocio na bolsa. Nao e falha
    # de cobertura, e preco velho entrando pela fonte de precos.
    d["frac_preco_sem_negociacao"] = (
        d["preco_sem_negociacao"] / d["pregoes_com_preco_yf"].replace(0, np.nan))

    erro = set()
    if status_download is not None and "status" in status_download.columns:
        erro = set(status_download.loc[
            status_download["status"] == "erro_persistente", "ticker"])
    bloqueados = tickers_continuidade_bloqueada or set()
    terminais = tickers_cobertos_por_terminal or {}

    def _situacao(tk: str, cob: float) -> str:
        if tk in erro:
            return "erro_temporario"
        if tk in bloqueados:
            return "continuidade_bloqueada"
        if tk in terminais:
            return "historico_sob_codigo_posterior"
        if pd.isna(cob) or cob <= 0:
            return "ausente"
        if cob >= limiar_completa:
            return "cobertura_completa"
        if cob >= limiar_parcial:
            return "cobertura_parcial"
        return "cobertura_parcial"

    d["situacao_cobertura"] = [
        _situacao(tk, c) for tk, c in d["cobertura_yahoo_vs_cotahist"].items()]
    d["ticker_terminal"] = [terminais.get(tk, "") for tk in d.index]
    d["risco_cobertura_fonte"] = np.where(
        d["situacao_cobertura"].isin(["ausente", "erro_temporario"]), "alto",
        np.where(d["situacao_cobertura"] == "cobertura_parcial", "medio", "baixo"))
    return d.sort_values("cobertura_yahoo_vs_cotahist", na_position="first")


def fingerprint_retornos(serie: pd.Series, casas: int = 8) -> str:
    """
    Impressao digital dos retornos observados de um ticker.

    Detecta o Yahoo devolvendo a mesma serie sob codigos diferentes, o que
    acontece quando ele migra o historico de um ticker renomeado. Series
    identicas nao podem entrar na rede como instrumentos diferentes: seria o
    mesmo ativo contado duas vezes, e o par entre elas teria correlacao
    perfeita por construcao.
    """
    import hashlib

    s = serie.dropna().round(casas)
    if s.empty:
        return ""
    bruto = "|".join(f"{d:%Y%m%d}:{v}" for d, v in s.items())
    return hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:16]


def detectar_series_duplicadas(
    painel_retornos: pd.DataFrame,
    limiar_correlacao: float = 0.999,
    min_sobreposicao: int = 60,
) -> pd.DataFrame:
    """
    Pares de tickers cujas series de retorno sao identicas ou quase.

    Nao resolve a duplicidade escolhendo um lado: devolve tabela de auditoria e
    a identidade canonica sai do mapeamento documental. Escolher aqui seria
    decidir questao societaria com estatistica.
    """
    fps = {c: fingerprint_retornos(painel_retornos[c]) for c in painel_retornos.columns}

    linhas = []
    por_fp: dict[str, list[str]] = {}
    for tk, fp in fps.items():
        if fp:
            por_fp.setdefault(fp, []).append(tk)
    identicos = {tuple(sorted(g)) for g in por_fp.values() if len(g) > 1}
    for grupo in identicos:
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                linhas.append({"ticker_a": grupo[i], "ticker_b": grupo[j],
                               "tipo": "fingerprint_identico", "correlacao": 1.0})

    validos = painel_retornos.loc[:, painel_retornos.notna().sum() >= min_sobreposicao]
    if validos.shape[1] > 1:
        corr = validos.corr(min_periods=min_sobreposicao)
        cols = list(corr.columns)
        ja = {(a, b) for a, b, *_ in
              [(l["ticker_a"], l["ticker_b"]) for l in linhas]} if linhas else set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iat[i, j]
                if pd.notna(c) and c >= limiar_correlacao:
                    par = tuple(sorted((cols[i], cols[j])))
                    if par in ja:
                        continue
                    linhas.append({"ticker_a": par[0], "ticker_b": par[1],
                                   "tipo": "correlacao_quase_perfeita",
                                   "correlacao": round(float(c), 6)})

    if not linhas:
        return pd.DataFrame(columns=["ticker_a", "ticker_b", "tipo", "correlacao"])

    res = pd.DataFrame(linhas)
    for lado in ("a", "b"):
        tks = res[f"ticker_{lado}"]
        res[f"fingerprint_{lado}"] = tks.map(fps)
        res[f"obs_{lado}"] = tks.map(painel_retornos.notna().sum())
        res[f"primeiro_{lado}"] = tks.map(
            {c: painel_retornos[c].first_valid_index() for c in painel_retornos.columns})
        res[f"ultimo_{lado}"] = tks.map(
            {c: painel_retornos[c].last_valid_index() for c in painel_retornos.columns})
    return res.sort_values(["tipo", "correlacao"], ascending=[True, False])
