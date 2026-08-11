"""
Classificacao setorial curada, pre-requisito da formacao de pares.

O setor nao esta no COTAHIST. Ele vem de `data/reference/setores_b3.csv`,
curado a mao, e essa tabela e entrada da metodologia, nao resultado - dai cada
linha carregar fonte, data e evidencia.

Duas origens, que nunca se misturam na mesma justificativa: a tabela oficial da
B3 para o que esta listado hoje, e curadoria para deslistado e instrumento
transformado. A segunda existe porque a primeira so tem o presente: CIEL3,
JBSS3 e GNDI3 nao estao na oficial de hoje, e montar o setor so com ela
reintroduziria o vies de sobrevivencia que a troca da fonte de retorno tirou.

O setor do Yahoo e proibido (nao e oficial, nao e point-in-time e nao cobre
deslistado) e `validar` rejeita a tabela inteira se ele aparecer como fonte.

Classificacao atual nao e classificacao historica: linha confirmada sem
`validade_inicio` seria lida como "valeu desde sempre", e a planilha da B3 so
diz o setor de hoje. Setor muda quando a empresa se reposiciona, a holding
troca de atividade ou uma fusao redefine o negocio, entao a linha confirmada
precisa de inicio explicito ou de `estabilidade_documentada`. O fim pode ficar
aberto enquanto a classificacao vigora.

Os grupos de pares se formam por (setor, subsetor), exigidos na linha
confirmada. `segmento` fica informativo e pode ficar vazio; serve para robustez
depois, sem virar restricao agora.

Nada e inferido de retorno, correlacao ou P&L: a classificacao e uma restricao
economica declarada antes de olhar resultado, e deriva-la dos numeros
transformaria a defesa contra data snooping na fonte dele. Pelo mesmo motivo,
sucessao nao herda setor - renomeacao, fusao e conversao de classe nao propagam
classificacao, e cada codigo observado precisa da propria linha.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUNAS = (
    "ticker_observado",
    "emissor_id",
    "setor",
    "subsetor",
    "segmento",
    "fonte",
    "data_fonte",
    "evidencia",
    "estabilidade_documentada",
    "validade_inicio",
    "validade_fim",
    "confianca",
    "status_revisao",
)

STATUS_VALIDOS = frozenset({
    "confirmado",                    # fonte registrada, entra nos pares
    "pendente",                      # ainda nao pesquisado
    "evidencia_insuficiente",        # pesquisado, sem confirmar
})
CONFIANCAS_VALIDAS = frozenset({"alta", "media", "baixa", ""})
ESTABILIDADE_VALIDA = frozenset({"sim", "nao", ""})

FONTES_PROIBIDAS = ("yahoo", "yfinance", "yf")

# ISIN brasileiro: 2 letras de pais + 4 do emissor + 3 do tipo + 3.
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{4}[A-Z0-9]{3}[A-Z0-9]{3}$")
_EMISSOR = re.compile(r"^[A-Z0-9]{4}$")

CHAVE_GRUPO = ("setor", "subsetor")
CAMPO_INFORMATIVO = "segmento"


def emissor_do_isin(codisi: str | None) -> str | None:
    """
    Codigo de emissor do ISIN, ou None se o ISIN nao tiver a forma esperada.

    O None e o ponto: `CODISI[2:6]` sobre campo vazio ou truncado devolve uma
    string qualquer, que viraria emissor inventado agrupando empresas sem
    relacao nenhuma.
    """
    if not isinstance(codisi, str):
        return None
    s = codisi.strip().upper()
    return s[2:6] if _ISIN.match(s) else None


def auditar_emissores(painel: pd.DataFrame) -> pd.DataFrame:
    """
    Pontos de atencao na derivacao do emissor a partir do ISIN, por ticker.

    Sao alertas de revisao, nao defeitos confirmados.
    `nomes_divergentes_no_emissor` marca o mesmo codigo sob nomes diferentes, o
    que tanto pode ser colisao real quanto grafia - BIDI3 alterna entre "BANCO
    INTER" e "INTER BANCO".

    A heuristica fica simples de proposito: resolver nome societario e trabalho
    documental, nao de string matching.
    """
    g = painel.groupby("CODNEG")
    perfil = pd.DataFrame({
        "codisi": g["CODISI"].first(),
        "nome": g["NOMRES"].first(),
    })
    perfil["emissor_id"] = perfil["codisi"].map(emissor_do_isin)

    nomes_por_emissor = (perfil.dropna(subset=["emissor_id"])
                         .groupby("emissor_id")["nome"]
                         .agg(lambda s: sorted(set(s))))
    colisao = {e: n for e, n in nomes_por_emissor.items() if len(n) > 1}

    perfil["alerta"] = ""
    perfil.loc[perfil["emissor_id"].isna(), "alerta"] = "isin_ausente_ou_invalido"
    perfil.loc[perfil["emissor_id"].isin(colisao), "alerta"] = "nomes_divergentes_no_emissor"
    perfil["nomes_no_emissor"] = perfil["emissor_id"].map(
        lambda e: " | ".join(colisao.get(e, [])))
    return perfil[perfil["alerta"] != ""]


def tabela_vazia() -> pd.DataFrame:
    """Esqueleto com as colunas do schema, para semear o arquivo."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUNAS})


def carregar(caminho: Path | str) -> pd.DataFrame:
    """Le a tabela como texto e devolve com as colunas do schema garantidas."""
    df = pd.read_csv(caminho, dtype=str).fillna("")
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"setores_b3.csv sem as colunas: {faltando}")
    return df[list(COLUNAS)]


def _intervalo(linha) -> tuple[pd.Timestamp, pd.Timestamp]:
    ini = pd.to_datetime(linha["validade_inicio"], errors="coerce")
    fim = pd.to_datetime(linha["validade_fim"], errors="coerce")
    return (ini if pd.notna(ini) else pd.Timestamp.min,
            fim if pd.notna(fim) else pd.Timestamp.max)


def _dia_seguinte(t: pd.Timestamp) -> pd.Timestamp:
    """Somar um dia a Timestamp.max estoura; fim aberto ja e o fim de tudo."""
    limite = pd.Timestamp.max - pd.Timedelta(days=1)
    return pd.Timestamp.max if t >= limite else t + pd.Timedelta(days=1)


def _dia_anterior(t: pd.Timestamp) -> pd.Timestamp:
    limite = pd.Timestamp.min + pd.Timedelta(days=1)
    return pd.Timestamp.min if t <= limite else t - pd.Timedelta(days=1)


def validar(df: pd.DataFrame) -> list[str]:
    """
    Problemas que impedem o uso da tabela. Lista vazia significa aprovada.

    Devolve todos de uma vez em vez de parar no primeiro, porque quem for
    corrigir a planilha prefere ver a lista inteira.
    """
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        return [f"colunas ausentes: {faltando}"]

    erros: list[str] = []

    fonte = df["fonte"].astype(str).str.lower()
    proibida = df[fonte.str.contains("|".join(FONTES_PROIBIDAS), na=False)]
    if len(proibida):
        erros.append(
            f"{len(proibida)} linha(s) com fonte proibida "
            f"({sorted(proibida['ticker_observado'].unique())[:5]}): o setor do "
            "Yahoo nao e oficial, nao e point-in-time e nao cobre deslistado")

    for coluna, validos in (("status_revisao", STATUS_VALIDOS),
                            ("confianca", CONFIANCAS_VALIDAS),
                            ("estabilidade_documentada", ESTABILIDADE_VALIDA)):
        ruim = set(df[coluna]) - validos
        if ruim:
            erros.append(f"{coluna} invalido: {sorted(ruim)}")

    confirmados = df[df["status_revisao"] == "confirmado"]
    for coluna in (*CHAVE_GRUPO, "fonte", "data_fonte", "evidencia", "emissor_id"):
        vazio = confirmados[confirmados[coluna].str.strip() == ""]
        if len(vazio):
            erros.append(f"{len(vazio)} linha(s) 'confirmado' sem {coluna}: "
                         f"{sorted(vazio['ticker_observado'].unique())[:5]}")

    # Confirmado sem inicio de validade seria lido como "desde sempre", e a
    # planilha da B3 so diz o setor de hoje.
    sem_inicio = confirmados[
        (confirmados["validade_inicio"].str.strip() == "")
        & (confirmados["estabilidade_documentada"] != "sim")]
    if len(sem_inicio):
        erros.append(
            f"{len(sem_inicio)} linha(s) 'confirmado' sem validade_inicio e sem "
            f"estabilidade_documentada: "
            f"{sorted(sem_inicio['ticker_observado'].unique())[:5]} - "
            "classificacao atual nao vale retroativamente por omissao")

    for coluna in ("data_fonte", "validade_inicio", "validade_fim"):
        preenchido = df[df[coluna].str.strip() != ""]
        invalido = preenchido[pd.to_datetime(preenchido[coluna],
                                             errors="coerce").isna()]
        if len(invalido):
            erros.append(f"{len(invalido)} linha(s) com {coluna} ilegivel: "
                         f"{sorted(invalido['ticker_observado'].unique())[:5]}")

    # Com janelas sobrepostas, "qual setor vigia?" fica ambiguo e a resposta
    # silenciosa seria a primeira linha do arquivo.
    for ticker, g in df.groupby("ticker_observado"):
        if len(g) < 2:
            continue
        janelas = sorted(_intervalo(l) for _, l in g.iterrows())
        for (_, f1), (i2, _) in zip(janelas, janelas[1:]):
            if i2 <= f1:
                erros.append(f"{ticker}: janelas de validade se sobrepoem")
                break

    vazio = confirmados[confirmados["ticker_observado"].str.strip() == ""]
    if len(vazio):
        erros.append(f"{len(vazio)} linha(s) 'confirmado' sem ticker_observado")

    malformado = confirmados[
        (confirmados["emissor_id"].str.strip() != "")
        & ~confirmados["emissor_id"].str.match(_EMISSOR, na=False)]
    if len(malformado):
        erros.append(f"{len(malformado)} linha(s) 'confirmado' com emissor_id "
                     f"malformado: {sorted(malformado['emissor_id'].unique())[:5]}")

    invertido = []
    for _, linha in df.iterrows():
        ini = pd.to_datetime(linha["validade_inicio"], errors="coerce")
        fim = pd.to_datetime(linha["validade_fim"], errors="coerce")
        if pd.notna(ini) and pd.notna(fim) and ini > fim:
            invertido.append(linha["ticker_observado"])
    if invertido:
        erros.append(f"validade_inicio posterior a validade_fim: "
                     f"{sorted(set(invertido))[:5]}")

    # Classes simultaneas do mesmo emissor sao a mesma empresa e nao podem
    # divergir. Mas o emissor pode mudar de setor com o tempo, e comparar so por
    # emissor proibiria justamente a mudanca historica que a validade temporal
    # existe para representar - dai a comparacao ser par a par entre intervalos
    # que se sobrepoem.
    conf = confirmados[confirmados["emissor_id"].str.strip() != ""]
    for emissor, g in conf.groupby("emissor_id"):
        linhas = [(l["ticker_observado"], *_intervalo(l), l) for _, l in g.iterrows()]
        for i, (tk_a, ini_a, fim_a, la) in enumerate(linhas):
            for tk_b, ini_b, fim_b, lb in linhas[i + 1:]:
                if max(ini_a, ini_b) > min(fim_a, fim_b):
                    continue                       # nao se sobrepoem
                difere = [c for c in CHAVE_GRUPO if la[c] != lb[c]]
                # Segmento so diverge com os dois preenchidos: vazio quer dizer
                # "nao registrado", nao "diferente".
                seg_a = la[CAMPO_INFORMATIVO].strip()
                seg_b = lb[CAMPO_INFORMATIVO].strip()
                if seg_a and seg_b and seg_a != seg_b:
                    difere.append(CAMPO_INFORMATIVO)
                if difere:
                    erros.append(
                        f"emissor {emissor}: {tk_a} e {tk_b} sao simultaneos e "
                        f"divergem em {difere}")
    return erros


def setor_vigente(df: pd.DataFrame, ticker_observado: str, data) -> pd.Series | None:
    """
    Classificacao confirmada e vigente para o ticker naquela data, ou None.

    Linha pendente ou com evidencia insuficiente devolve None: ela existe para
    registrar o que falta pesquisar, nao para ser usada enquanto isso. Sem
    herdar de sucessor e sem cair para a linha mais proxima - sem linha
    confirmada vigente, a resposta e que nao se sabe.
    """
    data = pd.Timestamp(data)
    g = df[(df["ticker_observado"] == ticker_observado)
           & (df["status_revisao"] == "confirmado")]
    for _, linha in g.iterrows():
        ini, fim = _intervalo(linha)
        if ini <= data <= fim:
            return linha
    return None


def tickers_operaveis_em(df: pd.DataFrame, data) -> set[str]:
    """
    Quem pode entrar na formacao de pares naquela data.

    Operabilidade e propriedade da data, nao do ticker: papel confirmado com
    validade a partir de 2021 nao era classificavel em 2017, e trata-lo como se
    fosse usaria informacao posterior para montar universo antigo.
    """
    data = pd.Timestamp(data)
    conf = df[df["status_revisao"] == "confirmado"]
    return {t for t in conf["ticker_observado"].unique()
            if setor_vigente(df, t, data) is not None}


def lacunas(
    df: pd.DataFrame,
    periodos: pd.DataFrame,
) -> pd.DataFrame:
    """
    O que falta curar, por ticker e por intervalo.

    `periodos` traz uma linha por ticker necessario, com `ticker_observado`,
    `inicio` e `fim`: a janela em que ele precisa de setor.

    A saida e por intervalo porque reduzir o ticker a um status so esconderia o
    caso que mais importa, o papel confirmado em parte do periodo e descoberto
    no resto. Assim, uma linha confirmada de 2021 em diante deixa a lacuna de
    2015 a 2020 visivel.
    """
    linhas = []
    for _, p in periodos.iterrows():
        tk = p["ticker_observado"]
        ini, fim = pd.Timestamp(p["inicio"]), pd.Timestamp(p["fim"])
        conf = df[(df["ticker_observado"] == tk)
                  & (df["status_revisao"] == "confirmado")]

        if not len(conf):
            existe = (df["ticker_observado"] == tk).any()
            linhas.append({
                "ticker_observado": tk, "inicio": ini.date(), "fim": fim.date(),
                "situacao": "sem_confirmacao" if existe else "sem_linha"})
            continue

        cobertos = sorted(_intervalo(l) for _, l in conf.iterrows())
        cursor = ini
        for c_ini, c_fim in cobertos:
            if c_fim < cursor:
                continue
            if c_ini > cursor:
                linhas.append({
                    "ticker_observado": tk, "inicio": cursor.date(),
                    "fim": min(_dia_anterior(c_ini), fim).date(),
                    "situacao": "lacuna_temporal"})
            cursor = max(cursor, _dia_seguinte(c_fim))
            if cursor > fim:
                break
        if cursor <= fim:
            linhas.append({"ticker_observado": tk, "inicio": cursor.date(),
                           "fim": fim.date(), "situacao": "lacuna_temporal"})
    return pd.DataFrame(linhas, columns=["ticker_observado", "inicio", "fim",
                                         "situacao"])
