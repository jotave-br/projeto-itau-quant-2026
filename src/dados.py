"""
Aquisicao e cache dos dados brutos.

Duas fontes, cada uma no que faz melhor. O COTAHIST da a existencia do
instrumento em cada data, o volume financeiro em reais, o numero de negocios e
os precos nao ajustados, e e point-in-time por construcao porque inclui quem
quebrou ou fechou capital. O yfinance da o preco ajustado por proventos e
desdobramentos. Nenhuma das duas sozinha resolve: o COTAHIST nao ajusta preco
(montar a serie de ajuste do zero nao cabe no prazo) e o yfinance nao tem quem
deixou de existir nem o volume oficial em reais.

Aqui o dado sai cru, completo e verificado. Quem escolhe universo e universo.py.

Os ZIP originais ficam imutaveis em data/raw/cotahist/, cada um com um
.meta.json ao lado registrando URL, data do download, tamanho, sha256 e versao
do layout. O parse vai para data/processed/cotahist/ em parquet.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import DadosConfig

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw" / "cotahist"
DIR_PROC = RAIZ / "data" / "processed" / "cotahist"

URL_COTAHIST = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Versao do layout usada por este parser, gravada no .meta.json de cada arquivo
# baixado: se a B3 mudar o formato, da para saber qual especificacao produziu
# cada cache.
LAYOUT_VERSAO = "COTAHIST-registro01-245col"
LARGURA_LINHA = 245

# Registro tipo 01 (cotacoes). Posicoes 0-indexadas, meio-abertas.
COLSPECS: list[tuple[int, int]] = [
    (0, 2), (2, 10), (10, 12), (12, 24), (24, 27), (27, 39), (39, 49),
    (49, 52), (52, 56), (56, 69), (69, 82), (82, 95), (95, 108), (108, 121),
    (121, 134), (134, 147), (147, 152), (152, 170), (170, 188), (188, 201),
    (201, 202), (202, 210), (210, 217), (217, 230), (230, 242), (242, 245),
]
NOMES: list[str] = [
    "TIPREG", "DATA", "CODBDI", "CODNEG", "TPMERC", "NOMRES", "ESPECI",
    "PRAZOT", "MODREF", "PREABE", "PREMAX", "PREMIN", "PREMED", "PREULT",
    "PREOFC", "PREOFV", "TOTNEG", "QUATOT", "VOLTOT", "PREEXE",
    "INDOPC", "DATVEN", "FATCOT", "PTOEXE", "CODISI", "DISMES",
]
# Campos com 2 casas decimais implicitas (o arquivo nao tem ponto decimal).
MONETARIOS = ["PREABE", "PREMAX", "PREMIN", "PREMED", "PREULT",
              "PREOFC", "PREOFV", "VOLTOT", "PREEXE"]
INTEIROS = ["TOTNEG", "QUATOT"]
TEXTO = ["CODNEG", "NOMRES", "ESPECI", "CODISI", "MODREF"]

# Trailer (TIPREG=99): posicao do total de registros, verificada empiricamente
# no arquivo de 2015. Conta todas as linhas, incluindo header e trailer.
TRAILER_TOTREG = (31, 42)

# ESPECI combina tipo, marcador de evento e segmento. Como os marcadores mudam
# ao longo da série, a classificação usa apenas o primeiro token.
TIPOS_ACEITOS = frozenset({
    "ON",                                    # ordinaria
    "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF",   # preferenciais e classes
    "UNT",                                   # unit
})
TIPOS_BLOQUEADOS = frozenset({
    "DRN", "DR1", "DR2", "DR3",   # BDR: recibo de acao estrangeira
    "DRE",                         # BDR de ETF estrangeiro
    "CI",                          # cota de fundo (ETF, FII)
    "IBO", "SML",                  # instrumentos de indice (IBOV11, SMLL11)
})
# A normalização transforma o rótulo truncado "SML)" em "SML". Tipos novos não
# são inferidos por prefixo: interrompem o pipeline para revisão explícita.

# O tipo do ISIN (posições 6:9) serve como checagem independente do ESPECI.
# Units podem aparecer como CDA ou UNT, portanto ambos são aceitos.
ISIN_ACOES = frozenset({"ACN", "CDA", "UNT"})

# Categorias que o ISIN identifica como nao sendo acao brasileira. Vem antes do
# ESPECI na classificacao porque a B3 escreve o rotulo do ESPECI de forma
# inconsistente justamente nos instrumentos de indice: o mesmo IBOV11 e "IBO" em
# 2015 e "IBO/" em 2025, e o SMLL11 e "SML)". O ISIN e estruturado e estavel -
# todos eles sao tipo IND em qualquer ano - e nesta familia e mais confiavel.
ISIN_NAO_ACOES = frozenset({
    "BDR",   # recibo de acao ou de ETF estrangeiro
    "CTF",   # cota de fundo (ETF, FII)
    "IND",   # instrumento de indice (IBOV11, SMLL11)
})


class ErroLayoutCotahist(RuntimeError):
    """O arquivo nao bate com o layout que este parser conhece."""


class TipoInstrumentoDesconhecido(RuntimeError):
    """
    Apareceu um tipo de papel que nao esta nem na allowlist nem na blocklist.

    Erro em vez de silencio: se a B3 criar uma categoria nova, o pipeline para e
    avisa, em vez de deixar instrumento nao classificado contaminar a amostra.
    """


@dataclass(frozen=True)
class ArquivoBaixado:
    caminho: Path
    meta: dict


def _sha256(caminho: Path, blocos: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for pedaco in iter(lambda: f.read(blocos), b""):
            h.update(pedaco)
    return h.hexdigest()


def baixar_cotahist_ano(ano: int, forcar: bool = False) -> ArquivoBaixado:
    """
    Baixa o COTAHIST anual e grava o ZIP original, sem modificacao.

    Ao lado do ZIP vai um .meta.json com URL, data do download, tamanho, sha256
    e versao do layout. Esses campos entram no manifesto da execucao e sao o que
    permite provar depois que duas rodadas usaram o mesmo dado.

    Idempotente: arquivo que ja existe nao e rebaixado (use forcar=True para
    ignorar o cache).
    """
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    destino = DIR_RAW / f"COTAHIST_A{ano}.ZIP"
    meta_path = destino.with_suffix(".meta.json")

    if destino.exists() and meta_path.exists() and not forcar:
        return ArquivoBaixado(destino, json.loads(meta_path.read_text("utf-8")))

    url = URL_COTAHIST.format(ano=ano)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            conteudo = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Falha ao baixar {url}: {e}") from e

    if conteudo[:2] != b"PK":
        raise ErroLayoutCotahist(
            f"{url} nao devolveu um ZIP (primeiros bytes: {conteudo[:8]!r})"
        )

    destino.write_bytes(conteudo)
    meta = {
        "ano": ano,
        "url": url,
        "baixado_em": datetime.now().isoformat(timespec="seconds"),
        "bytes": destino.stat().st_size,
        "sha256": _sha256(destino),
        "layout_versao": LAYOUT_VERSAO,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return ArquivoBaixado(destino, meta)


def validar_arquivo_bruto(caminho: Path) -> dict:
    """
    Verificacoes no texto cru, antes de qualquer conversao.

    Falha explicitamente em vez de deixar passar: arquivo com layout diferente
    do esperado produz numero errado em silencio, o pior tipo de defeito aqui.
    """
    with zipfile.ZipFile(caminho) as z:
        interno = z.namelist()[0]
        with z.open(interno) as f:
            linhas = f.read().decode("latin-1").splitlines()

    if not linhas:
        raise ErroLayoutCotahist(f"{caminho.name}: arquivo vazio")

    larguras = {len(ln) for ln in linhas}
    if larguras != {LARGURA_LINHA}:
        raise ErroLayoutCotahist(
            f"{caminho.name}: esperava linhas de {LARGURA_LINHA} caracteres, "
            f"encontrei larguras {sorted(larguras)}"
        )

    if linhas[0][:2] != "00":
        raise ErroLayoutCotahist(f"{caminho.name}: primeira linha nao e header 00")
    if linhas[-1][:2] != "99":
        raise ErroLayoutCotahist(f"{caminho.name}: ultima linha nao e trailer 99")

    tipos = {ln[:2] for ln in linhas}
    if not tipos <= {"00", "01", "99"}:
        raise ErroLayoutCotahist(
            f"{caminho.name}: tipos de registro inesperados: {sorted(tipos - {'00','01','99'})}"
        )

    # A B3 mudou a convenção do trailer em 2025: até
    # 2024 ele conta todas as linhas, incluindo header e trailer; de 2025 em
    # diante conta so os registros de cotacao. Verificado nos arquivos de
    # 2015/2020/2024 contra 2025/2026. Aceitamos as duas e registramos qual foi
    # usada - arquivo truncado de verdade nao bate com nenhuma das duas, entao a
    # checagem continua tendo dentes.
    ini, fim = TRAILER_TOTREG
    declarado = linhas[-1][ini:fim]
    if not declarado.strip().isdigit():
        raise ErroLayoutCotahist(f"{caminho.name}: total do trailer ilegivel: {declarado!r}")
    declarado_n = int(declarado)
    cotacoes = sum(1 for ln in linhas if ln[:2] == "01")

    if declarado_n == len(linhas):
        convencao = "todas_as_linhas"
    elif declarado_n == cotacoes:
        convencao = "somente_cotacoes"
    else:
        raise ErroLayoutCotahist(
            f"{caminho.name}: trailer declara {declarado_n:,}, que nao bate nem com "
            f"o total de linhas ({len(linhas):,}) nem com o numero de cotacoes "
            f"({cotacoes:,}). Arquivo possivelmente truncado."
        )

    return {
        "arquivo_interno": interno,
        "linhas_total": len(linhas),
        "cotacoes": cotacoes,
        "trailer_declarado": declarado_n,
        "trailer_convencao": convencao,
        "nome_arquivo_header": linhas[0][2:15].strip(),
        "data_geracao": linhas[0][23:31],
    }


def parsear_cotahist(caminho: Path) -> pd.DataFrame:
    """
    Le o COTAHIST de largura fixa e devolve DataFrame tipado.

    Aplica a escala decimal implicita - os campos monetarios vem sem ponto, com
    2 casas embutidas - e converte a data.
    """
    import io

    with zipfile.ZipFile(caminho) as z:
        with z.open(z.namelist()[0]) as f:
            bruto = io.BytesIO(f.read())

    df = pd.read_fwf(bruto, colspecs=COLSPECS, names=NOMES, dtype=str,
                     encoding="latin-1", header=None)
    df = df[df["TIPREG"] == "01"].copy()

    df["DATA"] = pd.to_datetime(df["DATA"], format="%Y%m%d")
    for c in MONETARIOS:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    for c in INTEIROS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TEXTO:
        df[c] = df[c].astype("string").str.strip()

    return adicionar_derivadas(df.reset_index(drop=True))


def adicionar_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta as colunas derivadas da classificacao de instrumento.

    Separada do parser de proposito: o parquet guarda so o resultado do parse e
    estas colunas sao recalculadas a cada leitura do cache. Com TIPO_PAPEL
    gravado, mudar a regra de classificacao invalidaria 12 anos de cache em
    silencio e o pipeline seguiria rodando com a regra antiga.

    Dado bruto e caro de obter e nunca muda; dado derivado e barato e muda com a
    metodologia. Nao devem morar juntos.
    """
    df = df.copy()
    df["TIPO_PAPEL"] = tipo_instrumento(df["ESPECI"])
    df["ISIN_TIPO"] = df["CODISI"].str.slice(6, 9)
    return df


def tipo_instrumento(especi: pd.Series) -> pd.Series:
    """
    Extrai e normaliza o tipo do papel a partir do campo composto ESPECI.

    Duas operacoes. O primeiro token separa o tipo do marcador de evento e do
    segmento ("ON      NM", "ON  ED  NM" e "UNT     N2" viram "ON", "ON" e
    "UNT"), porque os marcadores so aparecem no dia do evento e mudam o campo do
    mesmo ticker de um dia para o outro. Depois vem a remocao de pontuacao, que
    colapsa os rotulos truncados da B3 ("IBO" e "IBO/" para o mesmo IBOV11,
    "SML)" para o SMLL11) em vez de catalogar cada variante que a bolsa inventa.

    Nao ha colisao: os tipos observados entre 2015 e 2026 (ON, PN, PNA-PNF, UNT,
    DRN, DR1-DR3, DRE, CI, IBO, SML) continuam distintos depois da normalizacao.
    O ESPECI original fica preservado na coluna.
    """
    return (especi.fillna("")
            .str.split().str[0].fillna("")
            .str.replace(r"[^A-Za-z0-9]", "", regex=True)
            .str.upper()
            .astype("string"))


def validar_dataframe(df: pd.DataFrame, tolerancia_voltot: float = 0.01) -> dict:
    """
    Verificacoes de coerencia sobre os valores ja convertidos.

    So levanta erro no que e inequivocamente quebra de layout ou de escala.
    Anomalia pontual vira contagem no relatorio, porque dado de bolsa real tem
    caso de borda legitimo.
    """
    rel: dict = {"registros": len(df)}

    for c in ["VOLTOT", "TOTNEG", "QUATOT"]:
        neg = int((df[c] < 0).sum())
        if neg:
            raise ErroLayoutCotahist(f"{c} tem {neg:,} valores negativos")
        rel[f"{c}_negativos"] = 0

    precos_zerados = int((df["PREULT"] <= 0).sum())
    rel["preco_ultimo_nao_positivo"] = precos_zerados

    com_preco = df[df["PREULT"] > 0]
    incoerentes = com_preco[
        (com_preco["PREMIN"] > com_preco["PREMAX"])
        | (com_preco["PREULT"] < com_preco["PREMIN"])
        | (com_preco["PREULT"] > com_preco["PREMAX"])
        | (com_preco["PREMED"] < com_preco["PREMIN"])
        | (com_preco["PREMED"] > com_preco["PREMAX"])
    ]
    rel["precos_incoerentes"] = len(incoerentes)

    # Contratos a termo podem repetir a chave no arquivo completo. A unicidade
    # exigida é apenas por data e ticker dentro do pool de candidatos à vista.
    pool = df[
        df["TPMERC"].isin(("010",))
        & df["CODBDI"].isin(("02",))
        & df["TIPO_PAPEL"].isin(TIPOS_ACEITOS)
    ]
    dup_pool = int(pool.duplicated(subset=["DATA", "CODNEG"]).sum())
    rel["duplicatas_no_pool_a_vista"] = dup_pool
    rel["duplicatas_arquivo_inteiro"] = int(
        df.duplicated(subset=["DATA", "CODNEG", "TPMERC", "DISMES"]).sum()
    )
    if dup_pool:
        raise ErroLayoutCotahist(
            f"{dup_pool:,} duplicatas de (DATA, CODNEG) no mercado a vista/lote "
            "padrao. O painel de precos ficaria ambiguo."
        )

    # A tolerância de VOLTOT é avaliada no subconjunto líquido porque o PREMED,
    # com duas casas, produz grande erro relativo em papéis de centavos.
    amostra = df[(df["QUATOT"] > 0) & (df["PREMED"] > 0) & (df["VOLTOT"] > 0)]
    if len(amostra):
        erro = ((amostra["VOLTOT"] - amostra["QUATOT"] * amostra["PREMED"])
                / (amostra["QUATOT"] * amostra["PREMED"])).abs()
        rel["voltot_erro_mediano_geral"] = float(erro.median())

        liquidos = amostra[(amostra["TOTNEG"] >= 100) & (amostra["PREMED"] >= 1.0)]
        if len(liquidos):
            erro_liq = ((liquidos["VOLTOT"] - liquidos["QUATOT"] * liquidos["PREMED"])
                        / (liquidos["QUATOT"] * liquidos["PREMED"])).abs()
            rel["voltot_liquidos_n"] = int(len(liquidos))
            rel["voltot_liquidos_erro_mediano"] = float(erro_liq.median())
            rel["voltot_liquidos_erro_p99"] = float(erro_liq.quantile(0.99))
            rel["voltot_liquidos_acima_tolerancia_frac"] = float(
                (erro_liq > tolerancia_voltot).mean()
            )
            if erro_liq.median() > tolerancia_voltot:
                raise ErroLayoutCotahist(
                    "Escala do VOLTOT parece errada: erro relativo mediano de "
                    f"{erro_liq.median():.3%} contra QUATOT x PREMED nos papeis "
                    "liquidos"
                )

    # A faixa de preços é reportada separadamente para não confundir opções de
    # alto valor nominal com erros de escala no pool de candidatos.
    if len(com_preco):
        rel["preco_min_arquivo"] = float(com_preco["PREULT"].min())
        rel["preco_max_arquivo"] = float(com_preco["PREULT"].max())
    pool_precos = pool[pool["PREULT"] > 0]
    if len(pool_precos):
        rel["preco_min_pool"] = float(pool_precos["PREULT"].min())
        rel["preco_max_pool"] = float(pool_precos["PREULT"].max())

    return rel


def inventario_categorias(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Relatorio das categorias encontradas em CODBDI, TPMERC e TIPO_PAPEL.

    Se um ano futuro trouxer categoria nova, ela aparece aqui antes de virar
    problema silencioso.
    """
    saida = {}
    for col in ["CODBDI", "TPMERC", "TIPO_PAPEL"]:
        t = (df.groupby(col, dropna=False)
             .agg(registros=("CODNEG", "size"), tickers=("CODNEG", "nunique"))
             .sort_values("tickers", ascending=False))
        t["exemplos"] = [
            ", ".join(sorted(df.loc[df[col] == v, "CODNEG"].dropna().unique())[:4])
            for v in t.index
        ]
        saida[col] = t
    return saida


def filtrar_acoes_a_vista(
    df: pd.DataFrame,
    tpmerc: tuple[str, ...] = ("010",),
    codbdi: tuple[str, ...] = ("02",),
) -> pd.DataFrame:
    """
    Restringe aos instrumentos que sao acoes brasileiras negociadas a vista.

    Tres filtros, todos por campo oficial e nunca por padrao de ticker:

      TPMERC == 010    mercado a vista (exclui termo, opcoes, futuro)
      CODBDI == 02     lote padrao (exclui fracionario, ETF, FII, leilao)
      TIPO_PAPEL       allowlist explicita de acoes e units

    O terceiro nao e redundante: BDR e negociado em lote padrao e passa inteiro
    pelo CODBDI - em 2015 sao ~90 dos 555 tickers, 16% do pool. BDR acompanha
    mecanicamente o fechamento da bolsa de origem e quase nao negocia aqui,
    entao como seguidora produziria um lead-lag enorme e falso.

    Filtrar por sufixo de ticker seria errado: units terminam em 11 (SANB11,
    TAEE11, ALUP11) e preferenciais de classe E e F tambem (BRGE11, BRGE12).
    """
    # A checagem de tipo desconhecido roda so depois deste recorte: opcoes e
    # termo tem especificacoes proprias, fora do pool de candidatos, e varre-las
    # levantaria erro a toa.
    candidatos = df[df["TPMERC"].isin(tpmerc) & df["CODBDI"].isin(codbdi)]

    # O ISIN tem precedência porque o ESPECI é inconsistente para instrumentos
    # de índice. Combinações desconhecidas interrompem a classificação.
    nao_acao_por_isin = candidatos["ISIN_TIPO"].isin(ISIN_NAO_ACOES)

    fora = candidatos[
        ~nao_acao_por_isin
        & ~candidatos["TIPO_PAPEL"].isin(TIPOS_ACEITOS | TIPOS_BLOQUEADOS)
    ]
    if len(fora):
        partes = []
        for tp, g in fora.groupby("TIPO_PAPEL"):
            isin = g["ISIN_TIPO"].value_counts().head(3).to_dict()
            partes.append(
                f"  {tp!r}: {g['CODNEG'].nunique()} tickers, "
                f"ISIN diz {isin}, mediana de {g['TOTNEG'].median():.0f} negocios/dia, "
                f"exemplos {sorted(g['CODNEG'].dropna().unique())[:5]}"
            )
        raise TipoInstrumentoDesconhecido(
            "Tipos de papel nao classificados encontrados:\n"
            + "\n".join(partes)
            + "\nClassifique em TIPOS_ACEITOS ou TIPOS_BLOQUEADOS (src/dados.py) "
            "antes de seguir. ISIN_TIPO 'BDR' ou 'CTF' indica instrumento que "
            "NAO e acao brasileira."
        )

    # As duas fontes precisam concordar: ESPECI na allowlist e ISIN confirmando
    # que e acao. Na duvida o instrumento fica de fora - melhor perder um papel
    # do que contaminar a amostra com algo mal classificado.
    return candidatos[
        candidatos["TIPO_PAPEL"].isin(TIPOS_ACEITOS)
        & candidatos["ISIN_TIPO"].isin(ISIN_ACOES)
    ].copy()


def conferir_especi_contra_isin(df: pd.DataFrame) -> pd.DataFrame:
    """
    O tipo declarado no ESPECI bate com o codificado no ISIN?

    Devolve as linhas em que os dois discordam. Idealmente vazio; divergencia
    merece olhada antes de confiar no filtro.
    """
    acao_por_especi = df["TIPO_PAPEL"].isin(TIPOS_ACEITOS)
    acao_por_isin = df["ISIN_TIPO"].isin(ISIN_ACOES)
    return df.loc[acao_por_especi != acao_por_isin,
                  ["DATA", "CODNEG", "ESPECI", "TIPO_PAPEL", "CODISI", "ISIN_TIPO"]]


def carregar_ano(ano: int, usar_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Baixa (ou reusa o cache), valida, parseia e grava o parquet do ano.

    Devolve o DataFrame completo, sem filtro de instrumento, e o relatorio de
    validacao. Filtrar fica com quem chama, para a auditoria enxergar tambem o
    que foi descartado.
    """
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    parquet = DIR_PROC / f"COTAHIST_A{ano}.parquet"
    relatorio_path = DIR_PROC / f"COTAHIST_A{ano}.validacao.json"

    if usar_cache and parquet.exists() and relatorio_path.exists():
        # Derivadas recalculadas em vez de lidas do disco, para mudanca na regra
        # de classificacao valer sem reprocessar os arquivos.
        return (adicionar_derivadas(pd.read_parquet(parquet)),
                json.loads(relatorio_path.read_text("utf-8")))

    baixado = baixar_cotahist_ano(ano)
    rel_bruto = validar_arquivo_bruto(baixado.caminho)
    df = parsear_cotahist(baixado.caminho)
    rel_valores = validar_dataframe(df)

    if rel_bruto["cotacoes"] != len(df):
        raise ErroLayoutCotahist(
            f"Contagem divergente: {rel_bruto['cotacoes']:,} registros tipo 01 no "
            f"texto contra {len(df):,} linhas parseadas"
        )

    relatorio = {"ano": ano, "meta_download": baixado.meta,
                 "arquivo": rel_bruto, "valores": rel_valores}
    # So as colunas do parse vao para o disco; as derivadas sao recalculadas na
    # leitura (ver adicionar_derivadas).
    df.drop(columns=["TIPO_PAPEL", "ISIN_TIPO"]).to_parquet(parquet, index=False)
    relatorio_path.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=str), "utf-8"
    )
    return df, relatorio


DIR_YF = RAIZ / "data" / "raw" / "yfinance"

# Parametros do yfinance declarados na mao: os defaults da biblioteca mudam
# entre versoes, e o resultado nao pode depender de qual estava instalada no dia.
#
#   auto_adjust=False  mantem Close bruto e Adj Close separados. Usamos os dois:
#                      o bruto para conferir contra o PREULT, o ajustado para
#                      calcular retorno.
#   actions=True       traz proventos e desdobramentos, que explicam a diferenca
#                      entre os dois precos.
#   repair=False       desliga a reparacao heuristica do yfinance, que mascara
#                      problema de dado em silencio.
#   keepna=True        mantem as linhas sem dado, para a ausencia continuar
#                      visivel em vez de sumir do indice.
YF_PARAMS = {
    "auto_adjust": False,
    "actions": True,
    "repair": False,
    "keepna": True,
}
YF_TIMEOUT = 30           # segundos por requisicao, explicito
YF_TAMANHO_LOTE = 40      # tickers por chamada
YF_THREADS = 2            # limitado de proposito, para nao levar rate limit
YF_MAX_TENTATIVAS = 3
YF_ESPERA_BASE = 2.0      # segundos; dobra a cada tentativa


def _fim_exclusivo(fim: date | None) -> str | None:
    """
    Converte a ultima data desejada no `end` que o yfinance espera.

    O `end` do yfinance e exclusivo: pedir 2026-07-27 devolve ate 2026-07-26 e
    perde o ultimo pregao em silencio. Somamos um dia para a data pedida entrar.
    """
    return None if fim is None else str(fim + timedelta(days=1))


def baixar_yfinance(
    ticker: str, inicio: date, fim: date | None = None, forcar: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Baixa a serie de um ticker no yfinance e guarda em cache com procedencia.

    Alem dos precos, guarda proventos, desdobramentos, o ticker como foi
    solicitado, a versao do yfinance, o timestamp do download e o sha256 do
    cache. Sem isso nao da para reproduzir nem auditar um numero depois.
    """
    import yfinance as yf

    DIR_YF.mkdir(parents=True, exist_ok=True)
    simbolo = f"{ticker}{DadosConfig().sufixo_yfinance}"
    destino = DIR_YF / f"{ticker}.parquet"
    meta_path = DIR_YF / f"{ticker}.meta.json"

    if destino.exists() and meta_path.exists() and not forcar:
        return (pd.read_parquet(destino),
                json.loads(meta_path.read_text("utf-8")))

    meta = {
        "ticker_solicitado": ticker,
        "simbolo_yfinance": simbolo,
        "yfinance_versao": yf.__version__,
        "baixado_em": datetime.now().isoformat(timespec="seconds"),
        "parametros": dict(YF_PARAMS),
        "inicio": str(inicio),
        "fim": str(fim) if fim else None,
    }

    try:
        df = yf.download(
            simbolo, start=str(inicio), end=_fim_exclusivo(fim),
            progress=False, threads=False, timeout=YF_TIMEOUT, **YF_PARAMS,
        )
    except Exception as e:  # noqa: BLE001 - qualquer falha vira status, nao crash
        meta.update({"status": "erro_download", "erro": f"{type(e).__name__}: {e}"})
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        return pd.DataFrame(), meta

    if df is None or df.empty:
        meta["status"] = "serie_ausente"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        return pd.DataFrame(), meta

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["ticker_solicitado"] = ticker

    df.to_parquet(destino, index=False)
    meta.update({
        "status": "ok",
        "linhas": int(len(df)),
        "primeira_data": str(df["Date"].min().date()),
        "ultima_data": str(df["Date"].max().date()),
        "colunas": list(df.columns),
        "sha256_cache": _sha256(destino),
        "bytes": destino.stat().st_size,
    })
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return df, meta


def _gravar_serie_yf(ticker: str, df: pd.DataFrame, meta: dict) -> None:
    """Grava a serie de um ticker no cache, com o meta ao lado."""
    destino = DIR_YF / f"{ticker}.parquet"
    df = df.copy()
    df["ticker_solicitado"] = ticker
    df.to_parquet(destino, index=False)
    meta.update({
        "status": "sucesso",
        "linhas": int(len(df)),
        "primeira_data": str(pd.to_datetime(df["Date"]).min().date()),
        "ultima_data": str(pd.to_datetime(df["Date"]).max().date()),
        "colunas": list(df.columns),
        "sha256_cache": _sha256(destino),
        "bytes": destino.stat().st_size,
    })
    (DIR_YF / f"{ticker}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")


def _cache_valido(ticker: str) -> bool:
    p, m = DIR_YF / f"{ticker}.parquet", DIR_YF / f"{ticker}.meta.json"
    if not m.exists():
        return False
    try:
        meta = json.loads(m.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    # serie_vazia e resultado legitimo: nao adianta rebaixar a cada run.
    if meta.get("status") == "serie_vazia":
        return True
    return meta.get("status") == "sucesso" and p.exists()


def baixar_yfinance_lote(
    tickers: list[str],
    inicio: date,
    fim: date | None = None,
    tamanho_lote: int = YF_TAMANHO_LOTE,
    max_tentativas: int = YF_MAX_TENTATIVAS,
    forcar: bool = False,
    log=None,
) -> pd.DataFrame:
    """
    Baixa muitos tickers em lotes deterministicos, com retomada e manifesto.

    Uma chamada unica com centenas de simbolos e fragil: uma falha derruba tudo
    e nao da para saber de quem foi, entao lotes de ~40 isolam o estrago. Ticker
    com cache valido nao e rebaixado, e reexecutar depois de queda de rede
    continua de onde parou. As retentativas esperam o dobro a cada vez e, da
    segunda em diante, sao individuais: vazio em chamada de lote costuma ser
    rate limit, e nao ausencia real.

    Todo ticker sai com estado documentado - sucesso, serie_vazia ou
    erro_persistente -, porque a etapa nao termina so porque a chamada retornou.
    """
    import time

    import yfinance as yf

    DIR_YF.mkdir(parents=True, exist_ok=True)
    sufixo = DadosConfig().sufixo_yfinance
    registros: list[dict] = []

    pendentes = [t for t in tickers if forcar or not _cache_valido(t)]
    ja_em_cache = [t for t in tickers if t not in pendentes]
    for t in ja_em_cache:
        meta = json.loads((DIR_YF / f"{t}.meta.json").read_text("utf-8"))
        registros.append({"ticker": t, "status": meta.get("status"),
                          "tentativas": 0, "linhas": meta.get("linhas", 0),
                          "origem": "cache", "erro": None})
    if log:
        log.info("yfinance: %d em cache, %d a baixar", len(ja_em_cache), len(pendentes))

    meta_base = {
        "yfinance_versao": yf.__version__,
        "parametros": dict(YF_PARAMS),
        "timeout": YF_TIMEOUT,
        "inicio": str(inicio),
        "fim_desejado": str(fim) if fim else None,
        "fim_exclusivo_enviado": _fim_exclusivo(fim),
    }

    faltando = list(pendentes)
    for tentativa in range(1, max_tentativas + 1):
        if not faltando:
            break
        lote = tamanho_lote if tentativa == 1 else 1
        if tentativa > 1:
            espera = YF_ESPERA_BASE * (2 ** (tentativa - 2))
            if log:
                log.info("  tentativa %d para %d ticker(s), aguardando %.0fs",
                         tentativa, len(faltando), espera)
            time.sleep(espera)

        ainda_faltando = []
        for i in range(0, len(faltando), lote):
            grupo = faltando[i:i + lote]
            simbolos = [f"{t}{sufixo}" for t in grupo]
            try:
                bruto = yf.download(
                    simbolos if len(simbolos) > 1 else simbolos[0],
                    start=str(inicio), end=_fim_exclusivo(fim),
                    progress=False, threads=(YF_THREADS if len(simbolos) > 1 else False),
                    timeout=YF_TIMEOUT, **YF_PARAMS,
                )
            except Exception as e:  # noqa: BLE001
                for t in grupo:
                    ainda_faltando.append(t)
                    if tentativa == max_tentativas:
                        registros.append({
                            "ticker": t, "status": "erro_persistente",
                            "tentativas": tentativa, "linhas": 0,
                            "origem": "download",
                            "erro": f"{type(e).__name__}: {e}"[:200]})
                continue

            for t, simbolo in zip(grupo, simbolos):
                try:
                    if isinstance(bruto.columns, pd.MultiIndex):
                        sub = bruto.xs(simbolo, axis=1, level=1, drop_level=True)
                    else:
                        sub = bruto
                    sub = sub.dropna(how="all").reset_index()
                    sub = sub.rename(columns={"index": "Date"})
                except (KeyError, ValueError):
                    sub = pd.DataFrame()

                if sub.empty:
                    if tentativa < max_tentativas:
                        ainda_faltando.append(t)
                    else:
                        meta = {**meta_base, "ticker_solicitado": t,
                                "simbolo_yfinance": simbolo,
                                "baixado_em": datetime.now().isoformat(timespec="seconds"),
                                "status": "serie_vazia"}
                        (DIR_YF / f"{t}.meta.json").write_text(
                            json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
                        registros.append({"ticker": t, "status": "serie_vazia",
                                          "tentativas": tentativa, "linhas": 0,
                                          "origem": "download", "erro": None})
                    continue

                meta = {**meta_base, "ticker_solicitado": t,
                        "simbolo_yfinance": simbolo,
                        "baixado_em": datetime.now().isoformat(timespec="seconds"),
                        "tentativas": tentativa}
                _gravar_serie_yf(t, sub, meta)
                registros.append({"ticker": t, "status": "sucesso",
                                  "tentativas": tentativa, "linhas": len(sub),
                                  "origem": "download", "erro": None})

            if log and tentativa == 1 and (i // max(lote, 1)) % 5 == 4:
                log.info("  lote %d/%d", i + len(grupo), len(faltando))

        faltando = ainda_faltando

    # Nenhum ticker pode sair daqui sem estado documentado.
    vistos = {r["ticker"] for r in registros}
    for t in tickers:
        if t not in vistos:
            registros.append({"ticker": t, "status": "erro_persistente",
                              "tentativas": max_tentativas, "linhas": 0,
                              "origem": "download",
                              "erro": "sem estado apos todas as tentativas"})
    return pd.DataFrame(registros).set_index("ticker").loc[tickers].reset_index()


def carregar_series_yf(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Le do cache as series que existem. Ticker sem cache simplesmente falta."""
    saida = {}
    for t in tickers:
        p = DIR_YF / f"{t}.parquet"
        if p.exists():
            saida[t] = pd.read_parquet(p)
    return saida


def anos_do_periodo(inicio: date, fim: date | None = None) -> list[int]:
    """Lista de anos a baixar para cobrir o periodo do estudo."""
    ultimo = (fim or date.today()).year
    return list(range(inicio.year, ultimo + 1))


def carregar_periodo(
    inicio: date,
    fim: date | None = None,
    somente_acoes: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Painel consolidado do periodo inteiro, com os relatorios de validacao.

    Com `somente_acoes=True` aplica o filtro de instrumento (a vista, lote
    padrao, allowlist de ESPECI confirmada pelo ISIN). Com False devolve tudo,
    que e o que a auditoria precisa para enxergar o descartado.
    """
    quadros, relatorios = [], []
    for ano in anos_do_periodo(inicio, fim):
        df, rel = carregar_ano(ano)
        relatorios.append(rel)
        quadros.append(filtrar_acoes_a_vista(df) if somente_acoes else df)

    painel = pd.concat(quadros, ignore_index=True)
    painel = painel[painel["DATA"] >= pd.Timestamp(inicio)]
    if fim is not None:
        painel = painel[painel["DATA"] <= pd.Timestamp(fim)]
    return painel.reset_index(drop=True), relatorios
