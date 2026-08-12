"""Regras temporais e agregacao de eventos da V2."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.backtest import Janela
from src.v2 import cvm_ipe
from src.v2.universo import selecionar_pares_top20


COLUNA_MOTIVO_DIAGNOSTICO = "motivo_diagnostico"
TIPO_APRESENTACAO_ORIGINAL = "AP - Apresentação"
SEM_SESSAO_POSTERIOR = "sem_sessao_posterior_no_calendario"
FORA_DAS_JANELAS = "fora_das_janelas_de_teste"
NAO_LIDER_TOP20 = "emissor_nao_e_lider_top20_na_janela"


@dataclass(frozen=True)
class ResultadoEventos:
    """Eventos aceitos e descartados."""

    eventos: pd.DataFrame
    diagnosticos: pd.DataFrame


def _exigir_colunas(df: pd.DataFrame, colunas: set[str], nome: str) -> None:
    faltantes = sorted(colunas - set(df.columns))
    if faltantes:
        raise ValueError(f"{nome}: colunas ausentes: {', '.join(faltantes)}")


def _data_sem_hora(valor: object, coluna: str) -> pd.Timestamp:
    if pd.isna(valor):
        raise ValueError(f"{coluna}: data ausente")
    try:
        data = pd.Timestamp(valor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{coluna}: data invalida: {valor!r}") from exc
    if pd.isna(data) or data.tz is not None or data != data.normalize():
        raise ValueError(f"{coluna}: data deve ser sem hora e timezone: {valor!r}")
    return data


def _texto_nao_vazio(serie: pd.Series, coluna: str) -> pd.Series:
    vazio = serie.isna() | serie.astype(str).str.strip().eq("")
    if vazio.any():
        raise ValueError(f"{coluna}: valor vazio em {int(vazio.sum())} linha(s)")
    return serie.astype(str).str.strip()


def selecionar_apresentacoes_originais(documentos: pd.DataFrame) -> pd.DataFrame:
    """Mantem o AP original; no IPE historico sua versao e vazia ou ``1``."""
    _exigir_colunas(
        documentos, {"Tipo_Apresentacao", "Versao"}, "documentos IPE"
    )

    tipo = documentos["Tipo_Apresentacao"].astype("string")
    versao = documentos["Versao"].fillna("").astype(str).str.strip()
    mascara = tipo.eq(TIPO_APRESENTACAO_ORIGINAL) & versao.isin({"", "1"})
    return documentos.loc[mascara].copy().reset_index(drop=True)


def mapear_sessoes_disponiveis(
    eventos: pd.DataFrame,
    calendario: Sequence[object] | pd.DatetimeIndex,
) -> ResultadoEventos:
    """Mapeia cada entrega ao primeiro pregao posterior e separa sem cobertura."""
    _exigir_colunas(eventos, {"Data_Entrega"}, "eventos")
    if cvm_ipe.COLUNA_SESSAO in eventos.columns:
        raise ValueError(
            f"eventos: coluna {cvm_ipe.COLUNA_SESSAO!r} ja existe"
        )

    base = eventos.copy()
    entregas = [
        _data_sem_hora(valor, "Data_Entrega") for valor in base["Data_Entrega"]
    ]
    sessoes: list[pd.Timestamp | pd.NaT] = []
    cobertos: list[bool] = []
    cache: dict[pd.Timestamp, pd.Timestamp | None] = {}

    for entrega in entregas:
        if entrega not in cache:
            try:
                cache[entrega] = cvm_ipe.proxima_sessao_b3(entrega, calendario)
            except ValueError as exc:
                # Fim do calendário é descarte esperado; calendário ruim não é.
                if not str(exc).startswith(
                    "calendario nao cobre sessao posterior"
                ):
                    raise
                cache[entrega] = None
        sessao = cache[entrega]
        cobertos.append(sessao is not None)
        sessoes.append(pd.NaT if sessao is None else sessao)

    base[cvm_ipe.COLUNA_SESSAO] = pd.to_datetime(sessoes)
    mascara = pd.Series(cobertos, index=base.index, dtype=bool)
    aceitos = base.loc[mascara].reset_index(drop=True)
    diagnosticos = base.loc[~mascara].copy()
    diagnosticos[COLUNA_MOTIVO_DIAGNOSTICO] = SEM_SESSAO_POSTERIOR
    diagnosticos = diagnosticos.reset_index(drop=True)
    return ResultadoEventos(aceitos, diagnosticos)


def _validar_janelas(janelas: Sequence[Janela]) -> list[Janela]:
    ordenadas = sorted(janelas, key=lambda janela: janela.teste_inicio)
    rotulos: set[str] = set()
    fim_anterior: pd.Timestamp | None = None
    for janela in ordenadas:
        inicio = _data_sem_hora(janela.teste_inicio, "teste_inicio")
        fim = _data_sem_hora(janela.teste_fim, "teste_fim")
        if inicio >= fim:
            raise ValueError("janelas: teste_inicio deve ser anterior a teste_fim")
        if fim_anterior is not None and inicio < fim_anterior:
            raise ValueError("janelas: intervalos de teste sobrepostos")
        if janela.rotulo in rotulos:
            raise ValueError(f"janelas: rotulo duplicado: {janela.rotulo}")
        rotulos.add(janela.rotulo)
        fim_anterior = fim
    return ordenadas


def atribuir_janelas_teste(
    eventos: pd.DataFrame,
    janelas: Sequence[Janela],
) -> ResultadoEventos:
    """Atribui a rede congelada apenas dentro de ``[inicio, fim)`` do teste."""
    _exigir_colunas(eventos, {cvm_ipe.COLUNA_SESSAO}, "eventos")
    if "janela" in eventos.columns:
        raise ValueError("eventos: coluna 'janela' ja existe")

    ordenadas = _validar_janelas(janelas)
    base = eventos.copy()
    sessoes = [
        _data_sem_hora(valor, cvm_ipe.COLUNA_SESSAO)
        for valor in base[cvm_ipe.COLUNA_SESSAO]
    ]
    rotulos: list[str | None] = []
    for sessao in sessoes:
        rotulo = next(
            (
                janela.rotulo
                for janela in ordenadas
                if janela.teste_inicio <= sessao < janela.teste_fim
            ),
            None,
        )
        rotulos.append(rotulo)

    base["janela"] = pd.Series(rotulos, index=base.index, dtype="string")
    mascara = base["janela"].notna()
    aceitos = base.loc[mascara].reset_index(drop=True)
    diagnosticos = base.loc[~mascara].copy()
    diagnosticos[COLUNA_MOTIVO_DIAGNOSTICO] = FORA_DAS_JANELAS
    diagnosticos = diagnosticos.reset_index(drop=True)
    return ResultadoEventos(aceitos, diagnosticos)


def filtrar_eventos_lideres_top20(
    eventos: pd.DataFrame,
    pares: pd.DataFrame,
) -> ResultadoEventos:
    """Separa eventos de emissores lideres top20 da mesma janela."""
    _exigir_colunas(eventos, {"janela", "emissor_id"}, "eventos")
    if "lider" in eventos.columns:
        raise ValueError("eventos: coluna 'lider' ja existe")

    pares20 = selecionar_pares_top20(pares)
    lideres = pares20[["janela", "emissor_lider", "lider"]].drop_duplicates()
    ambiguos = lideres.duplicated(["janela", "emissor_lider"], keep=False)
    if ambiguos.any():
        raise ValueError(
            "pares top20: emissor possui mais de um ticker lider na mesma janela"
        )
    lideres = lideres.rename(columns={"emissor_lider": "emissor_id"})

    base = eventos.copy()
    base["janela"] = _texto_nao_vazio(base["janela"], "janela")
    base["emissor_id"] = _texto_nao_vazio(base["emissor_id"], "emissor_id")
    base["_ordem_evento"] = range(len(base))
    associados = base.merge(
        lideres,
        on=["janela", "emissor_id"],
        how="left",
        validate="many_to_one",
        indicator="_membership_lider",
    )
    mascara = associados["_membership_lider"].eq("both")
    aceitos = (
        associados.loc[mascara]
        .sort_values("_ordem_evento")
        .drop(columns=["_ordem_evento", "_membership_lider"])
        .reset_index(drop=True)
    )
    diagnosticos = associados.loc[~mascara].copy()
    diagnosticos[COLUNA_MOTIVO_DIAGNOSTICO] = NAO_LIDER_TOP20
    diagnosticos = (
        diagnosticos.sort_values("_ordem_evento")
        .drop(columns=["_ordem_evento", "_membership_lider"])
        .reset_index(drop=True)
    )
    return ResultadoEventos(aceitos, diagnosticos)


def _sem_acentos(valor: str) -> str:
    normalizado = unicodedata.normalize("NFKD", valor)
    return "".join(letra for letra in normalizado if not unicodedata.combining(letra))


def _normalizar_classificacao(valor: object) -> int:
    if isinstance(valor, bool) or pd.isna(valor):
        raise ValueError(f"classificacao invalida: {valor!r}")
    if isinstance(valor, (int, float)):
        if valor in (-1, 0, 1):
            return int(valor)
        raise ValueError(f"classificacao invalida: {valor!r}")

    texto = _sem_acentos(str(valor).strip().lower())
    mapa = {
        "+1": 1,
        "1": 1,
        "positiva": 1,
        "positivo": 1,
        "positive": 1,
        "-1": -1,
        "negativa": -1,
        "negativo": -1,
        "negative": -1,
        "0": 0,
        "neutra": 0,
        "neutro": 0,
        "neutral": 0,
    }
    try:
        return mapa[texto]
    except KeyError as exc:
        raise ValueError(f"classificacao invalida: {valor!r}") from exc


def agregar_sinais_eventos(
    classificados: pd.DataFrame,
    *,
    coluna_classificacao: str = "classificacao",
    coluna_data: str = "Data_Entrega",
) -> pd.DataFrame:
    """Consolida os documentos de cada líder/data em um sinal."""
    _exigir_colunas(
        classificados,
        {"lider", coluna_data, coluna_classificacao},
        "eventos classificados",
    )
    if classificados.empty:
        chaves = (["janela"] if "janela" in classificados.columns else []) + [
            "lider",
            coluna_data,
        ]
        return pd.DataFrame(
            columns=chaves
            + [
                "n_documentos",
                "n_positivas",
                "n_negativas",
                "n_neutras",
                "sinal",
                "abstencao",
                "motivo",
            ]
        )

    base = classificados.copy()
    base["lider"] = _texto_nao_vazio(base["lider"], "lider")
    base[coluna_data] = [
        _data_sem_hora(valor, coluna_data) for valor in base[coluna_data]
    ]
    base["_classe"] = [
        _normalizar_classificacao(valor) for valor in base[coluna_classificacao]
    ]

    chaves = ["lider", coluna_data]
    if "janela" in base.columns:
        base["janela"] = _texto_nao_vazio(base["janela"], "janela")
        chaves.insert(0, "janela")

    contexto = [
        coluna
        for coluna in ("emissor_id", cvm_ipe.COLUNA_SESSAO)
        if coluna in base.columns and coluna not in chaves
    ]

    linhas: list[dict[str, object]] = []
    for chave, grupo in base.groupby(chaves, sort=True, dropna=False):
        valores_chave = chave if isinstance(chave, tuple) else (chave,)
        linha = dict(zip(chaves, valores_chave, strict=True))
        for coluna in contexto:
            valores = grupo[coluna].drop_duplicates()
            if len(valores) != 1 or pd.isna(valores.iloc[0]):
                raise ValueError(
                    f"eventos classificados: {coluna} deve ser único por líder-dia"
                )
            linha[coluna] = valores.iloc[0]
        positivas = int(grupo["_classe"].eq(1).sum())
        negativas = int(grupo["_classe"].eq(-1).sum())
        neutras = int(grupo["_classe"].eq(0).sum())

        if positivas and negativas:
            sinal, abstencao, motivo = 0, True, "conflito_positivo_negativo"
        elif positivas:
            sinal, abstencao, motivo = 1, False, "positivas_concordantes"
        elif negativas:
            sinal, abstencao, motivo = -1, False, "negativas_concordantes"
        else:
            sinal, abstencao, motivo = 0, True, "somente_neutras"

        linha.update(
            {
                "n_documentos": len(grupo),
                "n_positivas": positivas,
                "n_negativas": negativas,
                "n_neutras": neutras,
                "sinal": sinal,
                "abstencao": abstencao,
                "motivo": motivo,
            }
        )
        linhas.append(linha)

    return pd.DataFrame(linhas)
