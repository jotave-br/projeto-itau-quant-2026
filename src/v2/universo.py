"""Leitura dos artefatos top 20 da V1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LIMITE_TOP20 = 20
CHAVES_ARESTA = ["janela", "lider", "seguidora"]
COLUNAS_UNIVERSO = {
    "janela", "CODNEG", "emissor_id", "posicao_final", "motivo_exclusao",
}
COLUNAS_PARES = {
    *CHAVES_ARESTA, "emissor_lider", "emissor_seguidora", "faixa_minima",
}
COLUNAS_REDE = {*CHAVES_ARESTA, "faixa_minima", "beta"}
COLUNAS_REDE_JUNCAO = (
    "beta", "lag", "direcao", "erro_padrao", "estat_t", "p_valor",
    "p_ajustado_bh", "aprovado_fdr", "alpha", "r2", "n",
    "data_inicio", "data_fim",
)


def _exigir_colunas(df: pd.DataFrame, colunas: set[str], nome: str) -> None:
    faltantes = sorted(colunas - set(df.columns))
    if faltantes:
        raise ValueError(f"{nome}: colunas ausentes: {', '.join(faltantes)}")


def _normalizar_texto(df: pd.DataFrame, colunas: list[str], nome: str) -> None:
    for coluna in colunas:
        vazio = df[coluna].isna() | df[coluna].astype(str).str.strip().eq("")
        if vazio.any():
            raise ValueError(f"{nome}: {coluna} vazio em {int(vazio.sum())} linha(s)")
        df[coluna] = df[coluna].astype(str).str.strip()


def _numerica(df: pd.DataFrame, coluna: str, nome: str) -> pd.Series:
    try:
        return pd.to_numeric(df[coluna], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nome}: {coluna} deve ser numerica") from exc


def _ler_csv(caminho: str | Path, nome: str) -> pd.DataFrame:
    path = Path(caminho)
    if not path.is_file():
        raise FileNotFoundError(f"{nome}: arquivo nao encontrado: {path}")
    return pd.read_csv(path)


def selecionar_universo_top20(universo: pd.DataFrame) -> pd.DataFrame:
    """Extrai e valida os 20 emissores elegiveis de cada janela da V1."""
    _exigir_colunas(universo, COLUNAS_UNIVERSO, "universo V1")
    if universo.empty:
        raise ValueError("universo V1: tabela vazia")

    base = universo.copy()
    _normalizar_texto(base, ["janela"], "universo V1")
    base["posicao_final"] = _numerica(base, "posicao_final", "universo V1")
    motivo = base["motivo_exclusao"].fillna("").astype(str).str.strip()
    mascara = (
        base["posicao_final"].notna()
        & base["posicao_final"].le(LIMITE_TOP20)
        & motivo.eq("")
    )
    top20 = base.loc[mascara].copy()
    _normalizar_texto(top20, ["CODNEG", "emissor_id"], "universo top20")

    janelas = sorted(base["janela"].unique())
    contagem = top20.groupby("janela").size().reindex(janelas, fill_value=0)
    ruins = contagem[contagem.ne(LIMITE_TOP20)]
    if not ruins.empty:
        detalhe = ", ".join(f"{janela}={int(n)}" for janela, n in ruins.items())
        raise ValueError(
            f"universo top20: esperado 20 ativos por janela; encontrado {detalhe}"
        )

    for janela, grupo in top20.groupby("janela", sort=True):
        if grupo["CODNEG"].nunique() != LIMITE_TOP20:
            raise ValueError(f"universo top20: ticker duplicado na janela {janela}")
        if grupo["emissor_id"].nunique() != LIMITE_TOP20:
            raise ValueError(
                f"universo top20: mais de uma classe do mesmo emissor na janela {janela}"
            )
        posicoes = set(grupo["posicao_final"].tolist())
        if posicoes != set(range(1, LIMITE_TOP20 + 1)):
            raise ValueError(
                f"universo top20: posicoes finais invalidas na janela {janela}"
            )

    return top20.sort_values(["janela", "posicao_final"]).reset_index(drop=True)


def carregar_universo_top20(caminho: str | Path) -> pd.DataFrame:
    """Le e valida ``universo_por_janela.csv`` da V1."""
    return selecionar_universo_top20(_ler_csv(caminho, "universo V1"))


def selecionar_pares_top20(pares: pd.DataFrame) -> pd.DataFrame:
    """Filtra pares cuja pior ponta estava entre as 20 mais liquidas."""
    _exigir_colunas(pares, COLUNAS_PARES, "pares V1")
    base = pares.copy()
    base["faixa_minima"] = _numerica(base, "faixa_minima", "pares V1")
    top20 = base.loc[base["faixa_minima"].le(LIMITE_TOP20)].copy()
    if top20.empty:
        return top20.reset_index(drop=True)

    _normalizar_texto(
        top20,
        CHAVES_ARESTA + ["emissor_lider", "emissor_seguidora"],
        "pares top20",
    )
    mesma_empresa = top20["emissor_lider"].eq(top20["emissor_seguidora"])
    if mesma_empresa.any():
        raise ValueError("pares top20: par com o mesmo emissor nas duas pontas")
    if top20.duplicated(CHAVES_ARESTA).any():
        raise ValueError("pares top20: aresta duplicada na mesma janela")
    return top20.sort_values(CHAVES_ARESTA).reset_index(drop=True)


def carregar_pares_top20(caminho: str | Path) -> pd.DataFrame:
    """Le e filtra ``pares_por_janela.csv`` da V1."""
    return selecionar_pares_top20(_ler_csv(caminho, "pares V1"))


def selecionar_rede_top20(rede: pd.DataFrame) -> pd.DataFrame:
    """Filtra a rede por liquidez, sem selecionar beta, sinal ou p-valor."""
    _exigir_colunas(rede, COLUNAS_REDE, "rede V1")
    base = rede.copy()
    base["faixa_minima"] = _numerica(base, "faixa_minima", "rede V1")
    top20 = base.loc[base["faixa_minima"].le(LIMITE_TOP20)].copy()
    if top20.empty:
        return top20.reset_index(drop=True)

    _normalizar_texto(top20, CHAVES_ARESTA, "rede top20")
    top20["beta"] = _numerica(top20, "beta", "rede top20")
    if top20["beta"].isna().any():
        raise ValueError("rede top20: beta ausente")
    if top20.duplicated(CHAVES_ARESTA).any():
        raise ValueError(
            "rede top20: mais de um beta por aresta; use a rede do lag principal"
        )
    return top20.sort_values(CHAVES_ARESTA).reset_index(drop=True)


def carregar_rede_top20(caminho: str | Path) -> pd.DataFrame:
    """Le e filtra ``rede_por_janela.csv`` da V1."""
    return selecionar_rede_top20(_ler_csv(caminho, "rede V1"))


def juntar_pares_com_betas_top20(
    pares: pd.DataFrame,
    rede: pd.DataFrame,
) -> pd.DataFrame:
    """Acopla a cada par seu beta congelado na mesma janela de treino."""
    pares20 = selecionar_pares_top20(pares)
    rede20 = selecionar_rede_top20(rede)
    if "beta" in pares20.columns:
        raise ValueError("pares V1: coluna beta inesperada; beta deve vir da rede")

    chaves_pares = set(map(tuple, pares20[CHAVES_ARESTA].itertuples(index=False)))
    chaves_rede = set(map(tuple, rede20[CHAVES_ARESTA].itertuples(index=False)))
    faltantes = sorted(chaves_pares - chaves_rede)
    extras = sorted(chaves_rede - chaves_pares)
    if faltantes or extras:
        partes = []
        if faltantes:
            partes.append(f"{len(faltantes)} par(es) sem beta")
        if extras:
            partes.append(f"{len(extras)} beta(s) sem par")
        raise ValueError("pares/rede top20 incompatíveis: " + "; ".join(partes))

    colunas_rede = CHAVES_ARESTA + [
        coluna for coluna in COLUNAS_REDE_JUNCAO if coluna in rede20.columns
    ]
    return pares20.merge(
        rede20[colunas_rede],
        on=CHAVES_ARESTA,
        how="left",
        validate="one_to_one",
    ).sort_values(CHAVES_ARESTA).reset_index(drop=True)


def carregar_pares_com_betas_top20(
    caminho_pares: str | Path,
    caminho_rede: str | Path,
) -> pd.DataFrame:
    """Le os dois artefatos e devolve a rede top20 pronta para a V2."""
    pares = _ler_csv(caminho_pares, "pares V1")
    rede = _ler_csv(caminho_rede, "rede V1")
    return juntar_pares_com_betas_top20(pares, rede)


def mapear_eventos_por_emissor_e_janela(
    eventos: pd.DataFrame,
    universo: pd.DataFrame,
    *,
    exigir_correspondencia: bool = False,
) -> pd.DataFrame:
    """Mapeia evento ao ticker PIT; emissores fora da top20 sao descartados."""
    _exigir_colunas(eventos, {"janela", "emissor_id"}, "eventos V2")
    if "CODNEG" in eventos.columns:
        raise ValueError(
            "eventos V2: CODNEG nao deve vir do evento; use o ticker PIT da V1"
        )

    top20 = selecionar_universo_top20(universo)
    base = eventos.copy()
    _normalizar_texto(base, ["janela", "emissor_id"], "eventos V2")
    base["_ordem_evento"] = range(len(base))

    colunas_membership = ["janela", "emissor_id", "CODNEG", "posicao_final"]
    colunas_membership += [
        coluna for coluna in ("setor", "subsetor", "data_formacao")
        if coluna in top20.columns
    ]
    mapeados = base.merge(
        top20[colunas_membership],
        on=["janela", "emissor_id"],
        how="left",
        validate="many_to_one",
        indicator="_membership_v1",
    )
    sem_membership = mapeados["_membership_v1"].eq("left_only")
    if exigir_correspondencia and sem_membership.any():
        chaves = (
            mapeados.loc[sem_membership, ["janela", "emissor_id"]]
            .drop_duplicates()
            .astype(str)
            .agg("/".join, axis=1)
            .tolist()
        )
        amostra = ", ".join(chaves[:5])
        raise ValueError(
            f"eventos V2: {int(sem_membership.sum())} evento(s) fora da top20: {amostra}"
        )

    return (
        mapeados.loc[~sem_membership]
        .sort_values("_ordem_evento")
        .drop(columns=["_ordem_evento", "_membership_v1"])
        .reset_index(drop=True)
    )
