"""Backtest da estratégia de eventos da V2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import pandas as pd

from src.config import CustosConfig, EstrategiaConfig
from src.custos import custo_por_ponta_bps


HORIZONTES_VALIDOS = (1, 3, 5)

SEM_SEGUIDORAS = "lider_sem_seguidoras_estruturais"
SESSAO_FORA_CALENDARIO = "sessao_disponivel_fora_do_calendario"
FIM_CALENDARIO = "holding_ultrapassa_fim_do_calendario"
CONTRIBUICAO_LIQUIDA_ZERO = "contribuicao_liquida_zero_na_coorte"
PRECO_ENTRADA_AUSENTE = "preco_entrada_ausente_ou_suspensao"
PRECO_FECHAMENTO_AUSENTE = "preco_fechamento_ausente_ou_suspensao"

COLUNAS_DIAGNOSTICO = [
    "janela",
    "Sessao_Disponivel",
    "lider",
    "seguidora",
    "ids_sinal_origem",
    "lideres_origem",
    "n_sinais_origem",
    "motivo_diagnostico",
    "detalhe",
]

COLUNAS_OPERACAO = [
    "id_operacao",
    "janela",
    "Sessao_Disponivel",
    "seguidora",
    "ids_sinal_origem",
    "lideres_origem",
    "contribuicoes_origem",
    "n_sinais_origem",
    "direcao",
    "h",
    "contribuicao_liquida_pre_normalizacao",
    "exposicao_bruta_coorte_pre_normalizacao",
    "peso_normalizado",
    "peso_pos_teto",
    "exposicao_bruta_coorte_pos_teto",
    "peso",
    "data_entrada",
    "data_saida",
    "preco_entrada",
    "preco_saida",
    "pnl_abertura_fechamento",
    "pnl_fechamento_fechamento",
    "pnl_bruto",
    "custo_entrada",
    "custo_saida",
    "custo_aluguel",
    "pnl_liquido",
]

COLUNAS_PNL_OPERACAO_DIA = [
    "id_operacao",
    "janela",
    "ids_sinal_origem",
    "lideres_origem",
    "seguidora",
    "DATA",
    "passo_holding",
    "tipo_retorno",
    "preco_inicio",
    "preco_fim",
    "retorno_ativo",
    "retorno_incremental_entrada",
    "peso",
    "pnl_abertura_fechamento",
    "pnl_fechamento_fechamento",
    "pnl_bruto",
    "custo_entrada",
    "custo_saida",
    "custo_giro",
    "custo_aluguel",
    "pnl_liquido",
]

COLUNAS_PNL_DIARIO = [
    "pnl_abertura_fechamento",
    "pnl_fechamento_fechamento",
    "pnl_bruto",
    "custo_entrada",
    "custo_saida",
    "custo_giro",
    "custo_aluguel",
    "pnl_liquido",
    "n_operacoes_ativas",
    "exposicao_bruta",
    "exposicao_liquida",
]


@dataclass(frozen=True)
class ResultadoBacktestEventos:
    """Tabelas produzidas pelo backtest."""

    operacoes: pd.DataFrame
    pnl_operacao_dia: pd.DataFrame
    pnl_diario: pd.DataFrame
    posicoes: pd.DataFrame
    diagnosticos: pd.DataFrame
    resumo: dict[str, int | float]


def _exigir_colunas(df: pd.DataFrame, colunas: set[str], nome: str) -> None:
    faltantes = sorted(colunas - set(df.columns))
    if faltantes:
        raise ValueError(f"{nome}: colunas ausentes: {', '.join(faltantes)}")


def _calendario_valido(
    calendario: Sequence[object] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    try:
        cal = pd.DatetimeIndex(calendario)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("calendario invalido") from exc
    if cal.empty:
        raise ValueError("calendario vazio")
    if cal.tz is not None:
        raise ValueError("calendario deve ser sem timezone")
    if cal.hasnans:
        raise ValueError("calendario contem data ausente")
    if not cal.equals(cal.normalize()):
        raise ValueError("calendario deve conter datas sem hora")
    if cal.has_duplicates:
        raise ValueError("calendario contem datas duplicadas")
    if not cal.is_monotonic_increasing:
        raise ValueError("calendario deve estar em ordem crescente")
    return cal


def _datas_sem_hora(serie: pd.Series, nome: str) -> pd.Series:
    try:
        datas = pd.to_datetime(serie, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{nome}: data invalida") from exc
    if datas.isna().any():
        raise ValueError(f"{nome}: data ausente")
    try:
        timezone = datas.dt.tz
        normalizadas = datas.dt.normalize()
    except AttributeError as exc:
        raise ValueError(f"{nome}: datas invalidas ou com timezones misturados") from exc
    if timezone is not None:
        raise ValueError(f"{nome}: datas devem ser sem timezone")
    if not datas.eq(normalizadas).all():
        raise ValueError(f"{nome}: datas devem ser sem hora")
    return datas


def _texto_nao_vazio(serie: pd.Series, nome: str) -> pd.Series:
    vazio = serie.isna() | serie.astype(str).str.strip().eq("")
    if vazio.any():
        raise ValueError(f"{nome}: valor vazio em {int(vazio.sum())} linha(s)")
    return serie.astype(str).str.strip()


def _direcao(valor: object) -> int:
    if isinstance(valor, bool) or isinstance(valor, str) or pd.isna(valor):
        raise ValueError(f"direcao deve ser numerica e igual a -1 ou 1: {valor!r}")
    if not isinstance(valor, Real):
        raise ValueError(f"direcao deve ser numerica e igual a -1 ou 1: {valor!r}")
    numero = float(valor)
    if numero not in (-1.0, 1.0):
        raise ValueError(f"direcao deve ser igual a -1 ou 1: {valor!r}")
    return int(numero)


def _validar_sinais(sinais: pd.DataFrame) -> pd.DataFrame:
    requeridas = {"janela", "Sessao_Disponivel", "lider", "direcao"}
    _exigir_colunas(sinais, requeridas, "sinais")
    base = sinais.copy()
    base["janela"] = _texto_nao_vazio(base["janela"], "janela")
    base["lider"] = _texto_nao_vazio(base["lider"], "lider")
    base["Sessao_Disponivel"] = _datas_sem_hora(
        base["Sessao_Disponivel"], "Sessao_Disponivel"
    )
    base["direcao"] = [_direcao(valor) for valor in base["direcao"]]

    chaves = ["janela", "Sessao_Disponivel", "lider"]
    duplicados = base.duplicated(chaves, keep=False)
    if duplicados.any():
        raise ValueError(
            "sinais ja devem estar agregados: lider-dia duplicado em "
            f"{int(duplicados.sum())} linha(s)"
        )
    janelas_por_sessao = base.groupby("Sessao_Disponivel")["janela"].nunique()
    if janelas_por_sessao.gt(1).any():
        raise ValueError("uma Sessao_Disponivel nao pode pertencer a duas janelas")

    if "Data_Entrega" in base.columns:
        entregas = _datas_sem_hora(base["Data_Entrega"], "Data_Entrega")
        if entregas.ge(base["Sessao_Disponivel"]).any():
            raise ValueError(
                "Data_Entrega deve ser estritamente anterior a Sessao_Disponivel"
            )
        base["Data_Entrega"] = entregas

    ordem = ["Sessao_Disponivel", "janela", "lider"]
    base = base.sort_values(ordem, kind="stable").reset_index(drop=True)
    base["id_sinal"] = [f"sinal_{i:06d}" for i in range(1, len(base) + 1)]
    return base


def _validar_pares(pares: pd.DataFrame) -> pd.DataFrame:
    requeridas = {"janela", "lider", "seguidora"}
    _exigir_colunas(pares, requeridas, "pares")
    base = pares.copy()
    for coluna in ("janela", "lider", "seguidora"):
        base[coluna] = _texto_nao_vazio(base[coluna], coluna)
    chaves = ["janela", "lider", "seguidora"]
    return (
        base[chaves]
        .drop_duplicates()
        .sort_values(chaves, kind="stable")
        .reset_index(drop=True)
    )


def _validar_painel_precos(
    painel: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    nome: str,
) -> pd.DataFrame:
    if not isinstance(painel, pd.DataFrame):
        raise TypeError(f"{nome} deve ser DataFrame")
    base = painel.copy()
    try:
        indice = pd.DatetimeIndex(base.index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{nome}: indice de datas invalido") from exc
    if indice.tz is not None or indice.hasnans or not indice.equals(indice.normalize()):
        raise ValueError(f"{nome}: indice deve conter datas validas sem hora")
    if indice.has_duplicates:
        raise ValueError(f"{nome}: indice contem datas duplicadas")

    colunas = pd.Index([str(coluna).strip() for coluna in base.columns])
    if colunas.has_duplicates or any(not coluna for coluna in colunas):
        raise ValueError(f"{nome}: tickers vazios ou duplicados")
    base.index = indice
    base.columns = colunas
    base = base.apply(pd.to_numeric, errors="coerce")
    return base.reindex(calendario)


def _paineis_de_cotahist(
    cotahist: pd.DataFrame,
    calendario: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    colunas = ["DATA", "CODNEG", "PREABE", "PREULT"]
    _exigir_colunas(cotahist, set(colunas), "cotahist")
    base = cotahist[colunas].copy()
    base["DATA"] = _datas_sem_hora(base["DATA"], "DATA")
    base["CODNEG"] = _texto_nao_vazio(base["CODNEG"], "CODNEG")
    duplicados = base.duplicated(["DATA", "CODNEG"], keep=False)
    if duplicados.any():
        raise ValueError(
            "cotahist: mais de um preco por DATA/CODNEG em "
            f"{int(duplicados.sum())} linha(s)"
        )
    for coluna in ("PREABE", "PREULT"):
        base[coluna] = pd.to_numeric(base[coluna], errors="coerce")
    abertura = base.pivot(index="DATA", columns="CODNEG", values="PREABE")
    fechamento = base.pivot(index="DATA", columns="CODNEG", values="PREULT")
    return abertura.reindex(calendario), fechamento.reindex(calendario)


def _preparar_precos(
    calendario: pd.DatetimeIndex,
    cotahist: pd.DataFrame | None,
    painel_preabe: pd.DataFrame | None,
    painel_preult: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tem_cotahist = cotahist is not None
    tem_paineis = painel_preabe is not None or painel_preult is not None
    if tem_cotahist and tem_paineis:
        raise ValueError("informe cotahist ou os dois paineis, nao ambos")
    if tem_cotahist:
        return _paineis_de_cotahist(cotahist, calendario)
    if painel_preabe is None or painel_preult is None:
        raise ValueError("informe cotahist ou painel_preabe e painel_preult")
    return (
        _validar_painel_precos(painel_preabe, calendario, "painel_preabe"),
        _validar_painel_precos(painel_preult, calendario, "painel_preult"),
    )


def _preco_valido(valor: object) -> bool:
    if pd.isna(valor):
        return False
    numero = float(valor)
    return math.isfinite(numero) and numero > 0.0


def _diagnostico_sinal(
    sinal: pd.Series,
    motivo: str,
    detalhe: str,
) -> dict[str, object]:
    return {
        "janela": sinal["janela"],
        "Sessao_Disponivel": sinal["Sessao_Disponivel"],
        "lider": sinal["lider"],
        "seguidora": None,
        "ids_sinal_origem": [sinal["id_sinal"]],
        "lideres_origem": [sinal["lider"]],
        "n_sinais_origem": 1,
        "motivo_diagnostico": motivo,
        "detalhe": detalhe,
    }


def _diagnostico_coorte(
    *,
    janela: str,
    sessao: pd.Timestamp,
    seguidora: str,
    ids_sinal: list[str],
    lideres: list[str],
    motivo: str,
    detalhe: str,
) -> dict[str, object]:
    return {
        "janela": janela,
        "Sessao_Disponivel": sessao,
        "lider": lideres[0] if len(lideres) == 1 else None,
        "seguidora": seguidora,
        "ids_sinal_origem": list(ids_sinal),
        "lideres_origem": list(lideres),
        "n_sinais_origem": len(ids_sinal),
        "motivo_diagnostico": motivo,
        "detalhe": detalhe,
    }


def _tabelas_vazias(
    calendario: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    operacoes = pd.DataFrame(columns=COLUNAS_OPERACAO)
    fluxos = pd.DataFrame(columns=COLUNAS_PNL_OPERACAO_DIA)
    diario = pd.DataFrame(0.0, index=calendario, columns=COLUNAS_PNL_DIARIO)
    diario[["n_operacoes_ativas"]] = diario[["n_operacoes_ativas"]].astype(int)
    diario.index.name = "DATA"
    posicoes = pd.DataFrame(index=calendario)
    posicoes.index.name = "DATA"
    return operacoes, fluxos, diario, posicoes


def _origens(grupo: pd.DataFrame) -> tuple[list[str], list[str], list[float]]:
    ordenado = grupo.sort_values(["id_sinal", "lider"], kind="stable")
    return (
        ordenado["id_sinal"].tolist(),
        ordenado["lider"].tolist(),
        [float(valor) for valor in ordenado["contribuicao"]],
    )


def rodar_backtest_eventos(
    sinais: pd.DataFrame,
    pares: pd.DataFrame,
    calendario: Sequence[object] | pd.DatetimeIndex,
    *,
    cotahist: pd.DataFrame | None = None,
    painel_preabe: pd.DataFrame | None = None,
    painel_preult: pd.DataFrame | None = None,
    h: int | None = None,
    cfg_estrategia: EstrategiaConfig | None = None,
    cfg_custos: CustosConfig | None = None,
    taxa_aluguel_anual: float | None = None,
) -> ResultadoBacktestEventos:
    """Executa o backtest para H em {1, 3, 5} sem imputar preços."""
    cal = _calendario_valido(calendario)
    cfg_e = cfg_estrategia or EstrategiaConfig()
    cfg_c = cfg_custos or CustosConfig()
    horizonte = cfg_e.holding_dias if h is None else h
    if isinstance(horizonte, bool) or horizonte not in HORIZONTES_VALIDOS:
        raise ValueError(f"h deve ser um de {HORIZONTES_VALIDOS}")
    horizonte = int(horizonte)

    teto = float(cfg_e.peso_maximo_por_posicao)
    if not math.isfinite(teto) or not 0.0 < teto <= 1.0:
        raise ValueError("peso_maximo_por_posicao deve estar em (0, 1]")
    taxa_aluguel = (
        cfg_c.aluguel_cenario_base
        if taxa_aluguel_anual is None
        else float(taxa_aluguel_anual)
    )
    if not math.isfinite(taxa_aluguel) or taxa_aluguel < 0.0:
        raise ValueError("taxa_aluguel_anual deve ser nao negativa")
    if cfg_c.dias_uteis_ano <= 0:
        raise ValueError("dias_uteis_ano deve ser positivo")

    sinais_ok = _validar_sinais(sinais)
    pares_ok = _validar_pares(pares)
    abertura, fechamento = _preparar_precos(
        cal, cotahist, painel_preabe, painel_preult
    )

    mapa_pares = {
        chave: tuple(grupo["seguidora"].sort_values(kind="stable"))
        for chave, grupo in pares_ok.groupby(["janela", "lider"], sort=True)
    }
    posicao_calendario = {data: i for i, data in enumerate(cal)}
    custo_ponta = custo_por_ponta_bps(cfg_c) / 1e4

    contribuicoes: list[dict[str, object]] = []
    linhas_diagnostico: list[dict[str, object]] = []

    # Os pesos são montados antes de consultar qualquer preço.
    for _, sinal in sinais_ok.iterrows():
        sessao = sinal["Sessao_Disponivel"]
        if sessao not in posicao_calendario:
            linhas_diagnostico.append(
                _diagnostico_sinal(
                    sinal,
                    SESSAO_FORA_CALENDARIO,
                    "Sessao_Disponivel nao pertence ao calendario recebido",
                )
            )
            continue
        inicio = posicao_calendario[sessao]
        if inicio + horizonte > len(cal):
            linhas_diagnostico.append(
                _diagnostico_sinal(
                    sinal,
                    FIM_CALENDARIO,
                    f"faltam {inicio + horizonte - len(cal)} sessao(oes) para H={horizonte}",
                )
            )
            continue
        seguidoras = mapa_pares.get((sinal["janela"], sinal["lider"]), ())
        if not seguidoras:
            linhas_diagnostico.append(
                _diagnostico_sinal(
                    sinal,
                    SEM_SEGUIDORAS,
                    "nenhum par estrutural para lider/janela",
                )
            )
            continue
        contribuicao = int(sinal["direcao"]) / len(seguidoras)
        for seguidora in seguidoras:
            contribuicoes.append(
                {
                    "janela": sinal["janela"],
                    "Sessao_Disponivel": sessao,
                    "seguidora": seguidora,
                    "id_sinal": sinal["id_sinal"],
                    "lider": sinal["lider"],
                    "contribuicao": contribuicao,
                }
            )

    candidatos: list[dict[str, object]] = []
    if contribuicoes:
        expandidas = pd.DataFrame(contribuicoes)
        chaves = ["Sessao_Disponivel", "janela", "seguidora"]
        for (sessao, janela, seguidora), grupo in expandidas.groupby(
            chaves, sort=True
        ):
            ids, lideres, parcelas = _origens(grupo)
            liquida = math.fsum(parcelas)
            if math.isclose(liquida, 0.0, rel_tol=0.0, abs_tol=1e-15):
                linhas_diagnostico.append(
                    _diagnostico_coorte(
                        janela=janela,
                        sessao=sessao,
                        seguidora=seguidora,
                        ids_sinal=ids,
                        lideres=lideres,
                        motivo=CONTRIBUICAO_LIQUIDA_ZERO,
                        detalhe=f"contribuicoes compensadas: {parcelas}",
                    )
                )
                continue
            candidatos.append(
                {
                    "janela": janela,
                    "Sessao_Disponivel": sessao,
                    "seguidora": seguidora,
                    "ids_sinal_origem": ids,
                    "lideres_origem": lideres,
                    "contribuicoes_origem": parcelas,
                    "n_sinais_origem": len(ids),
                    "contribuicao_liquida_pre_normalizacao": liquida,
                }
            )

    linhas_operacoes: list[dict[str, object]] = []
    linhas_fluxos: list[dict[str, object]] = []

    # Preço faltante deixa a safra subinvestida; não redistribuímos o peso.
    if candidatos:
        tabela_candidatos = pd.DataFrame(candidatos)
        for (sessao, janela), grupo in tabela_candidatos.groupby(
            ["Sessao_Disponivel", "janela"], sort=True
        ):
            grupo = grupo.sort_values("seguidora", kind="stable")
            exposicao_pre = math.fsum(
                abs(float(valor))
                for valor in grupo["contribuicao_liquida_pre_normalizacao"]
            )
            preparados: list[tuple[pd.Series, float, float]] = []
            for _, candidato in grupo.iterrows():
                normalizado = (
                    float(candidato["contribuicao_liquida_pre_normalizacao"])
                    / exposicao_pre
                )
                pos_teto = math.copysign(min(abs(normalizado), teto), normalizado)
                preparados.append((candidato, normalizado, pos_teto))
            exposicao_pos_teto = math.fsum(
                abs(pos_teto) for _, _, pos_teto in preparados
            )

            inicio = posicao_calendario[sessao]
            datas_holding = cal[inicio : inicio + horizonte]
            for candidato, normalizado, pos_teto in preparados:
                seguidora = candidato["seguidora"]
                ids = list(candidato["ids_sinal_origem"])
                lideres = list(candidato["lideres_origem"])
                parcelas = list(candidato["contribuicoes_origem"])
                preco_entrada = (
                    abertura.at[sessao, seguidora]
                    if seguidora in abertura.columns
                    else pd.NA
                )
                if not _preco_valido(preco_entrada):
                    linhas_diagnostico.append(
                        _diagnostico_coorte(
                            janela=janela,
                            sessao=sessao,
                            seguidora=seguidora,
                            ids_sinal=ids,
                            lideres=lideres,
                            motivo=PRECO_ENTRADA_AUSENTE,
                            detalhe=f"PREABE invalido em {sessao.date()}",
                        )
                    )
                    continue

                if seguidora not in fechamento.columns:
                    fechamentos = pd.Series(index=datas_holding, dtype=float)
                else:
                    fechamentos = fechamento.loc[datas_holding, seguidora]
                invalidos = [
                    data
                    for data, valor in fechamentos.items()
                    if not _preco_valido(valor)
                ]
                if invalidos:
                    datas_txt = ", ".join(str(data.date()) for data in invalidos)
                    linhas_diagnostico.append(
                        _diagnostico_coorte(
                            janela=janela,
                            sessao=sessao,
                            seguidora=seguidora,
                            ids_sinal=ids,
                            lideres=lideres,
                            motivo=PRECO_FECHAMENTO_AUSENTE,
                            detalhe=f"PREULT invalido em: {datas_txt}",
                        )
                    )
                    continue

                peso = pos_teto / horizonte
                numero_operacao = len(linhas_operacoes) + 1
                id_operacao = f"op_{numero_operacao:08d}"
                custo_entrada_total = abs(peso) * custo_ponta
                custo_saida_total = abs(peso) * custo_ponta
                aluguel_dia = (
                    abs(peso) * taxa_aluguel / cfg_c.dias_uteis_ano
                    if peso < 0.0
                    else 0.0
                )

                pnl_open_close = 0.0
                pnl_close_close = 0.0
                for passo, data in enumerate(datas_holding, start=1):
                    preco_fim = float(fechamentos.loc[data])
                    if passo == 1:
                        preco_inicio = float(preco_entrada)
                        tipo = "abertura_fechamento"
                    else:
                        preco_inicio = float(fechamentos.iloc[passo - 2])
                        tipo = "fechamento_fechamento"
                    retorno = preco_fim / preco_inicio - 1.0
                    # A operacao compra uma quantidade fixa na abertura e so
                    # gira novamente na saida. Portanto, o P&L de cada sessao
                    # deve ser medido contra o preco de entrada. Multiplicar o
                    # peso pelo retorno percentual de cada dia equivaleria a
                    # um rebalanceamento diario que nao existe (nem pagaria
                    # custos no modelo).
                    retorno_incremental_entrada = (
                        preco_fim - preco_inicio
                    ) / float(preco_entrada)
                    bruto = peso * retorno_incremental_entrada
                    parte_open = bruto if passo == 1 else 0.0
                    parte_close = bruto if passo > 1 else 0.0
                    custo_entrada_dia = custo_entrada_total if passo == 1 else 0.0
                    custo_saida_dia = (
                        custo_saida_total if passo == horizonte else 0.0
                    )
                    custo_giro = custo_entrada_dia + custo_saida_dia
                    liquido = bruto - custo_giro - aluguel_dia
                    linhas_fluxos.append(
                        {
                            "id_operacao": id_operacao,
                            "janela": janela,
                            "ids_sinal_origem": list(ids),
                            "lideres_origem": list(lideres),
                            "seguidora": seguidora,
                            "DATA": data,
                            "passo_holding": passo,
                            "tipo_retorno": tipo,
                            "preco_inicio": preco_inicio,
                            "preco_fim": preco_fim,
                            "retorno_ativo": retorno,
                            "retorno_incremental_entrada": (
                                retorno_incremental_entrada
                            ),
                            "peso": peso,
                            "pnl_abertura_fechamento": parte_open,
                            "pnl_fechamento_fechamento": parte_close,
                            "pnl_bruto": bruto,
                            "custo_entrada": custo_entrada_dia,
                            "custo_saida": custo_saida_dia,
                            "custo_giro": custo_giro,
                            "custo_aluguel": aluguel_dia,
                            "pnl_liquido": liquido,
                        }
                    )
                    pnl_open_close += parte_open
                    pnl_close_close += parte_close

                pnl_bruto = pnl_open_close + pnl_close_close
                custo_aluguel_total = aluguel_dia * horizonte
                linhas_operacoes.append(
                    {
                        "id_operacao": id_operacao,
                        "janela": janela,
                        "Sessao_Disponivel": sessao,
                        "seguidora": seguidora,
                        "ids_sinal_origem": ids,
                        "lideres_origem": lideres,
                        "contribuicoes_origem": parcelas,
                        "n_sinais_origem": len(ids),
                        "direcao": 1 if peso > 0.0 else -1,
                        "h": horizonte,
                        "contribuicao_liquida_pre_normalizacao": float(
                            candidato["contribuicao_liquida_pre_normalizacao"]
                        ),
                        "exposicao_bruta_coorte_pre_normalizacao": exposicao_pre,
                        "peso_normalizado": normalizado,
                        "peso_pos_teto": pos_teto,
                        "exposicao_bruta_coorte_pos_teto": exposicao_pos_teto,
                        "peso": peso,
                        "data_entrada": datas_holding[0],
                        "data_saida": datas_holding[-1],
                        "preco_entrada": float(preco_entrada),
                        "preco_saida": float(fechamentos.iloc[-1]),
                        "pnl_abertura_fechamento": pnl_open_close,
                        "pnl_fechamento_fechamento": pnl_close_close,
                        "pnl_bruto": pnl_bruto,
                        "custo_entrada": custo_entrada_total,
                        "custo_saida": custo_saida_total,
                        "custo_aluguel": custo_aluguel_total,
                        "pnl_liquido": (
                            pnl_bruto
                            - custo_entrada_total
                            - custo_saida_total
                            - custo_aluguel_total
                        ),
                    }
                )

    diagnosticos = pd.DataFrame(linhas_diagnostico, columns=COLUNAS_DIAGNOSTICO)
    if not linhas_operacoes:
        operacoes, fluxos, diario, posicoes = _tabelas_vazias(cal)
    else:
        operacoes = pd.DataFrame(linhas_operacoes, columns=COLUNAS_OPERACAO)
        fluxos = pd.DataFrame(
            linhas_fluxos, columns=COLUNAS_PNL_OPERACAO_DIA
        ).sort_values(["DATA", "id_operacao"], kind="stable").reset_index(drop=True)

        posicoes = (
            fluxos.pivot_table(
                index="DATA",
                columns="seguidora",
                values="peso",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(cal, fill_value=0.0)
            .sort_index(axis=1)
        )
        posicoes.index.name = "DATA"

        somaveis = [
            "pnl_abertura_fechamento",
            "pnl_fechamento_fechamento",
            "pnl_bruto",
            "custo_entrada",
            "custo_saida",
            "custo_giro",
            "custo_aluguel",
            "pnl_liquido",
        ]
        diario = fluxos.groupby("DATA", sort=True)[somaveis].sum().reindex(
            cal, fill_value=0.0
        )
        diario["n_operacoes_ativas"] = (
            fluxos.groupby("DATA")["id_operacao"].nunique().reindex(cal, fill_value=0)
        ).astype(int)
        diario["exposicao_bruta"] = posicoes.abs().sum(axis=1)
        diario["exposicao_liquida"] = posicoes.sum(axis=1)
        diario = diario[COLUNAS_PNL_DIARIO]
        diario.index.name = "DATA"

    ids_operados: set[str] = set()
    if not operacoes.empty:
        for ids in operacoes["ids_sinal_origem"]:
            ids_operados.update(ids)
    resumo: dict[str, int | float] = {
        "h": horizonte,
        "n_sinais": len(sinais_ok),
        "n_sinais_com_operacao": len(ids_operados),
        "n_coortes": int(sinais_ok["Sessao_Disponivel"].nunique()),
        "n_coortes_com_operacao": int(
            operacoes["Sessao_Disponivel"].nunique()
        )
        if not operacoes.empty
        else 0,
        "n_operacoes": len(operacoes),
        "n_diagnosticos": len(diagnosticos),
        "n_net_zero": int(
            diagnosticos["motivo_diagnostico"].eq(
                CONTRIBUICAO_LIQUIDA_ZERO
            ).sum()
        ),
        "pnl_bruto_total": float(diario["pnl_bruto"].sum()),
        "pnl_liquido_total": float(diario["pnl_liquido"].sum()),
        "custo_giro_total": float(diario["custo_giro"].sum()),
        "custo_aluguel_total": float(diario["custo_aluguel"].sum()),
        "exposicao_bruta_maxima": float(diario["exposicao_bruta"].max()),
    }
    return ResultadoBacktestEventos(
        operacoes=operacoes,
        pnl_operacao_dia=fluxos,
        pnl_diario=diario,
        posicoes=posicoes,
        diagnosticos=diagnosticos,
        resumo=resumo,
    )
