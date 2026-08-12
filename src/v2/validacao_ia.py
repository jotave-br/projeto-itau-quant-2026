"""Amostragem cega e validacao humana do classificador de eventos."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd


DIRECOES = ("positiva", "negativa", "neutra")
CLASSES_CONJUNTAS = (
    "nao_especifico",
    "especifico_positiva",
    "especifico_negativa",
    "especifico_neutra",
)
LIMIAR_MACRO_F1 = 0.70
LIMIAR_KAPPA = 0.60

# F1 de uma classe com punhado de documentos é ruído. O gate usa só as classes
# com suporte suficiente no gold; o suporte é medido no gold justamente para que
# uma IA que subdispara a classe não escape — nesse caso o suporte sobe e a
# classe volta a pesar. Ver docs/PROTOCOLO_ROTULAGEM_V2.md.
SUPORTE_MINIMO_CLASSE = 10
MIN_CLASSES_COM_SUPORTE = 2

_COLUNAS_IA = {
    "direcao",
    "especifico_empresa",
    "evidencia",
    "abster",
    "motivo_abstencao",
    "status_ia",
    "erro_ia",
    "confianca",
    "prompt_versao",
    "prompt_hash",
    "texto_hash",
    "request_key",
    "cache_key",
    "think_suportado",
    "ollama_versao",
    "modelo_nome",
    "modelo_digest",
    "modelo_quantizacao",
    "total_duration_ns",
    "load_duration_ns",
    "prompt_eval_count",
    "prompt_eval_duration_ns",
    "eval_count",
    "eval_duration_ns",
    "tempo_cliente_segundos",
    "classificado_em_utc",
}
_COLUNAS_MERCADO_EXATAS = {
    "preabe",
    "preult",
    "beta",
    "p_valor",
    "p_ajustado_bh",
    "aprovado_fdr",
    "sinal",
}
_MARCADORES_MERCADO = (
    "preco",
    "pnl",
    "p&l",
    "retorno",
    "drawdown",
    "sharpe",
)
_MARCADORES_IA = (
    "direcao",
    "especifico_empresa",
    "evidencia",
    "abster",
    "abstencao",
    "prompt",
    "modelo",
    "ollama",
    "confianca",
    "probabilidade",
    "logit",
    "classificacao",
)


@dataclass(frozen=True)
class ResultadoValidacaoIA:
    """Resultado da validação humana."""

    matriz_confusao: pd.DataFrame
    metricas_por_classe: pd.DataFrame
    macro_f1: float
    macro_f1_com_suporte: float
    classes_subdimensionadas: tuple[str, ...]
    suporte_minimo_classe: int
    cobertura: float
    taxa_abstencao: float
    n_total: int
    n_cobertos: int
    n_abstencoes: int
    kappa_avaliadores: float
    aprovado_macro_f1: bool
    aprovado_kappa: bool
    aprovado: bool


def _exigir_colunas(df: pd.DataFrame, colunas: set[str], nome: str) -> None:
    faltantes = sorted(colunas - set(df.columns))
    if faltantes:
        raise ValueError(f"{nome}: colunas ausentes: {', '.join(faltantes)}")


def _texto_sem_acento(valor: object) -> str:
    texto = unicodedata.normalize("NFKD", str(valor).strip().casefold())
    return "".join(
        caractere for caractere in texto if not unicodedata.combining(caractere)
    )


def _normalizar_ids(serie: pd.Series, nome: str) -> pd.Series:
    vazios = serie.isna() | serie.astype(str).str.strip().eq("")
    if vazios.any():
        raise ValueError(f"{nome}: ID vazio em {int(vazios.sum())} linha(s)")
    ids = serie.astype(str).str.strip()
    duplicados = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicados:
        raise ValueError(f"{nome}: ID duplicado: {duplicados[:3]}")
    return ids


def _direcoes_validas(serie: pd.Series, nome: str) -> pd.Series:
    if serie.isna().any():
        raise ValueError(f"{nome}: direcao ausente")
    direcoes = serie.astype(str).str.strip().str.casefold()
    invalidas = sorted(set(direcoes) - set(DIRECOES))
    if invalidas:
        raise ValueError(
            f"{nome}: direcao deve ser positiva, negativa ou neutra: {invalidas}"
        )
    return direcoes


def _coluna_cegada(coluna: object) -> bool:
    nome = _texto_sem_acento(coluna).replace(" ", "_").replace("-", "_")
    if nome in _COLUNAS_IA or nome in _COLUNAS_MERCADO_EXATAS:
        return True
    if nome.startswith("ia_") or nome.endswith("_ia"):
        return True
    return any(
        marcador in nome
        for marcador in (*_MARCADORES_IA, *_MARCADORES_MERCADO)
    )


def _alocar_estratos(disponiveis: dict[str, int], tamanho: int) -> dict[str, int]:
    base, resto = divmod(tamanho, len(DIRECOES))
    alocacao = {
        direcao: base + int(indice < resto)
        for indice, direcao in enumerate(DIRECOES)
    }
    insuficientes = {
        direcao: (disponiveis[direcao], alocacao[direcao])
        for direcao in DIRECOES
        if disponiveis[direcao] < alocacao[direcao]
    }
    if insuficientes:
        detalhe = ", ".join(
            f"{classe}={disponivel}/{necessario}"
            for classe, (disponivel, necessario) in insuficientes.items()
        )
        raise ValueError(f"direcoes insuficientes para amostra balanceada: {detalhe}")
    return alocacao


def selecionar_amostra_cega(
    classificacoes: pd.DataFrame,
    tamanho: int,
    seed: int,
    *,
    coluna_id: str = "ID_Documento",
    coluna_data: str = "Data_Entrega",
    coluna_emissor: str = "emissor_id",
    coluna_ano: str | None = None,
) -> pd.DataFrame:
    """Seleciona uma amostra balanceada sem expor o rótulo da IA."""
    if isinstance(tamanho, bool) or not isinstance(tamanho, Integral):
        raise TypeError("tamanho deve ser inteiro")
    if tamanho < 1:
        raise ValueError("tamanho deve ser positivo")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed deve ser inteiro")
    if seed < 0:
        raise ValueError("seed deve ser nao negativo")

    requeridas = {
        coluna_id,
        "direcao",
        "especifico_empresa",
        coluna_emissor,
    }
    requeridas.add(coluna_ano if coluna_ano is not None else coluna_data)
    _exigir_colunas(classificacoes, requeridas, "classificacoes IA")
    if tamanho > len(classificacoes):
        raise ValueError(
            f"tamanho solicitado ({tamanho}) excede classificacoes "
            f"({len(classificacoes)})"
        )

    base = classificacoes.copy()
    base[coluna_id] = _normalizar_ids(base[coluna_id], "classificacoes IA")
    base["direcao"] = _direcoes_validas(base["direcao"], "classificacoes IA")
    especificos = base["especifico_empresa"].map(
        lambda valor: isinstance(valor, (bool, np.bool_))
    )
    if not especificos.all():
        raise ValueError("classificacoes IA: especifico_empresa deve ser bool")
    base["especifico_empresa"] = base["especifico_empresa"].astype(bool)
    incoerentes = ~base["especifico_empresa"] & base["direcao"].ne("neutra")
    if incoerentes.any():
        raise ValueError(
            "classificacoes IA: evento nao especifico deve ter direcao neutra"
        )
    emissores_vazios = (
        base[coluna_emissor].isna()
        | base[coluna_emissor].astype(str).str.strip().eq("")
    )
    if emissores_vazios.any():
        raise ValueError("classificacoes IA: emissor vazio")
    base[coluna_emissor] = base[coluna_emissor].astype(str).str.strip()

    if coluna_ano is None:
        datas = pd.to_datetime(base[coluna_data], errors="coerce")
        if datas.isna().any():
            raise ValueError("classificacoes IA: Data_Entrega invalida")
        base["_ano_amostra"] = datas.dt.year.astype(int)
    else:
        anos = pd.to_numeric(base[coluna_ano], errors="coerce")
        if anos.isna().any() or (anos % 1).ne(0).any():
            raise ValueError("classificacoes IA: ano invalido")
        base["_ano_amostra"] = anos.astype(int)

    base = base.sort_values(coluna_id, kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    base["_desempate_amostra"] = rng.random(len(base))
    base["_estrato_amostra"] = base["direcao"]
    disponiveis = {
        direcao: int(base["_estrato_amostra"].eq(direcao).sum())
        for direcao in DIRECOES
    }
    alocacao = _alocar_estratos(disponiveis, int(tamanho))

    anos_usados: set[int] = set()
    emissores_usados: set[str] = set()
    selecionados: list[int] = []
    ordem_estratos = [str(valor) for valor in rng.permutation(DIRECOES)]
    for direcao in ordem_estratos:
        candidatos = base.index[base["_estrato_amostra"].eq(direcao)].tolist()
        for _ in range(alocacao[direcao]):
            melhor = max(
                candidatos,
                key=lambda indice: (
                    int(
                        direcao == "neutra"
                        and not bool(base.at[indice, "especifico_empresa"])
                    ),
                    int(base.at[indice, "_ano_amostra"] not in anos_usados)
                    + int(base.at[indice, coluna_emissor] not in emissores_usados),
                    -float(base.at[indice, "_desempate_amostra"]),
                    base.at[indice, coluna_id],
                ),
            )
            selecionados.append(melhor)
            anos_usados.add(int(base.at[melhor, "_ano_amostra"]))
            emissores_usados.add(str(base.at[melhor, coluna_emissor]))
            candidatos.remove(melhor)

    amostra = base.loc[selecionados].copy()
    amostra["_ordem_cega"] = rng.random(len(amostra))
    amostra = amostra.sort_values(
        ["_ordem_cega", coluna_id], kind="stable"
    ).reset_index(drop=True)
    remover = [coluna for coluna in amostra.columns if _coluna_cegada(coluna)]
    remover += [
        "_ano_amostra",
        "_desempate_amostra",
        "_estrato_amostra",
        "_ordem_cega",
    ]
    return amostra.drop(columns=remover, errors="ignore")


def _validar_rotulos_individuais(
    rotulos: pd.DataFrame,
    nome: str,
    coluna_id: str,
) -> pd.DataFrame:
    esperadas = {coluna_id, "especifico_empresa", "direcao"}
    encontradas = set(rotulos.columns)
    if encontradas != esperadas:
        faltantes = sorted(esperadas - encontradas)
        extras = sorted(encontradas - esperadas)
        raise ValueError(
            f"{nome}: colunas devem ser somente rotulos humanos; "
            f"faltantes={faltantes}, extras={extras}"
        )

    base = rotulos.copy()
    base[coluna_id] = _normalizar_ids(base[coluna_id], nome)
    booleanos = base["especifico_empresa"].map(
        lambda valor: isinstance(valor, (bool, np.bool_))
    )
    if not booleanos.all():
        raise ValueError(f"{nome}: especifico_empresa deve ser bool")
    base["especifico_empresa"] = base["especifico_empresa"].astype(bool)
    base["direcao"] = _direcoes_validas(base["direcao"], nome)
    incoerentes = ~base["especifico_empresa"] & base["direcao"].ne("neutra")
    if incoerentes.any():
        raise ValueError(f"{nome}: evento nao especifico deve ter direcao neutra")
    return base.sort_values(coluna_id, kind="stable").reset_index(drop=True)


def _classe_conjunta(especifico: bool, direcao: str) -> str:
    return f"especifico_{direcao}" if especifico else "nao_especifico"


def validar_rotulos_humanos(
    avaliador_a: pd.DataFrame,
    avaliador_b: pd.DataFrame,
    *,
    ids_esperados: Sequence[object] | None = None,
    coluna_id: str = "ID_Documento",
) -> pd.DataFrame:
    """Valida e alinha os rótulos dos dois avaliadores."""
    a = _validar_rotulos_individuais(avaliador_a, "avaliador A", coluna_id)
    b = _validar_rotulos_individuais(avaliador_b, "avaliador B", coluna_id)
    ids_a, ids_b = set(a[coluna_id]), set(b[coluna_id])
    if ids_a != ids_b:
        raise ValueError(
            "avaliadores devem rotular os mesmos IDs; "
            f"somente_A={sorted(ids_a - ids_b)[:3]}, "
            f"somente_B={sorted(ids_b - ids_a)[:3]}"
        )

    if ids_esperados is not None:
        if isinstance(ids_esperados, (str, bytes)):
            raise TypeError("ids_esperados deve ser uma sequencia de IDs")
        esperados = [str(valor).strip() for valor in ids_esperados]
        if any(not valor for valor in esperados) or len(set(esperados)) != len(
            esperados
        ):
            raise ValueError("ids_esperados contem vazio ou duplicata")
        if ids_a != set(esperados):
            raise ValueError("rotulos humanos nao cobrem exatamente ids_esperados")

    combinado = a.merge(
        b,
        on=coluna_id,
        how="inner",
        validate="one_to_one",
        suffixes=("_a", "_b"),
    )
    combinado["classe_a"] = [
        _classe_conjunta(especifico, direcao)
        for especifico, direcao in zip(
            combinado["especifico_empresa_a"], combinado["direcao_a"], strict=True
        )
    ]
    combinado["classe_b"] = [
        _classe_conjunta(especifico, direcao)
        for especifico, direcao in zip(
            combinado["especifico_empresa_b"], combinado["direcao_b"], strict=True
        )
    ]
    return combinado.sort_values(coluna_id, kind="stable").reset_index(drop=True)


def calcular_cohen_kappa(rotulos_validados: pd.DataFrame) -> float:
    """Calcula o kappa entre os dois avaliadores."""
    _exigir_colunas(
        rotulos_validados, {"classe_a", "classe_b"}, "rotulos validados"
    )
    if rotulos_validados.empty:
        raise ValueError("rotulos validados: tabela vazia")
    a = rotulos_validados["classe_a"].astype(str)
    b = rotulos_validados["classe_b"].astype(str)
    classes = sorted(set(a) | set(b))
    invalidas = sorted(set(classes) - set(CLASSES_CONJUNTAS))
    if invalidas or rotulos_validados[["classe_a", "classe_b"]].isna().any().any():
        raise ValueError(f"rotulos validados: classe conjunta invalida: {invalidas}")
    observado = float(a.eq(b).mean())
    esperado = sum(
        float(a.eq(classe).mean() * b.eq(classe).mean()) for classe in classes
    )
    if math.isclose(esperado, 1.0, abs_tol=1e-15):
        return float("nan")
    return (observado - esperado) / (1.0 - esperado)


def calcular_fleiss_kappa(votos: pd.DataFrame) -> float:
    """Kappa de Fleiss para um painel com mais de dois avaliadores.

    Recebe uma coluna por avaliador e uma linha por documento, com a classe
    conjunta em cada célula.
    """
    if votos.empty:
        raise ValueError("painel: tabela vazia")
    if votos.shape[1] < 2:
        raise ValueError("painel: sao necessarios ao menos dois avaliadores")
    if votos.isna().any().any():
        raise ValueError("painel: ha rotulo ausente")
    invalidas = sorted(set(votos.to_numpy().ravel()) - set(CLASSES_CONJUNTAS))
    if invalidas:
        raise ValueError(f"painel: classe conjunta invalida: {invalidas}")

    n_avaliadores = votos.shape[1]
    contagens = pd.DataFrame(
        {
            classe: votos.eq(classe).sum(axis=1)
            for classe in CLASSES_CONJUNTAS
        }
    )
    # Concordância dentro de cada documento, corrigida pelo par possível.
    por_documento = (
        contagens.pow(2).sum(axis=1) - n_avaliadores
    ) / (n_avaliadores * (n_avaliadores - 1))
    observado = float(por_documento.mean())
    proporcoes = contagens.sum() / (len(votos) * n_avaliadores)
    esperado = float(proporcoes.pow(2).sum())
    if math.isclose(esperado, 1.0, abs_tol=1e-15):
        return float("nan")
    return (observado - esperado) / (1.0 - esperado)


def consolidar_painel(
    votos: pd.DataFrame,
    *,
    desempate: str,
) -> pd.DataFrame:
    """Resolve o gold do painel por maioria, com desempate declarado.

    Uma classe só vence quando tem mais da metade dos votos. Sem maioria, vale
    o rótulo do avaliador de desempate.
    """
    if desempate not in votos.columns:
        raise ValueError(f"painel: avaliador de desempate ausente: {desempate}")
    if votos.empty:
        raise ValueError("painel: tabela vazia")
    invalidas = sorted(set(votos.to_numpy().ravel()) - set(CLASSES_CONJUNTAS))
    if invalidas:
        raise ValueError(f"painel: classe conjunta invalida: {invalidas}")

    n_avaliadores = votos.shape[1]
    minimo = n_avaliadores // 2 + 1
    linhas = []
    for indice, linha in votos.iterrows():
        contagem = linha.value_counts()
        vencedora = contagem.index[0]
        apoio = int(contagem.iloc[0])
        if apoio >= minimo:
            origem = "unanime" if apoio == n_avaliadores else "maioria"
            classe = str(vencedora)
        else:
            origem = "desempate"
            classe = str(linha[desempate])
            apoio = int(contagem.get(classe, 0))
        linhas.append(
            {
                "indice": indice,
                "classe_gold": classe,
                "origem_rotulo": origem,
                "votos_na_classe": apoio,
                "avaliadores": n_avaliadores,
            }
        )
    consolidado = pd.DataFrame(linhas).set_index("indice")
    consolidado.index.name = votos.index.name
    consolidado["especifico_empresa"] = consolidado["classe_gold"].ne(
        "nao_especifico"
    )
    consolidado["direcao"] = [
        classe.removeprefix("especifico_") if classe != "nao_especifico" else "neutra"
        for classe in consolidado["classe_gold"]
    ]
    return consolidado


def criar_tabela_divergencias(
    rotulos_validados: pd.DataFrame,
    *,
    coluna_id: str = "ID_Documento",
) -> pd.DataFrame:
    """Separa os desacordos que precisam de adjudicação."""
    colunas = [
        coluna_id,
        "especifico_empresa_a",
        "direcao_a",
        "classe_a",
        "especifico_empresa_b",
        "direcao_b",
        "classe_b",
    ]
    _exigir_colunas(rotulos_validados, set(colunas), "rotulos validados")
    divergencias = rotulos_validados.loc[
        rotulos_validados["classe_a"].ne(rotulos_validados["classe_b"]), colunas
    ].copy()
    divergencias["gold_especifico_empresa"] = pd.Series(
        pd.NA, index=divergencias.index, dtype="boolean"
    )
    divergencias["gold_direcao"] = pd.Series(
        pd.NA, index=divergencias.index, dtype="string"
    )
    return divergencias.reset_index(drop=True)


def _validar_predicoes_ia(
    predicoes: pd.DataFrame,
    coluna_id: str,
) -> pd.DataFrame:
    _exigir_colunas(
        predicoes,
        {coluna_id, "especifico_empresa", "direcao", "abster"},
        "predicoes IA",
    )
    base = predicoes[[coluna_id, "especifico_empresa", "direcao", "abster"]].copy()
    base[coluna_id] = _normalizar_ids(base[coluna_id], "predicoes IA")
    for coluna in ("especifico_empresa", "abster"):
        validos = base[coluna].map(
            lambda valor: isinstance(valor, (bool, np.bool_))
        )
        if not validos.all():
            raise ValueError(f"predicoes IA: {coluna} deve ser bool")
        base[coluna] = base[coluna].astype(bool)
    base["direcao"] = _direcoes_validas(base["direcao"], "predicoes IA")

    abstencao_invalida = base["abster"] & base["direcao"].ne("neutra")
    operacao_invalida = ~base["abster"] & (
        ~base["especifico_empresa"] | base["direcao"].eq("neutra")
    )
    nao_especifico_invalido = (
        ~base["especifico_empresa"] & base["direcao"].ne("neutra")
    )
    if (
        abstencao_invalida.any()
        or operacao_invalida.any()
        or nao_especifico_invalido.any()
    ):
        raise ValueError("predicoes IA: invariantes de abstencao violadas")
    return base


def avaliar_ia_contra_gold(
    predicoes_ia: pd.DataFrame,
    gold_adjudicado: pd.DataFrame,
    kappa_avaliadores: float,
    *,
    coluna_id: str = "ID_Documento",
) -> ResultadoValidacaoIA:
    """Compara a IA com os rótulos adjudicados."""
    if isinstance(kappa_avaliadores, bool) or not isinstance(
        kappa_avaliadores, Real
    ):
        raise TypeError("kappa_avaliadores deve ser numerico")
    kappa = float(kappa_avaliadores)
    if math.isinf(kappa) or (math.isfinite(kappa) and not -1.0 <= kappa <= 1.0):
        raise ValueError("kappa_avaliadores deve estar entre -1 e 1 ou ser NaN")

    ia = _validar_predicoes_ia(predicoes_ia, coluna_id)
    gold = _validar_rotulos_individuais(
        gold_adjudicado, "gold adjudicado", coluna_id
    )
    if ia.empty:
        raise ValueError("predicoes IA: tabela vazia")
    if set(ia[coluna_id]) != set(gold[coluna_id]):
        raise ValueError("predicoes IA e gold devem conter exatamente os mesmos IDs")

    # Abstenção entra na cobertura, não como uma quinta classe.
    ia["classe_ia"] = [
        _classe_conjunta(especifico, direcao)
        for especifico, direcao in zip(
            ia["especifico_empresa"], ia["direcao"], strict=True
        )
    ]
    gold["classe_gold"] = [
        _classe_conjunta(especifico, direcao)
        for especifico, direcao in zip(
            gold["especifico_empresa"], gold["direcao"], strict=True
        )
    ]
    comparacao = gold[[coluna_id, "classe_gold"]].merge(
        ia[[coluna_id, "classe_ia", "abster"]],
        on=coluna_id,
        how="inner",
        validate="one_to_one",
    )

    matriz = pd.crosstab(
        comparacao["classe_gold"], comparacao["classe_ia"], dropna=False
    ).reindex(
        index=CLASSES_CONJUNTAS,
        columns=CLASSES_CONJUNTAS,
        fill_value=0,
    )
    matriz = matriz.astype(int)
    matriz.index.name = "gold"
    matriz.columns.name = "ia"

    linhas_metricas: list[dict[str, int | float | str]] = []
    for classe in CLASSES_CONJUNTAS:
        tp = int(matriz.at[classe, classe])
        suporte = int(matriz.loc[classe].sum())
        preditos = int(matriz[classe].sum())
        fp = preditos - tp
        fn = suporte - tp
        precisao = tp / preditos if preditos else 0.0
        recall = tp / suporte if suporte else 0.0
        f1 = (
            2.0 * precisao * recall / (precisao + recall)
            if precisao + recall
            else 0.0
        )
        linhas_metricas.append(
            {
                "classe": classe,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "suporte": suporte,
                "preditos": preditos,
                "precisao": precisao,
                "recall": recall,
                "f1": f1,
            }
        )
    metricas = pd.DataFrame(linhas_metricas)
    metricas["suporte_adequado"] = metricas["suporte"] >= SUPORTE_MINIMO_CLASSE
    macro_f1 = float(metricas["f1"].mean())
    com_suporte = metricas[metricas["suporte_adequado"]]
    macro_f1_com_suporte = (
        float(com_suporte["f1"].mean()) if len(com_suporte) else float("nan")
    )
    subdimensionadas = tuple(
        metricas.loc[~metricas["suporte_adequado"], "classe"].tolist()
    )
    n_total = len(comparacao)
    n_abstencoes = int(comparacao["abster"].sum())
    n_cobertos = n_total - n_abstencoes
    cobertura = n_cobertos / n_total
    taxa_abstencao = n_abstencoes / n_total
    aprovado_macro = bool(
        math.isfinite(macro_f1_com_suporte)
        and macro_f1_com_suporte >= LIMIAR_MACRO_F1
        and len(com_suporte) >= MIN_CLASSES_COM_SUPORTE
        and metricas["suporte"].gt(0).all()
        and metricas["preditos"].gt(0).all()
    )
    aprovado_kappa = math.isfinite(kappa) and kappa >= LIMIAR_KAPPA

    return ResultadoValidacaoIA(
        matriz_confusao=matriz,
        metricas_por_classe=metricas,
        macro_f1=macro_f1,
        macro_f1_com_suporte=macro_f1_com_suporte,
        classes_subdimensionadas=subdimensionadas,
        suporte_minimo_classe=SUPORTE_MINIMO_CLASSE,
        cobertura=cobertura,
        taxa_abstencao=taxa_abstencao,
        n_total=n_total,
        n_cobertos=n_cobertos,
        n_abstencoes=n_abstencoes,
        kappa_avaliadores=kappa,
        aprovado_macro_f1=aprovado_macro,
        aprovado_kappa=aprovado_kappa,
        aprovado=aprovado_macro and aprovado_kappa,
    )
