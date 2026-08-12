from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402
from scripts import v2_04_preparar_validacao_humana as preparar  # noqa: E402
from src.v2 import ia_eventos, validacao_ia  # noqa: E402


RAIZ = Path(__file__).resolve().parent.parent
DIR_VALIDACAO = RAIZ / "outputs" / "validacao_humana"
AVALIADOR_A_PADRAO = DIR_VALIDACAO / "avaliador_a.csv"
AVALIADOR_B_PADRAO = DIR_VALIDACAO / "avaliador_b.csv"
CHAVE_INTERNA_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "chave_validacao_humana.csv"
)
RESUMO_INTERNO_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "resumo_validacao_humana.csv"
)

ALIASES_FICHA = {
    "id_anonimo": {"id_anonimo", "codigo_anonimo", "id_amostra"},
    "texto": {"texto", "texto_documento", "conteudo"},
    "especifico_empresa": {
        "especifico_empresa",
        "evento_especifico",
        "especifico",
    },
    "direcao": {"direcao", "direcao_evento"},
}
ALIASES_ADJUDICACAO = {
    "id_anonimo": ALIASES_FICHA["id_anonimo"],
    "texto": ALIASES_FICHA["texto"],
    "especifico_empresa": {
        "gold_especifico_empresa",
        "especifico_empresa",
        "evento_especifico",
    },
    "direcao": {"gold_direcao", "direcao", "direcao_evento"},
}
COLUNAS_CHAVE = (
    "id_anonimo",
    "ID_Documento",
    "especifico_empresa_ia",
    "direcao_ia",
    "abster_ia",
    "ordem_avaliador_a",
    "ordem_avaliador_b",
    "protocolo_rotulagem",
    "desenho_amostra",
    "classe_ia",
    "protocolo_rotulagem_hash",
    "tamanho_amostra",
    "seed_amostra",
    "lote_hash",
    "prompt_versao",
    "prompt_hash",
    "alvo_direcao",
    "modelo",
    "modelo_digest",
    "modelo_quantizacao",
    "ollama_versao",
)
COLUNAS_RESUMO_LOTE = (
    "lote_hash",
    "prompt_versao",
    "prompt_hash",
    "alvo_direcao",
    "modelo",
    "modelo_digest",
    "modelo_quantizacao",
    "ollama_versao",
    "documentos_elegiveis",
    "classificacoes_ok",
    "erros_tecnicos",
    "taxa_erro_tecnico",
    "abstencoes",
    "taxa_abstencao",
    "tamanho_amostra",
    "seed_amostra",
    "protocolo_rotulagem",
    "protocolo_rotulagem_hash",
    "desenho_amostra",
    *validacao_ia.CLASSES_CONJUNTAS,
)
VERDADEIROS = {"true", "verdadeiro", "sim", "1"}
FALSOS = {"false", "falso", "nao", "0"}


class AdjudicacaoPendente(ValueError):
    def __init__(self, tabela: pd.DataFrame):
        self.tabela = tabela
        super().__init__(
            f"{len(tabela)} divergencia(s) exigem adjudicacao antes da avaliacao"
        )


@dataclass(frozen=True)
class ResultadoAvaliacaoHumana:
    rotulos_humanos: pd.DataFrame
    divergencias: pd.DataFrame
    gold_adjudicado: pd.DataFrame
    avaliacao: validacao_ia.ResultadoValidacaoIA


def _argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia a classificacao da IA contra rotulos humanos cegos.",
        epilog=(
            "Fichas: id_anonimo (ou codigo_anonimo/id_amostra), texto "
            "(ou texto_documento/conteudo), especifico_empresa "
            "(ou evento_especifico/especifico) e direcao. Adjudicacao: "
            "id_anonimo, texto, gold_especifico_empresa e gold_direcao; os nomes "
            "especifico_empresa e direcao tambem sao aceitos nesse arquivo. "
            "Booleanos aceitam true/false, sim/nao ou 1/0. Retornos: 0 "
            "aprovado, 2 gates reprovados, 3 aguardando adjudicacao e 1 erro."
        ),
    )
    parser.add_argument("--avaliador-a", type=Path, default=AVALIADOR_A_PADRAO)
    parser.add_argument("--avaliador-b", type=Path, default=AVALIADOR_B_PADRAO)
    parser.add_argument("--chave-interna", type=Path, default=CHAVE_INTERNA_PADRAO)
    parser.add_argument("--resumo-interno", type=Path, default=RESUMO_INTERNO_PADRAO)
    parser.add_argument("--adjudicacao", type=Path)
    return parser.parse_args(argv)


def _absoluto(caminho: Path) -> Path:
    return caminho if caminho.is_absolute() else RAIZ / caminho


def _nome_coluna(valor: object) -> str:
    texto = unicodedata.normalize("NFKD", str(valor).strip().casefold())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")


def _token(valor: object) -> str:
    texto = unicodedata.normalize("NFKD", str(valor).strip().casefold())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _ler_csv(caminho: Path) -> pd.DataFrame:
    if not caminho.is_file():
        raise FileNotFoundError(f"arquivo nao encontrado: {caminho}")
    if caminho.suffix.casefold() != ".csv":
        raise ValueError(f"a validacao humana exige CSV: {caminho}")
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        try:
            cabecalho = next(csv.reader(arquivo))
        except StopIteration as exc:
            raise ValueError(f"CSV vazio: {caminho}") from exc
    nomes = [_nome_coluna(coluna) for coluna in cabecalho]
    if any(not nome for nome in nomes) or len(nomes) != len(set(nomes)):
        raise ValueError(f"CSV contem cabecalho vazio ou duplicado: {caminho}")
    try:
        return pd.read_csv(caminho, dtype=object, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV vazio: {caminho}") from exc


def _resolver_coluna(
    tabela: pd.DataFrame,
    aliases: set[str],
    campo: str,
    nome: str,
) -> object:
    encontradas = [
        coluna for coluna in tabela.columns if _nome_coluna(coluna) in aliases
    ]
    if len(encontradas) != 1:
        raise ValueError(
            f"{nome}: campo {campo} deve ter um unico cabecalho reconhecido; "
            f"encontrados={encontradas}"
        )
    return encontradas[0]


def _normalizar_ids_anonimos(serie: pd.Series, nome: str) -> pd.Series:
    ids = serie.astype(str).str.strip().str.upper()
    if ids.eq("").any():
        raise ValueError(f"{nome}: id_anonimo vazio")
    if ids.duplicated(keep=False).any():
        repetidos = ids[ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"{nome}: id_anonimo duplicado: {repetidos[:3]}")
    invalidos = ~ids.str.fullmatch(r"VH-[0-9A-F]{12}")
    if invalidos.any():
        raise ValueError(f"{nome}: id_anonimo invalido: {ids[invalidos].iloc[0]}")
    return ids


def _normalizar_ids_reais(serie: pd.Series, nome: str) -> pd.Series:
    ids = serie.astype(str).str.strip()
    if ids.eq("").any():
        raise ValueError(f"{nome}: ID_Documento vazio")
    if ids.duplicated(keep=False).any():
        repetidos = ids[ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"{nome}: ID_Documento duplicado: {repetidos[:3]}")
    return ids


def _normalizar_booleanos(
    serie: pd.Series,
    campo: str,
    nome: str,
) -> pd.Series:
    valores: list[bool] = []
    for valor in serie:
        token = _token(valor)
        if token in VERDADEIROS:
            valores.append(True)
        elif token in FALSOS:
            valores.append(False)
        else:
            raise ValueError(
                f"{nome}: {campo} deve usar true ou false; recebido={valor!r}"
            )
    return pd.Series(valores, index=serie.index, dtype=bool)


def _normalizar_direcoes(serie: pd.Series, nome: str) -> pd.Series:
    direcoes = serie.map(_token)
    invalidas = sorted(set(direcoes) - set(validacao_ia.DIRECOES))
    if invalidas:
        raise ValueError(
            f"{nome}: direcao deve ser positiva, negativa ou neutra: {invalidas}"
        )
    return direcoes.astype(str)


def _validar_invariantes(
    tabela: pd.DataFrame,
    coluna_especifico: str,
    coluna_direcao: str,
    nome: str,
) -> None:
    invalidos = ~tabela[coluna_especifico] & tabela[coluna_direcao].ne("neutra")
    if invalidos.any():
        raise ValueError(f"{nome}: evento nao especifico deve ter direcao neutra")


def normalizar_ficha(tabela: pd.DataFrame, nome: str) -> pd.DataFrame:
    colunas = {
        campo: _resolver_coluna(tabela, aliases, campo, nome)
        for campo, aliases in ALIASES_FICHA.items()
    }
    extras = [coluna for coluna in tabela.columns if coluna not in colunas.values()]
    if extras:
        raise ValueError(f"{nome}: colunas extras nao permitidas: {extras}")
    base = tabela[list(colunas.values())].rename(
        columns={coluna: campo for campo, coluna in colunas.items()}
    )
    base["id_anonimo"] = _normalizar_ids_anonimos(base["id_anonimo"], nome)
    base["texto"] = base["texto"].astype(str)
    if base["texto"].str.strip().eq("").any():
        raise ValueError(f"{nome}: texto vazio")
    base["especifico_empresa"] = _normalizar_booleanos(
        base["especifico_empresa"], "especifico_empresa", nome
    )
    base["direcao"] = _normalizar_direcoes(base["direcao"], nome)
    _validar_invariantes(base, "especifico_empresa", "direcao", nome)
    return base


def normalizar_chave(
    tabela: pd.DataFrame,
    *,
    exigir_amostra_oficial: bool = False,
) -> pd.DataFrame:
    encontradas = set(tabela.columns)
    esperadas = set(COLUNAS_CHAVE)
    if encontradas != esperadas:
        raise ValueError(
            "chave interna com schema invalido; "
            f"faltantes={sorted(esperadas - encontradas)}, "
            f"extras={sorted(encontradas - esperadas)}"
        )
    if tabela.empty:
        raise ValueError("chave interna vazia")
    base = tabela[list(COLUNAS_CHAVE)].copy()
    base["id_anonimo"] = _normalizar_ids_anonimos(
        base["id_anonimo"], "chave interna"
    )
    base["ID_Documento"] = _normalizar_ids_reais(
        base["ID_Documento"], "chave interna"
    )
    base["especifico_empresa_ia"] = _normalizar_booleanos(
        base["especifico_empresa_ia"], "especifico_empresa_ia", "chave interna"
    )
    base["abster_ia"] = _normalizar_booleanos(
        base["abster_ia"], "abster_ia", "chave interna"
    )
    base["direcao_ia"] = _normalizar_direcoes(
        base["direcao_ia"], "chave interna"
    )
    _validar_invariantes(
        base, "especifico_empresa_ia", "direcao_ia", "chave interna"
    )
    abstencao_invalida = base["abster_ia"] & base["direcao_ia"].ne("neutra")
    operacao_invalida = ~base["abster_ia"] & (
        ~base["especifico_empresa_ia"] | base["direcao_ia"].eq("neutra")
    )
    if abstencao_invalida.any() or operacao_invalida.any():
        raise ValueError("chave interna: invariantes de abstencao violadas")
    for coluna in ("ordem_avaliador_a", "ordem_avaliador_b"):
        numeros = pd.to_numeric(base[coluna], errors="coerce")
        if numeros.isna().any() or (numeros % 1).ne(0).any():
            raise ValueError(f"chave interna: {coluna} deve conter inteiros")
        base[coluna] = numeros.astype(int)
        if sorted(base[coluna].tolist()) != list(range(1, len(base) + 1)):
            raise ValueError(f"chave interna: {coluna} nao e uma permutacao valida")

    protocolo_hash = hashlib.sha256(
        preparar.PROTOCOLO_ROTULAGEM.read_bytes()
    ).hexdigest()
    constantes = {
        "protocolo_rotulagem": preparar.VERSAO_PROTOCOLO,
        "protocolo_rotulagem_hash": protocolo_hash,
        "desenho_amostra": preparar.DESENHO_AMOSTRA,
        "prompt_versao": ia_eventos.PROMPT_VERSAO,
        "prompt_hash": ia_eventos.HASH_PROMPT,
        "alvo_direcao": ia_eventos.ALVO_DIRECAO,
        "modelo": preparar.MODELO_CONGELADO,
        "modelo_digest": preparar.DIGEST_MODELO_CONGELADO,
        "modelo_quantizacao": preparar.QUANTIZACAO_CONGELADA,
    }
    for coluna, esperado in constantes.items():
        if not base[coluna].fillna("").astype(str).eq(esperado).all():
            raise ValueError(f"chave interna: {coluna} diverge do contrato")
    for coluna in ("ollama_versao", "lote_hash"):
        valores = base[coluna].fillna("").astype(str).str.strip()
        if valores.eq("").any() or valores.nunique() != 1:
            raise ValueError(f"chave interna: {coluna} deve ser unico e nao vazio")
        base[coluna] = valores
    if not base["lote_hash"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("chave interna: lote_hash invalido")

    for coluna, esperado in (
        ("tamanho_amostra", len(base)),
        ("seed_amostra", preparar.SEED_VALIDACAO),
    ):
        valores = pd.to_numeric(base[coluna], errors="coerce")
        if valores.isna().any() or (valores % 1).ne(0).any():
            raise ValueError(f"chave interna: {coluna} deve conter inteiros")
        base[coluna] = valores.astype(int)
        if not base[coluna].eq(esperado).all():
            raise ValueError(f"chave interna: {coluna} diverge do contrato")

    classes = [
        (
            f"especifico_{direcao}" if especifico else "nao_especifico"
        )
        for especifico, direcao in zip(
            base["especifico_empresa_ia"], base["direcao_ia"], strict=True
        )
    ]
    if base["classe_ia"].astype(str).tolist() != classes:
        raise ValueError("chave interna: classe_ia inconsistente")
    if exigir_amostra_oficial:
        if len(base) != preparar.TAMANHO_PADRAO:
            raise ValueError("chave interna: amostra oficial deve ter 90 documentos")
        contagens = base["direcao_ia"].value_counts().to_dict()
        if contagens != {direcao: 30 for direcao in validacao_ia.DIRECOES}:
            raise ValueError("chave interna: amostra oficial deve ter 30 por direcao")
    return base.sort_values("id_anonimo", kind="stable").reset_index(drop=True)


def normalizar_resumo_lote(
    tabela: pd.DataFrame,
    chave: pd.DataFrame,
) -> pd.DataFrame:
    encontradas = set(tabela.columns)
    esperadas = set(COLUNAS_RESUMO_LOTE)
    if encontradas != esperadas or len(tabela) != 1:
        raise ValueError(
            "resumo interno com schema ou numero de linhas invalido; "
            f"faltantes={sorted(esperadas - encontradas)}, "
            f"extras={sorted(encontradas - esperadas)}"
        )
    base = tabela[list(COLUNAS_RESUMO_LOTE)].copy()
    linha_chave = chave.iloc[0]
    metadados = (
        "lote_hash",
        "prompt_versao",
        "prompt_hash",
        "alvo_direcao",
        "modelo",
        "modelo_digest",
        "modelo_quantizacao",
        "ollama_versao",
        "protocolo_rotulagem",
        "protocolo_rotulagem_hash",
        "desenho_amostra",
    )
    for coluna in metadados:
        if str(base.at[0, coluna]).strip() != str(linha_chave[coluna]).strip():
            raise ValueError(f"resumo interno: {coluna} diverge da chave")

    colunas_inteiras = (
        "documentos_elegiveis",
        "classificacoes_ok",
        "erros_tecnicos",
        "abstencoes",
        "tamanho_amostra",
        "seed_amostra",
        *validacao_ia.CLASSES_CONJUNTAS,
    )
    for coluna in colunas_inteiras:
        numero = pd.to_numeric(base.at[0, coluna], errors="coerce")
        if pd.isna(numero) or numero < 0 or numero % 1:
            raise ValueError(f"resumo interno: {coluna} deve ser inteiro nao negativo")
        base.at[0, coluna] = int(numero)
    total = int(base.at[0, "documentos_elegiveis"])
    total_ok = int(base.at[0, "classificacoes_ok"])
    erros = int(base.at[0, "erros_tecnicos"])
    abstencoes = int(base.at[0, "abstencoes"])
    if total != total_ok + erros or total_ok < len(chave):
        raise ValueError("resumo interno: contagens de cobertura inconsistentes")
    if sum(int(base.at[0, classe]) for classe in validacao_ia.CLASSES_CONJUNTAS) != total_ok:
        raise ValueError("resumo interno: contagens por classe inconsistentes")
    if int(base.at[0, "tamanho_amostra"]) != len(chave):
        raise ValueError("resumo interno: tamanho_amostra diverge da chave")
    if int(base.at[0, "seed_amostra"]) != preparar.SEED_VALIDACAO:
        raise ValueError("resumo interno: seed_amostra diverge do contrato")

    taxas = {
        "taxa_erro_tecnico": erros / total,
        "taxa_abstencao": abstencoes / total_ok,
    }
    for coluna, esperado in taxas.items():
        valor = pd.to_numeric(base.at[0, coluna], errors="coerce")
        if pd.isna(valor) or abs(float(valor) - esperado) > 1e-12:
            raise ValueError(f"resumo interno: {coluna} inconsistente")
        base.at[0, coluna] = float(valor)
    return base


def _validar_ordem_ficha(
    ficha: pd.DataFrame,
    chave: pd.DataFrame,
    coluna_ordem: str,
    nome: str,
) -> None:
    posicoes = {
        documento: posicao
        for posicao, documento in enumerate(ficha["id_anonimo"], start=1)
    }
    esperadas = chave.set_index("id_anonimo")[coluna_ordem].to_dict()
    if posicoes != esperadas:
        raise ValueError(f"{nome}: ordem dos IDs diverge da chave interna")


def _normalizar_adjudicacao(tabela: pd.DataFrame) -> pd.DataFrame:
    nome = "adjudicacao"
    colunas = {
        campo: _resolver_coluna(tabela, aliases, campo, nome)
        for campo, aliases in ALIASES_ADJUDICACAO.items()
    }
    extras = [coluna for coluna in tabela.columns if coluna not in colunas.values()]
    if extras:
        raise ValueError(f"{nome}: colunas extras nao permitidas: {extras}")
    base = tabela[list(colunas.values())].rename(
        columns={coluna: campo for campo, coluna in colunas.items()}
    )
    base["id_anonimo"] = _normalizar_ids_anonimos(base["id_anonimo"], nome)
    base["texto"] = base["texto"].astype(str)
    if base["texto"].str.strip().eq("").any():
        raise ValueError("adjudicacao: texto vazio")
    base["especifico_empresa"] = _normalizar_booleanos(
        base["especifico_empresa"], "gold_especifico_empresa", nome
    )
    base["direcao"] = _normalizar_direcoes(base["direcao"], nome)
    _validar_invariantes(base, "especifico_empresa", "direcao", nome)
    return base


def _template_adjudicacao(
    divergencias: pd.DataFrame,
    ficha_a: pd.DataFrame,
) -> pd.DataFrame:
    tabela = divergencias[["id_anonimo"]].merge(
        ficha_a[["id_anonimo", "texto"]],
        on="id_anonimo",
        how="left",
        validate="one_to_one",
    )
    tabela["gold_especifico_empresa"] = pd.NA
    tabela["gold_direcao"] = pd.NA
    return tabela[
        [
            "id_anonimo",
            "texto",
            "gold_especifico_empresa",
            "gold_direcao",
        ]
    ]


def avaliar_validacao(
    ficha_a: pd.DataFrame,
    ficha_b: pd.DataFrame,
    chave_interna: pd.DataFrame,
    adjudicacao: pd.DataFrame | None = None,
    *,
    exigir_amostra_oficial: bool = False,
) -> ResultadoAvaliacaoHumana:
    a = normalizar_ficha(ficha_a, "avaliador A")
    b = normalizar_ficha(ficha_b, "avaliador B")
    chave = normalizar_chave(
        chave_interna,
        exigir_amostra_oficial=exigir_amostra_oficial,
    )
    ids_chave = set(chave["id_anonimo"])
    if set(a["id_anonimo"]) != ids_chave or set(b["id_anonimo"]) != ids_chave:
        raise ValueError("fichas e chave interna devem conter exatamente os mesmos IDs")
    textos = a[["id_anonimo", "texto"]].merge(
        b[["id_anonimo", "texto"]],
        on="id_anonimo",
        how="inner",
        validate="one_to_one",
        suffixes=("_a", "_b"),
    )
    if textos["texto_a"].ne(textos["texto_b"]).any():
        raise ValueError("avaliadores receberam textos diferentes para o mesmo ID")
    _validar_ordem_ficha(a, chave, "ordem_avaliador_a", "avaliador A")
    _validar_ordem_ficha(b, chave, "ordem_avaliador_b", "avaliador B")

    rotulos = validacao_ia.validar_rotulos_humanos(
        a[["id_anonimo", "especifico_empresa", "direcao"]],
        b[["id_anonimo", "especifico_empresa", "direcao"]],
        ids_esperados=chave["id_anonimo"].tolist(),
        coluna_id="id_anonimo",
    )
    kappa = validacao_ia.calcular_cohen_kappa(rotulos)
    divergencias = validacao_ia.criar_tabela_divergencias(
        rotulos, coluna_id="id_anonimo"
    )
    template = _template_adjudicacao(divergencias, a)
    ids_divergentes = set(divergencias["id_anonimo"])

    if ids_divergentes and adjudicacao is None:
        raise AdjudicacaoPendente(template)
    if adjudicacao is None:
        adj = pd.DataFrame(
            columns=["id_anonimo", "especifico_empresa", "direcao"]
        )
    else:
        adj = _normalizar_adjudicacao(adjudicacao)
    if set(adj["id_anonimo"]) != ids_divergentes:
        raise ValueError(
            "adjudicacao deve conter exatamente os IDs humanos divergentes"
        )
    if ids_divergentes:
        textos_adj = adj.set_index("id_anonimo")["texto"]
        textos_originais = a.set_index("id_anonimo")["texto"]
        if any(
            textos_adj.at[codigo] != textos_originais.at[codigo]
            for codigo in ids_divergentes
        ):
            raise ValueError("adjudicacao: texto diverge da ficha original")

    gold = rotulos[
        ["id_anonimo", "especifico_empresa_a", "direcao_a", "classe_a", "classe_b"]
    ].copy()
    gold = gold.rename(
        columns={
            "especifico_empresa_a": "especifico_empresa",
            "direcao_a": "direcao",
        }
    )
    gold["origem_rotulo"] = "concordancia"
    if ids_divergentes:
        adj_indexada = adj.set_index("id_anonimo")
        mascara = gold["id_anonimo"].isin(ids_divergentes)
        for indice in gold.index[mascara]:
            codigo = gold.at[indice, "id_anonimo"]
            gold.at[indice, "especifico_empresa"] = bool(
                adj_indexada.at[codigo, "especifico_empresa"]
            )
            gold.at[indice, "direcao"] = str(adj_indexada.at[codigo, "direcao"])
            gold.at[indice, "origem_rotulo"] = "adjudicacao"
        divergencias = divergencias.drop(
            columns=["gold_especifico_empresa", "gold_direcao"]
        ).merge(
            adj.drop(columns=["texto"]).rename(
                columns={
                    "especifico_empresa": "gold_especifico_empresa",
                    "direcao": "gold_direcao",
                }
            ),
            on="id_anonimo",
            how="left",
            validate="one_to_one",
        )

    gold = gold.drop(columns=["classe_a", "classe_b"]).merge(
        chave[["id_anonimo", "ID_Documento"]],
        on="id_anonimo",
        how="left",
        validate="one_to_one",
    )
    gold = gold[
        [
            "id_anonimo",
            "ID_Documento",
            "especifico_empresa",
            "direcao",
            "origem_rotulo",
        ]
    ].sort_values("id_anonimo", kind="stable").reset_index(drop=True)
    predicoes = chave[
        [
            "ID_Documento",
            "especifico_empresa_ia",
            "direcao_ia",
            "abster_ia",
        ]
    ].rename(
        columns={
            "especifico_empresa_ia": "especifico_empresa",
            "direcao_ia": "direcao",
            "abster_ia": "abster",
        }
    )
    avaliacao = validacao_ia.avaliar_ia_contra_gold(
        predicoes,
        gold[["ID_Documento", "especifico_empresa", "direcao"]],
        kappa,
    )
    rotulos = chave[["id_anonimo", "ID_Documento"]].merge(
        rotulos, on="id_anonimo", how="left", validate="one_to_one"
    )
    rotulos["concordam"] = rotulos["classe_a"].eq(rotulos["classe_b"])
    return ResultadoAvaliacaoHumana(rotulos, divergencias, gold, avaliacao)


def _resumo(resultado: ResultadoAvaliacaoHumana) -> pd.DataFrame:
    avaliacao = resultado.avaliacao
    return pd.DataFrame(
        [
            {
                "macro_f1": avaliacao.macro_f1,
                "macro_f1_com_suporte": avaliacao.macro_f1_com_suporte,
                "classes_subdimensionadas": ";".join(
                    avaliacao.classes_subdimensionadas
                ),
                "suporte_minimo_classe": avaliacao.suporte_minimo_classe,
                "limiar_macro_f1": validacao_ia.LIMIAR_MACRO_F1,
                "aprovado_macro_f1": avaliacao.aprovado_macro_f1,
                "kappa_avaliadores": avaliacao.kappa_avaliadores,
                "limiar_kappa": validacao_ia.LIMIAR_KAPPA,
                "aprovado_kappa": avaliacao.aprovado_kappa,
                "cobertura": avaliacao.cobertura,
                "taxa_abstencao": avaliacao.taxa_abstencao,
                "documentos": avaliacao.n_total,
                "documentos_cobertos": avaliacao.n_cobertos,
                "abstencoes": avaliacao.n_abstencoes,
                "divergencias_humanas": len(resultado.divergencias),
                "aprovado": avaliacao.aprovado,
            }
        ]
    )


def _gravar_tabelas(
    resultado: ResultadoAvaliacaoHumana,
    resumo_lote: pd.DataFrame,
    diretorio: Path,
) -> None:
    diretorio.mkdir(parents=True, exist_ok=True)
    tabelas = {
        "rotulos_humanos_reconciliados.csv": resultado.rotulos_humanos,
        "divergencias_adjudicadas.csv": resultado.divergencias,
        "gold_adjudicado.csv": resultado.gold_adjudicado,
        "matriz_confusao.csv": resultado.avaliacao.matriz_confusao.reset_index(),
        "metricas_por_classe.csv": resultado.avaliacao.metricas_por_classe,
        "resumo_validacao.csv": _resumo(resultado),
        "resumo_lote_ia.csv": resumo_lote,
    }
    for nome, tabela in tabelas.items():
        tabela.to_csv(diretorio / nome, index=False, encoding="utf-8-sig")


def _extras(
    resultado: ResultadoAvaliacaoHumana,
    resumo_lote: pd.DataFrame,
) -> dict[str, object]:
    avaliacao = resultado.avaliacao
    lote = resumo_lote.iloc[0]
    return {
        "macro_f1": float(avaliacao.macro_f1),
        "macro_f1_com_suporte": float(avaliacao.macro_f1_com_suporte),
        "classes_subdimensionadas": list(avaliacao.classes_subdimensionadas),
        "kappa_avaliadores": float(avaliacao.kappa_avaliadores),
        "cobertura": float(avaliacao.cobertura),
        "taxa_abstencao": float(avaliacao.taxa_abstencao),
        "documentos": int(avaliacao.n_total),
        "divergencias_humanas": int(len(resultado.divergencias)),
        "aprovado_macro_f1": bool(avaliacao.aprovado_macro_f1),
        "aprovado_kappa": bool(avaliacao.aprovado_kappa),
        "aprovado": bool(avaliacao.aprovado),
        "documentos_elegiveis_lote": int(lote["documentos_elegiveis"]),
        "classificacoes_ok_lote": int(lote["classificacoes_ok"]),
        "erros_tecnicos_lote": int(lote["erros_tecnicos"]),
        "taxa_erro_tecnico_lote": float(lote["taxa_erro_tecnico"]),
        "taxa_abstencao_lote": float(lote["taxa_abstencao"]),
        "lote_hash": str(lote["lote_hash"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    avaliador_a_path = _absoluto(args.avaliador_a)
    avaliador_b_path = _absoluto(args.avaliador_b)
    chave_path = _absoluto(args.chave_interna)
    resumo_path = _absoluto(args.resumo_interno)
    adjudicacao_path = (
        _absoluto(args.adjudicacao) if args.adjudicacao is not None else None
    )
    entradas = [avaliador_a_path, avaliador_b_path, chave_path, resumo_path]
    if adjudicacao_path is not None:
        entradas.append(adjudicacao_path)

    execucao = criar_execucao("v2_validacao_ia")
    log = configurar_log(execucao, "v2_05_validacao_ia")
    config = {
        "avaliador_a": str(avaliador_a_path),
        "avaliador_b": str(avaliador_b_path),
        "chave_interna": str(chave_path),
        "resumo_interno": str(resumo_path),
        "adjudicacao": str(adjudicacao_path) if adjudicacao_path else None,
        "limiar_macro_f1": validacao_ia.LIMIAR_MACRO_F1,
        "limiar_kappa": validacao_ia.LIMIAR_KAPPA,
        "suporte_minimo_classe": validacao_ia.SUPORTE_MINIMO_CLASSE,
    }
    try:
        chave_bruta = _ler_csv(chave_path)
        chave_validada = normalizar_chave(
            chave_bruta,
            exigir_amostra_oficial=True,
        )
        resumo_lote = normalizar_resumo_lote(
            _ler_csv(resumo_path),
            chave_validada,
        )
        resultado = avaliar_validacao(
            _ler_csv(avaliador_a_path),
            _ler_csv(avaliador_b_path),
            chave_bruta,
            _ler_csv(adjudicacao_path) if adjudicacao_path is not None else None,
            exigir_amostra_oficial=True,
        )
        _gravar_tabelas(resultado, resumo_lote, execucao.tabelas)
        extras = _extras(resultado, resumo_lote)
        extras["diretorio_tabelas"] = str(execucao.tabelas)
        gravar_manifesto(
            execucao,
            config,
            arquivos_dados=entradas,
            status="concluida",
            extras=extras,
        )
        log.info(
            "macro-F1 %.4f (gate %.4f) | kappa %.4f | cobertura %.4f | gate %s",
            resultado.avaliacao.macro_f1,
            resultado.avaliacao.macro_f1_com_suporte,
            resultado.avaliacao.kappa_avaliadores,
            resultado.avaliacao.cobertura,
            "aprovado" if resultado.avaliacao.aprovado else "reprovado",
        )
        if resultado.avaliacao.classes_subdimensionadas:
            log.info(
                "fora do macro-F1 do gate por suporte < %d: %s",
                resultado.avaliacao.suporte_minimo_classe,
                ", ".join(resultado.avaliacao.classes_subdimensionadas),
            )
        return 0 if resultado.avaliacao.aprovado else 2
    except AdjudicacaoPendente as exc:
        destino = execucao.tabelas / "divergencias_para_adjudicacao.csv"
        exc.tabela.to_csv(destino, index=False, encoding="utf-8-sig")
        gravar_manifesto(
            execucao,
            config,
            arquivos_dados=[caminho for caminho in entradas if caminho.exists()],
            status="aguardando_adjudicacao",
            extras={
                "divergencias_humanas": int(len(exc.tabela)),
                "ficha_adjudicacao": str(destino),
            },
        )
        log.error("adjudicacao pendente: %s", destino)
        return 3
    except Exception as exc:
        log.exception("validacao humana falhou: %s", exc)
        gravar_manifesto(
            execucao,
            config,
            arquivos_dados=[caminho for caminho in entradas if caminho.exists()],
            status="falhou",
            extras={"erro": str(exc)},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
