from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v2 import ia_eventos, validacao_ia  # noqa: E402


RAIZ = Path(__file__).resolve().parent.parent
CORPUS_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "corpus_lideres_top20.parquet"
)
CLASSIFICACOES_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "classificacoes_ia.parquet"
)
DIR_PLANILHAS_PADRAO = RAIZ / "outputs" / "validacao_humana"
AVALIADOR_A_PADRAO = DIR_PLANILHAS_PADRAO / "avaliador_a.csv"
AVALIADOR_B_PADRAO = DIR_PLANILHAS_PADRAO / "avaliador_b.csv"
CHAVE_INTERNA_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "chave_validacao_humana.csv"
)
RESUMO_INTERNO_PADRAO = (
    RAIZ / "data" / "processed" / "cvm_ipe" / "resumo_validacao_humana.csv"
)
PROTOCOLO_ROTULAGEM = RAIZ / "docs" / "PROTOCOLO_ROTULAGEM_V2.md"
VERSAO_PROTOCOLO = "rotulagem-eventos-1.0.0"
TAMANHO_PADRAO = 90
SEED_VALIDACAO = 20260811
MODELO_CONGELADO = "qwen3:14b"
DIGEST_MODELO_CONGELADO = (
    "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
)
QUANTIZACAO_CONGELADA = "Q4_K_M"
DESENHO_AMOSTRA = "balanceada_direcao_com_nao_especificos"
COLUNAS_PLANILHA = (
    "id_anonimo",
    "texto",
    "especifico_empresa",
    "direcao",
)


@dataclass(frozen=True)
class ResultadoPreparacao:
    avaliador_a: pd.DataFrame
    avaliador_b: pd.DataFrame
    chave_interna: pd.DataFrame
    resumo_interno: pd.DataFrame


@dataclass(frozen=True)
class LoteValidado:
    classificacoes_ok: pd.DataFrame
    classificacoes_completas: pd.DataFrame
    identidade: dict[str, str]
    lote_hash: str


def _argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara a amostra cega para validacao humana da V2.",
        epilog=(
            "Preencha especifico_empresa com true ou false e direcao com "
            "positiva, negativa ou neutra."
        ),
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_PADRAO)
    parser.add_argument(
        "--classificacoes", type=Path, default=CLASSIFICACOES_PADRAO
    )
    parser.add_argument(
        "--avaliador-a", type=Path, default=AVALIADOR_A_PADRAO
    )
    parser.add_argument(
        "--avaliador-b", type=Path, default=AVALIADOR_B_PADRAO
    )
    parser.add_argument(
        "--chave-interna", type=Path, default=CHAVE_INTERNA_PADRAO
    )
    parser.add_argument(
        "--resumo-interno", type=Path, default=RESUMO_INTERNO_PADRAO
    )
    parser.add_argument("--tamanho", type=int, default=TAMANHO_PADRAO)
    parser.add_argument("--sobrescrever", action="store_true")
    return parser.parse_args(argv)


def _absoluto(caminho: Path) -> Path:
    return caminho if caminho.is_absolute() else RAIZ / caminho


def _ler_tabela(caminho: Path) -> pd.DataFrame:
    if not caminho.is_file():
        raise FileNotFoundError(f"arquivo nao encontrado: {caminho}")
    sufixo = caminho.suffix.casefold()
    if sufixo in {".parquet", ".pq"}:
        return pd.read_parquet(caminho)
    if sufixo == ".csv":
        return pd.read_csv(caminho)
    raise ValueError(f"formato de tabela nao suportado: {caminho.suffix}")


def _exigir_colunas(
    tabela: pd.DataFrame,
    colunas: set[str],
    nome: str,
) -> None:
    faltantes = sorted(colunas - set(tabela.columns))
    if faltantes:
        raise ValueError(f"{nome} sem colunas: {faltantes}")


def _normalizar_ids(tabela: pd.DataFrame, nome: str) -> pd.DataFrame:
    base = tabela.copy()
    vazios = base["ID_Documento"].isna() | base["ID_Documento"].astype(
        str
    ).str.strip().eq("")
    if vazios.any():
        raise ValueError(f"{nome} contem ID_Documento vazio")
    base["ID_Documento"] = base["ID_Documento"].astype(str).str.strip()
    if base["ID_Documento"].duplicated().any():
        raise ValueError(f"{nome} contem ID_Documento duplicado")
    return base


def _unico_nao_vazio(tabela: pd.DataFrame, coluna: str) -> str:
    valores = tabela[coluna].fillna("").astype(str).str.strip()
    if valores.eq("").any() or valores.nunique() != 1:
        raise ValueError(f"classificacoes ok sem identidade unica em {coluna}")
    return str(valores.iloc[0])


def _validar_saidas_ok(tabela: pd.DataFrame) -> pd.DataFrame:
    base = tabela.copy()
    for coluna in ("especifico_empresa", "abster"):
        validos = base[coluna].map(lambda valor: isinstance(valor, (bool, np.bool_)))
        if not validos.all():
            raise ValueError(f"classificacoes ok: {coluna} deve ser bool")
        base[coluna] = base[coluna].astype(bool)
    base["direcao"] = base["direcao"].fillna("").astype(str).str.strip()
    invalidas = sorted(set(base["direcao"]) - set(validacao_ia.DIRECOES))
    if invalidas:
        raise ValueError(f"classificacoes ok: direcao invalida: {invalidas}")
    nao_especifico = ~base["especifico_empresa"] & base["direcao"].ne("neutra")
    abstencao_invalida = base["abster"] & base["direcao"].ne("neutra")
    operacao_invalida = ~base["abster"] & (
        ~base["especifico_empresa"] | base["direcao"].eq("neutra")
    )
    if nao_especifico.any() or abstencao_invalida.any() or operacao_invalida.any():
        raise ValueError("classificacoes ok: invariantes de abstencao violadas")
    return base


def _classe_ia(especifico: bool, direcao: str) -> str:
    return f"especifico_{direcao}" if especifico else "nao_especifico"


def _valor_canonico(valor: object) -> object:
    if pd.isna(valor):
        return None
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, np.integer):
        return int(valor)
    return valor


def _hash_lote(tabela: pd.DataFrame) -> str:
    colunas = [
        "ID_Documento",
        "status_ia",
        "erro_ia",
        "especifico_empresa",
        "direcao",
        "abster",
        "prompt_versao",
        "prompt_hash",
        "alvo_direcao",
        "modelo",
        "modelo_digest",
        "modelo_quantizacao",
        "ollama_versao",
        "texto_hash",
        "_texto_hash_atual",
    ]
    registros = []
    for registro in tabela.sort_values("ID_Documento", kind="stable")[colunas].to_dict(
        "records"
    ):
        registros.append(
            {chave: _valor_canonico(valor) for chave, valor in registro.items()}
        )
    bruto = json.dumps(
        {"documentos": registros},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(bruto).hexdigest()


def _juntar_dados(
    corpus: pd.DataFrame,
    classificacoes: pd.DataFrame,
) -> LoteValidado:
    colunas_corpus = (
        "ID_Documento",
        "Data_Entrega",
        "emissor_id",
        "texto_llm",
        "status_documento",
    )
    colunas_ia = (
        "ID_Documento",
        "status_ia",
        "erro_ia",
        "especifico_empresa",
        "direcao",
        "abster",
        "prompt_versao",
        "prompt_hash",
        "alvo_direcao",
        "modelo",
        "modelo_digest",
        "modelo_quantizacao",
        "ollama_versao",
        "texto_hash",
    )
    _exigir_colunas(corpus, set(colunas_corpus), "corpus")
    _exigir_colunas(classificacoes, set(colunas_ia), "classificacoes")
    corpus_base = _normalizar_ids(corpus[list(colunas_corpus)], "corpus")
    elegiveis = corpus_base.loc[
        corpus_base["status_documento"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("ok")
        & corpus_base["texto_llm"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if elegiveis.empty:
        raise ValueError("nenhum documento elegivel no corpus")

    ia_base = _normalizar_ids(classificacoes[list(colunas_ia)], "classificacoes")
    ids_elegiveis = set(elegiveis["ID_Documento"])
    ids_classificados = set(ia_base["ID_Documento"])
    if ids_elegiveis != ids_classificados:
        faltantes = sorted(ids_elegiveis - ids_classificados)
        extras = sorted(ids_classificados - ids_elegiveis)
        raise ValueError(
            "classificacoes nao cobrem exatamente os documentos elegiveis; "
            f"faltantes={faltantes[:3]}, extras={extras[:3]}"
        )

    status = (
        ia_base["status_ia"].fillna("").astype(str).str.strip().str.casefold()
    )
    invalidos = sorted(set(status) - {"ok", "erro"})
    if invalidos:
        raise ValueError(f"classificacoes: status_ia invalido: {invalidos}")
    ia_base["status_ia"] = status
    ok = ia_base.loc[ia_base["status_ia"].eq("ok")].copy()
    if ok.empty:
        raise ValueError("nenhuma classificacao com status_ia=ok")
    ok = _validar_saidas_ok(ok)

    esperados = {
        "prompt_versao": ia_eventos.PROMPT_VERSAO,
        "prompt_hash": ia_eventos.HASH_PROMPT,
        "alvo_direcao": ia_eventos.ALVO_DIRECAO,
        "modelo": MODELO_CONGELADO,
        "modelo_digest": DIGEST_MODELO_CONGELADO,
        "modelo_quantizacao": QUANTIZACAO_CONGELADA,
    }
    for coluna, esperado in esperados.items():
        valores = ok[coluna].fillna("").astype(str).str.strip()
        if not valores.eq(esperado).all():
            raise ValueError(
                f"classificacoes ok usam {coluna} diferente do contrato atual"
            )
        ok[coluna] = valores
    ollama = _unico_nao_vazio(ok, "ollama_versao")
    ok["ollama_versao"] = ollama

    hashes_atuais = {
        registro["ID_Documento"]: ia_eventos.identidade_requisicao(
            str(registro["texto_llm"]), MODELO_CONGELADO
        ).hash_texto
        for registro in elegiveis.to_dict("records")
    }
    hashes_ok = ok["texto_hash"].fillna("").astype(str).str.strip()
    esperados_ok = ok["ID_Documento"].map(hashes_atuais)
    divergentes = hashes_ok.ne(esperados_ok)
    if divergentes.any():
        ids = ok.loc[divergentes, "ID_Documento"].head(3).tolist()
        raise ValueError(f"classificacoes ok com texto_hash desatualizado: {ids}")
    ok["texto_hash"] = hashes_ok

    ia_base = ia_base.set_index("ID_Documento")
    ia_base.update(ok.set_index("ID_Documento"))
    ia_base = ia_base.reset_index()
    ia_base["_texto_hash_atual"] = ia_base["ID_Documento"].map(hashes_atuais)
    lote_hash = _hash_lote(ia_base)
    base = ok.merge(
        elegiveis.drop(columns=["status_documento"]),
        on="ID_Documento",
        how="left",
        validate="one_to_one",
    )
    identidade = {
        **esperados,
        "ollama_versao": ollama,
    }
    return LoteValidado(base, ia_base, identidade, lote_hash)


def _id_anonimo(documento: str, seed: int) -> str:
    bruto = f"validacao-humana-v2\0{seed}\0{documento}".encode("utf-8")
    return "VH-" + hashlib.sha256(bruto).hexdigest()[:12].upper()


def _permutacao(tamanho: int, seed: int, avaliador: str) -> np.ndarray:
    bruto = hashlib.sha256(f"{seed}:{avaliador}".encode("ascii")).digest()
    seed_ordem = int.from_bytes(bruto[:8], "big")
    return np.random.default_rng(seed_ordem).permutation(tamanho)


def _hash_protocolo() -> str:
    if not PROTOCOLO_ROTULAGEM.is_file():
        raise FileNotFoundError(f"protocolo nao encontrado: {PROTOCOLO_ROTULAGEM}")
    return hashlib.sha256(PROTOCOLO_ROTULAGEM.read_bytes()).hexdigest()


def _resumo_lote(
    lote: LoteValidado,
    tamanho: int,
    seed: int,
    protocolo_hash: str,
) -> pd.DataFrame:
    completas = lote.classificacoes_completas
    ok = completas.loc[completas["status_ia"].eq("ok")].copy()
    classes = [
        _classe_ia(bool(especifico), str(direcao))
        for especifico, direcao in zip(
            ok["especifico_empresa"], ok["direcao"], strict=True
        )
    ]
    contagens = pd.Series(classes, dtype="string").value_counts().to_dict()
    total = len(completas)
    total_ok = len(ok)
    erros = total - total_ok
    abstencoes = int(ok["abster"].sum())
    linha: dict[str, object] = {
        "lote_hash": lote.lote_hash,
        **lote.identidade,
        "documentos_elegiveis": total,
        "classificacoes_ok": total_ok,
        "erros_tecnicos": erros,
        "taxa_erro_tecnico": erros / total,
        "abstencoes": abstencoes,
        "taxa_abstencao": abstencoes / total_ok,
        "tamanho_amostra": int(tamanho),
        "seed_amostra": int(seed),
        "protocolo_rotulagem": VERSAO_PROTOCOLO,
        "protocolo_rotulagem_hash": protocolo_hash,
        "desenho_amostra": DESENHO_AMOSTRA,
    }
    for classe in validacao_ia.CLASSES_CONJUNTAS:
        linha[classe] = int(contagens.get(classe, 0))
    return pd.DataFrame([linha])


def preparar_validacao(
    corpus: pd.DataFrame,
    classificacoes: pd.DataFrame,
    tamanho: int = TAMANHO_PADRAO,
    seed: int = SEED_VALIDACAO,
) -> ResultadoPreparacao:
    lote = _juntar_dados(corpus, classificacoes)
    base = lote.classificacoes_ok
    amostra = validacao_ia.selecionar_amostra_cega(base, tamanho, seed)
    ids = sorted(amostra["ID_Documento"].astype(str))
    anonimos = {documento: _id_anonimo(documento, seed) for documento in ids}
    if len(set(anonimos.values())) != len(anonimos):
        raise RuntimeError("colisao entre IDs anonimos")

    selecionados = base.set_index("ID_Documento").loc[ids].reset_index()
    planilha = pd.DataFrame(
        {
            "id_anonimo": selecionados["ID_Documento"].map(anonimos),
            "texto": selecionados["texto_llm"].astype(str),
            "especifico_empresa": "",
            "direcao": "",
        }
    )
    ordem_a = _permutacao(len(planilha), seed, "a")
    ordem_b = _permutacao(len(planilha), seed, "b")
    if len(planilha) > 1 and np.array_equal(ordem_a, ordem_b):
        ordem_b = np.roll(ordem_b, 1)
    avaliador_a = planilha.iloc[ordem_a].reset_index(drop=True)
    avaliador_b = planilha.iloc[ordem_b].reset_index(drop=True)

    posicao_a = {
        codigo: posicao
        for posicao, codigo in enumerate(avaliador_a["id_anonimo"], start=1)
    }
    posicao_b = {
        codigo: posicao
        for posicao, codigo in enumerate(avaliador_b["id_anonimo"], start=1)
    }
    chave = selecionados[
        ["ID_Documento", "especifico_empresa", "direcao", "abster"]
    ].copy()
    chave.insert(0, "id_anonimo", chave["ID_Documento"].map(anonimos))
    chave = chave.rename(
        columns={
            "especifico_empresa": "especifico_empresa_ia",
            "direcao": "direcao_ia",
            "abster": "abster_ia",
        }
    )
    chave["ordem_avaliador_a"] = chave["id_anonimo"].map(posicao_a)
    chave["ordem_avaliador_b"] = chave["id_anonimo"].map(posicao_b)
    chave["protocolo_rotulagem"] = VERSAO_PROTOCOLO
    chave["desenho_amostra"] = DESENHO_AMOSTRA
    chave["classe_ia"] = [
        _classe_ia(bool(especifico), str(direcao))
        for especifico, direcao in zip(
            chave["especifico_empresa_ia"], chave["direcao_ia"], strict=True
        )
    ]
    protocolo_hash = _hash_protocolo()
    chave["protocolo_rotulagem_hash"] = protocolo_hash
    chave["tamanho_amostra"] = int(tamanho)
    chave["seed_amostra"] = int(seed)
    chave["lote_hash"] = lote.lote_hash
    for coluna, valor in lote.identidade.items():
        chave[coluna] = valor
    chave = chave.sort_values("id_anonimo", kind="stable").reset_index(drop=True)
    resumo = _resumo_lote(lote, tamanho, seed, protocolo_hash)
    return ResultadoPreparacao(avaliador_a, avaliador_b, chave, resumo)


def _gravar_csv_atomico(tabela: pd.DataFrame, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    descritor, nome = tempfile.mkstemp(
        prefix=f".{caminho.name}.", suffix=".tmp", dir=caminho.parent
    )
    temporario = Path(nome)
    try:
        with os.fdopen(descritor, "w", encoding="utf-8-sig", newline="") as arquivo:
            tabela.to_csv(arquivo, index=False, lineterminator="\n")
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, caminho)
    except BaseException:
        temporario.unlink(missing_ok=True)
        raise


def gravar_resultado(
    resultado: ResultadoPreparacao,
    avaliador_a: Path,
    avaliador_b: Path,
    chave_interna: Path,
    resumo_interno: Path,
    *,
    sobrescrever: bool = False,
) -> None:
    destinos = [avaliador_a, avaliador_b, chave_interna, resumo_interno]
    resolvidos = [caminho.resolve() for caminho in destinos]
    if len(set(resolvidos)) != len(resolvidos):
        raise ValueError("os quatro arquivos de saida devem ser distintos")
    existentes = [caminho for caminho in destinos if caminho.exists()]
    if existentes and not sobrescrever:
        raise FileExistsError(f"arquivo de saida ja existe: {existentes[0]}")
    _gravar_csv_atomico(resultado.avaliador_a, avaliador_a)
    _gravar_csv_atomico(resultado.avaliador_b, avaliador_b)
    _gravar_csv_atomico(resultado.chave_interna, chave_interna)
    _gravar_csv_atomico(resultado.resumo_interno, resumo_interno)


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    if args.tamanho != TAMANHO_PADRAO:
        raise SystemExit(f"--tamanho deve ser {TAMANHO_PADRAO} na execucao oficial")
    corpus_path = _absoluto(args.corpus)
    classificacoes_path = _absoluto(args.classificacoes)
    avaliador_a_path = _absoluto(args.avaliador_a)
    avaliador_b_path = _absoluto(args.avaliador_b)
    chave_path = _absoluto(args.chave_interna)
    resumo_path = _absoluto(args.resumo_interno)
    resultado = preparar_validacao(
        _ler_tabela(corpus_path),
        _ler_tabela(classificacoes_path),
        tamanho=args.tamanho,
    )
    gravar_resultado(
        resultado,
        avaliador_a_path,
        avaliador_b_path,
        chave_path,
        resumo_path,
        sobrescrever=args.sobrescrever,
    )
    print(f"Amostra cega preparada: {len(resultado.chave_interna)} documentos")
    print(f"Avaliador A: {avaliador_a_path}")
    print(f"Avaliador B: {avaliador_b_path}")
    print(f"Chave interna: {chave_path}")
    print(f"Resumo interno: {resumo_path}")
    print(f"Protocolo: {PROTOCOLO_ROTULAGEM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
