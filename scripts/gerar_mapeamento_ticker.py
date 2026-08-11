"""
Gera e atualiza data/reference/mudancas_ticker.csv.

Tres cuidados sustentam a tabela:

  1. linhagem sugerida e linhagem confirmada sao coisas separadas.
     suggested_corporate_lineage_id vem do grafo completo de candidatos e e so
     pista; corporate_lineage_id e reconstruida so com as arestas confirmadas
     na revisao. Assim um falso positivo como ESTC3 -> ALSO3 nao conecta Yduqs e
     Allos sozinho, e ninguem precisa reatribuir linhagem a mao - que e onde
     erro humano entraria.

  2. campo preenchido por pessoa nunca e sobrescrito, mesmo em linha pendente.
     Reexecutar o pipeline nao pode apagar conferencia documental.

  3. diagnostico desatualizado e sinalizado, nao escondido. Os diagnosticos sao
     regerados sempre e, se mudarem depois de uma revisao, a linha recebe
     needs_revalidation=sim. O status decidido continua intacto: quem decide e
     a pessoa.

Nao aplica continuidade de serie nenhuma. A tabela e insumo de revisao.

Uso:
    .venv\\Scripts\\python.exe scripts\\gerar_mapeamento_ticker.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.qualidade_dados import DETECTOR_VERSAO  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "reference" / "mudancas_ticker.csv"
CHAVE = ["ticker_antigo", "ticker_novo"]

# Preenchidas por pessoa. Nunca sobrescritas quando tiverem valor.
COLUNAS_MANUAIS = [
    "canonical_instrument_id", "tipo_evento", "proporcao_conversao",
    "tratar_retorno_fronteira", "continuidade_autorizada", "fonte",
    "data_confirmacao", "revisor", "status", "observacoes",
    "reviewed_detector_version",
]

# Entram no hash de diagnostico, porque sao os numeros que sustentam a decisao:
# se algum mudar depois da revisao, ela precisa ser reconferida.
COLUNAS_DIAGNOSTICO = [
    "nome_antigo", "nome_novo", "isin_antigo", "isin_novo",
    "especie_antiga", "especie_nova", "ultimo_pregao_antigo",
    "primeiro_pregao_novo", "intervalo_pregoes", "razao_precos",
    "voltot_mediano_antes", "voltot_mediano_depois",
    "melhor_posicao_antigo", "melhor_posicao_novo", "criterio_relevancia",
    # a validacao contra o COTAHIST tambem sustenta a decisao
    "resultado_validacao_yf", "corr_retornos", "erro_retorno_mediano",
    "razao_precos_mediana", "dispersao_razao_precos",
]

# Colunas de validacao, copiadas da tabela de revisao a cada execucao.
COLUNAS_VALIDACAO = [
    "yf_ticker_consultado", "yf_primeira_data", "yf_ultima_data",
    "n_datas_sobrepostas", "n_datas_eventos_excluidas", "corr_retornos",
    "erro_retorno_mediano", "erro_retorno_p95", "razao_precos_mediana",
    "dispersao_razao_precos", "resultado_validacao_yf",
    "motivo_validacao_yf", "validation_version",
]

# Status para quem passou na conferencia numerica mas ainda nao tem confirmacao
# documental. Continua sendo estado aberto: numero nao autoriza continuidade.
CATEGORIAS_CONSISTENTES = ("retornos_consistentes_sem_quebra_escala",
                           "retornos_consistentes_com_quebra_escala")
STATUS_NUMERICO = "consistente_numericamente_pendente_documentacao"
STATUS_ABERTOS = {"pendente", "pendente_prioridade_baixa", STATUS_NUMERICO, ""}

# Tipos de evento que representam relacao societaria real. Falso positivo e caso
# nao confirmado nao geram aresta de linhagem.
EVENTOS_COM_LINHAGEM = {"rename_1to1", "conversao_classe", "fusao_incorporacao"}


def _union_find(pares: list[tuple[str, str]]) -> dict[str, str]:
    """Componentes conectados. O rotulo do grupo e o menor ticker dele."""
    pai: dict[str, str] = {}

    def find(x: str) -> str:
        pai.setdefault(x, x)
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    for a, b in pares:
        ra, rb = find(a), find(b)
        if ra != rb:
            pai[max(ra, rb)] = min(ra, rb)
    return {t: find(t) for t in pai}


def _hash_diagnostico(row: pd.Series) -> str:
    bruto = "|".join(str(row.get(c, "")) for c in COLUNAS_DIAGNOSTICO)
    return hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:16]


def _prioridade(row: pd.Series) -> str:
    pos = [p for p in (row["melhor_posicao_antigo"], row["melhor_posicao_novo"])
           if pd.notna(p)]
    melhor = min(pos) if pos else float("inf")
    if melhor <= 50 or "ponta_curta" in str(row["criterio_relevancia"]):
        return "alta"
    return "media" if melhor <= 100 else "baixa"


def main() -> int:
    arquivos = sorted(RAIZ.glob("outputs/runs/*_etapa1/tabelas/revisao_mudanca_ticker.csv"))
    if not arquivos:
        print("Nenhuma tabela de revisao. Rode antes:")
        print("  .venv\\Scripts\\python.exe scripts\\01_baixar_e_auditar_dados.py")
        return 1
    rev = pd.read_csv(arquivos[-1])
    print(f"lendo {len(rev)} candidatos de {arquivos[-1].parent.parent.name}")

    df = pd.DataFrame({
        "ticker_antigo": rev["ticker_antigo"],
        "ticker_novo": rev["ticker_novo"],
        "nome_antigo": rev["nome_antigo"],
        "nome_novo": rev["nome_novo"],
        "isin_antigo": rev["isin_antigo"],
        "isin_novo": rev["isin_novo"],
        "especie_antiga": rev["especie_antiga"],
        "especie_nova": rev["especie_nova"],
        "ultimo_pregao_antigo": rev["ultimo_pregao_antigo"],
        "primeiro_pregao_novo": rev["primeiro_pregao_novo"],
        "intervalo_pregoes": rev["intervalo_pregoes"],
        "razao_precos": rev["razao_precos"],
        "voltot_mediano_antes": rev["voltot_mediano_antes"],
        "voltot_mediano_depois": rev["voltot_mediano_depois"],
        "melhor_posicao_antigo": rev["melhor_posicao_antigo"],
        "melhor_posicao_novo": rev["melhor_posicao_novo"],
        "criterio_relevancia": rev["criterio_relevancia"],
        "metodo_deteccao": rev["confianca_deteccao"],
    })
    for col in COLUNAS_VALIDACAO:
        df[col] = rev[col] if col in rev.columns else ""
    df["prioridade"] = df.apply(_prioridade, axis=1)

    # Pista automatica vinda do grafo completo. Nao autoriza nada.
    sugerida = _union_find(list(zip(df["ticker_antigo"], df["ticker_novo"])))
    df["suggested_corporate_lineage_id"] = df["ticker_antigo"].map(sugerida)

    # Defaults dos campos manuais.
    df["canonical_instrument_id"] = ""
    df["corporate_lineage_id"] = ""      # reconstruida so com arestas aprovadas
    df["tipo_evento"] = ""
    df["proporcao_conversao"] = ""
    # pendente_revisao, e nao nao_aplicavel: "nao se aplica" e uma decisao
    # (falso positivo, relacao rejeitada, evento que nao autoriza continuidade)
    # e nao um estado inicial.
    df["tratar_retorno_fronteira"] = "pendente_revisao"
    df["continuidade_autorizada"] = "nao"
    df["fonte"] = ""
    df["data_confirmacao"] = ""
    df["revisor"] = ""
    # A validacao numerica qualifica o estado de espera, mas nao decide:
    # STATUS_NUMERICO diz "os numeros batem, falta o documento".
    df["status"] = df["prioridade"].map(
        {"alta": "pendente", "media": "pendente",
         "baixa": "pendente_prioridade_baixa"})
    if "resultado_validacao_yf" in df.columns:
        df.loc[df["resultado_validacao_yf"].isin(CATEGORIAS_CONSISTENTES),
               "status"] = STATUS_NUMERICO
    df["observacoes"] = ""
    df.loc[df["especie_antiga"] != df["especie_nova"], "observacoes"] = (
        "especie difere entre as pontas: nao pode ser rename_1to1 sem "
        "confirmar proporcao e direitos")
    df["detector_version"] = DETECTOR_VERSAO
    df["reviewed_detector_version"] = ""
    df["needs_revalidation"] = "nao"
    df["diagnostic_hash"] = df.apply(_hash_diagnostico, axis=1)

    # Mescla com o arquivo existente.
    preservadas = revalidar = orfas_mantidas = 0
    if DESTINO.exists():
        antigo = pd.read_csv(DESTINO, dtype=str).fillna("")
        ant = antigo.set_index(CHAVE)
        novo = df.set_index(CHAVE)
        comuns = novo.index.intersection(ant.index)

        for chave in comuns:
            linha_ant = ant.loc[chave]
            revisada = str(linha_ant.get("status", "")) not in STATUS_ABERTOS

            # Preserva o que foi escrito por pessoa, comparando contra o valor
            # semeado e nao contra vazio: varias colunas nascem com default
            # nao-vazio (status=pendente, continuidade_autorizada=nao,
            # tratar_retorno_fronteira=pendente_revisao). Sem isso o script
            # contaria os proprios defaults como decisao humana - num arquivo
            # intocado, 127 campos "preservados" em vez de zero.
            for col in COLUNAS_MANUAIS:
                valor = str(linha_ant.get(col, "")).strip()
                semeado = str(novo.loc[chave, col]).strip()
                if valor and valor != semeado:
                    novo.loc[chave, col] = valor
                    preservadas += 1

            # Diagnostico que mudou depois da revisao e sinalizado, sem mexer no
            # status decidido.
            hash_ant = str(linha_ant.get("diagnostic_hash", "")).strip()
            versao_rev = str(linha_ant.get("reviewed_detector_version", "")).strip()
            if revisada:
                mudou_diag = bool(hash_ant) and hash_ant != novo.loc[chave, "diagnostic_hash"]
                mudou_ver = bool(versao_rev) and versao_rev != DETECTOR_VERSAO
                if mudou_diag or mudou_ver:
                    novo.loc[chave, "needs_revalidation"] = "sim"
                    revalidar += 1
                elif not versao_rev:
                    # Revisada sem versao registrada: carimba a atual.
                    novo.loc[chave, "reviewed_detector_version"] = DETECTOR_VERSAO

        df = novo.reset_index()

        # Linha ja decidida que sumiu da deteccao permanece no arquivo.
        decididas = ant[~ant["status"].isin(STATUS_ABERTOS)]
        orfas = decididas.index.difference(novo.index)
        if len(orfas):
            extra = decididas.loc[orfas].reset_index()
            extra["needs_revalidation"] = "sim"
            extra["observacoes"] = (
                extra.get("observacoes", "").astype(str)
                + " | saiu da deteccao atual, decisao preservada")
            df = pd.concat([df, extra], ignore_index=True)
            orfas_mantidas = len(orfas)

    # Linhagem confirmada, so com arestas de status confirmado.
    aprovadas = df[
        (df["status"] == "confirmado")
        & (df["tipo_evento"].isin(EVENTOS_COM_LINHAGEM))
    ]
    confirmada = _union_find(list(zip(aprovadas["ticker_antigo"],
                                      aprovadas["ticker_novo"])))
    df["corporate_lineage_id"] = df["ticker_antigo"].map(confirmada).fillna("")

    ordem = {"alta": 0, "media": 1, "baixa": 2}
    df = (df.assign(_o=df["prioridade"].map(ordem))
          .sort_values(["_o", "melhor_posicao_antigo", "melhor_posicao_novo"],
                       na_position="last")
          .drop(columns="_o").reset_index(drop=True))
    df.to_csv(DESTINO, index=False, encoding="utf-8")

    print(f"gravado: {DESTINO.relative_to(RAIZ)}  ({len(df)} linhas)")
    print(f"  detector           : {DETECTOR_VERSAO} (congelado)")
    print(f"  campos preservados : {preservadas}")
    print(f"  needs_revalidation : {revalidar}")
    if orfas_mantidas:
        print(f"  decisoes orfas mantidas: {orfas_mantidas}")
    for p in ("alta", "media", "baixa"):
        print(f"  prioridade {p:5}   : {int((df['prioridade'] == p).sum())}")
    print(f"  linhagem confirmada: {int((df['corporate_lineage_id'] != '').sum())} "
          "linhas (so arestas aprovadas)")

    if "resultado_validacao_yf" in df.columns:
        print("\n  --- validacao numerica contra o COTAHIST ---")
        print("  (evidencia e prioridade; nao autoriza continuidade)")
        for cat, n in df["resultado_validacao_yf"].value_counts().items():
            print(f"    {cat:34} {n:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
