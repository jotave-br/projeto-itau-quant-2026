"""Baixa os metadados IPE e separa Fatos Relevantes das líderes da V2."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import (  # noqa: E402
    configurar_log,
    criar_execucao,
    gravar_manifesto,
)
from src.v2 import cvm_ipe  # noqa: E402


RAIZ = Path(__file__).resolve().parent.parent
REFERENCIA_LIDERES = RAIZ / "data" / "reference" / "lideres_v2_cvm.csv"
DIR_PROCESSADO = RAIZ / "data" / "processed" / "cvm_ipe"
FATOS_PROCESSADOS = DIR_PROCESSADO / "fatos_relevantes_lideres.parquet"

COLUNAS_REFERENCIA = (
    "emissor_id",
    "codigo_cvm",
    "cnpj",
    "denominacao_cvm",
    "validade_inicio",
    "validade_fim",
    "status_revisao",
)


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ano-inicial",
        type=int,
        default=CONFIG_PADRAO.periodo.inicio.year,
    )
    parser.add_argument("--ano-final", type=int, default=date.today().year)
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="baixa novamente mesmo quando o ZIP em cache está íntegro",
    )
    return parser.parse_args()


def _carregar_lideres(caminho: Path = REFERENCIA_LIDERES) -> pd.DataFrame:
    lideres = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    faltantes = sorted(set(COLUNAS_REFERENCIA) - set(lideres.columns))
    if faltantes:
        raise ValueError(f"referência de líderes sem colunas: {faltantes}")
    if lideres.empty:
        raise ValueError("referência de líderes está vazia")
    obrigatorias_preenchidas = (
        "emissor_id",
        "codigo_cvm",
        "cnpj",
        "denominacao_cvm",
        "validade_inicio",
        "status_revisao",
    )
    for coluna in obrigatorias_preenchidas:
        vazias = lideres[coluna].str.strip().eq("")
        if vazias.any():
            raise ValueError(
                f"referência de líderes tem {int(vazias.sum())} {coluna} vazio(s)"
            )
    if not lideres["status_revisao"].eq("confirmado").all():
        raise ValueError("toda líder da V2 precisa estar confirmada")

    lideres = lideres.copy()
    lideres["codigo_cvm"] = lideres["codigo_cvm"].str.lstrip("0").replace("", "0")
    for coluna in ("emissor_id", "codigo_cvm", "cnpj"):
        duplicada = lideres[coluna].duplicated(False)
        if duplicada.any():
            valores = lideres.loc[duplicada, coluna].unique().tolist()
            raise ValueError(f"{coluna} duplicado na referência: {valores}")
    lideres["validade_inicio"] = pd.to_datetime(
        lideres["validade_inicio"], format="%Y-%m-%d", errors="raise"
    )
    lideres["validade_fim"] = pd.to_datetime(
        lideres["validade_fim"].replace("", pd.NA),
        format="%Y-%m-%d",
        errors="raise",
    )
    intervalo_invalido = (
        lideres["validade_fim"].notna()
        & lideres["validade_fim"].lt(lideres["validade_inicio"])
    )
    if intervalo_invalido.any():
        raise ValueError("validade_fim anterior a validade_inicio na referência")
    return lideres


def _juntar_lideres(fatos: pd.DataFrame, lideres: pd.DataFrame) -> pd.DataFrame:
    fatos = fatos.copy()
    fatos["codigo_cvm_norm"] = (
        fatos["Codigo_CVM"].astype("string").str.lstrip("0").replace("", "0")
    )
    saida = fatos.merge(
        lideres,
        left_on="codigo_cvm_norm",
        right_on="codigo_cvm",
        how="inner",
        validate="many_to_one",
    )
    divergente = saida["CNPJ_Companhia"].ne(saida["cnpj"])
    if divergente.any():
        amostra = saida.loc[
            divergente,
            ["emissor_id", "Codigo_CVM", "CNPJ_Companhia", "cnpj"],
        ].head(5)
        raise ValueError(
            "código CVM encontrou CNPJ diferente da referência:\n"
            + amostra.to_string(index=False)
        )

    data_entrega = pd.to_datetime(saida["Data_Entrega"], format="%Y-%m-%d")
    vigente = data_entrega.ge(saida["validade_inicio"]) & (
        saida["validade_fim"].isna() | data_entrega.le(saida["validade_fim"])
    )
    return saida.loc[vigente].drop(columns="codigo_cvm_norm").reset_index(drop=True)


def main() -> int:
    args = _argumentos()
    if args.ano_inicial > args.ano_final:
        raise SystemExit("--ano-inicial não pode ser posterior a --ano-final")

    execucao = criar_execucao("v2_ipe")
    log = configurar_log(execucao, "v2_01_ipe")
    cfg_manifesto = {
        "ano_inicial": args.ano_inicial,
        "ano_final": args.ano_final,
        "categoria": cvm_ipe.CATEGORIA_FATO_RELEVANTE,
        "forcar_download": args.forcar,
        "fonte": cvm_ipe.URL_IPE_ANUAL,
    }
    artefatos = []
    fatos_por_ano = []
    resumos = []
    anomalias_referencia = []

    try:
        lideres = _carregar_lideres()
        log.info("%d líderes com código CVM e CNPJ confirmados", len(lideres))

        for ano in range(args.ano_inicial, args.ano_final + 1):
            artefato = cvm_ipe.baixar_zip_ipe(ano, forcar=args.forcar)
            artefatos.append(artefato.caminho)
            bruto = cvm_ipe.ler_zip_ipe(artefato.caminho)
            fatos = cvm_ipe.adicionar_ids(cvm_ipe.filtrar_fatos_relevantes(bruto))

            anos_entrega = pd.to_datetime(
                fatos["Data_Entrega"], format="%Y-%m-%d"
            ).dt.year
            fora_do_arquivo = anos_entrega.ne(ano)
            if fora_do_arquivo.any():
                raise ValueError(
                    f"ZIP {ano} contém {int(fora_do_arquivo.sum())} fatos "
                    "com Data_Entrega de outro ano"
                )

            fatos["Ano_Arquivo"] = ano
            ano_referencia = fatos["Data_Referencia"].str.slice(0, 4).astype(int)
            referencia_fora_horizonte = ~ano_referencia.between(1900, ano + 1)
            if referencia_fora_horizonte.any():
                anomalias = fatos.loc[
                    referencia_fora_horizonte,
                    [
                        cvm_ipe.COLUNA_ID,
                        "Codigo_CVM",
                        "Nome_Companhia",
                        "Data_Referencia",
                        "Data_Entrega",
                        "Link_Download",
                    ],
                ].copy()
                anomalias["Ano_Arquivo"] = ano
                anomalias_referencia.append(anomalias)
                log.warning(
                    "%d | %d Data_Referencia fora do horizonte; valor preservado",
                    ano,
                    len(anomalias),
                )
            fatos_lideres = _juntar_lideres(fatos, lideres)
            fatos_por_ano.append(fatos_lideres)
            resumos.append(
                {
                    "ano": ano,
                    "linhas_ipe": len(bruto),
                    "fatos_relevantes": len(fatos),
                    "fatos_das_lideres": len(fatos_lideres),
                    "lideres_com_fato": fatos_lideres["emissor_id"].nunique(),
                    "data_referencia_fora_horizonte": int(
                        referencia_fora_horizonte.sum()
                    ),
                    "zip_sha256": artefato.sha256,
                    "zip_cache": artefato.de_cache,
                }
            )
            log.info(
                "%d | %d linhas | %d fatos | %d das líderes%s",
                ano,
                len(bruto),
                len(fatos),
                len(fatos_lideres),
                " | cache" if artefato.de_cache else "",
            )

        todos = pd.concat(fatos_por_ano, ignore_index=True)
        if todos[cvm_ipe.COLUNA_ID].duplicated().any():
            raise ValueError("ID_Documento duplicado entre os ZIPs anuais")

        DIR_PROCESSADO.mkdir(parents=True, exist_ok=True)
        todos.to_parquet(FATOS_PROCESSADOS, index=False)
        resumo = pd.DataFrame(resumos)
        resumo.to_csv(
            execucao.tabelas / "cvm_ipe_resumo_ano.csv",
            index=False,
            encoding="utf-8",
        )
        if anomalias_referencia:
            pd.concat(anomalias_referencia, ignore_index=True).to_csv(
                execucao.tabelas / "cvm_ipe_anomalias_data_referencia.csv",
                index=False,
                encoding="utf-8",
            )
        cobertura = (
            todos.groupby("emissor_id", as_index=False)
            .agg(
                documentos=(cvm_ipe.COLUNA_ID, "size"),
                primeira_entrega=("Data_Entrega", "min"),
                ultima_entrega=("Data_Entrega", "max"),
            )
            .sort_values("emissor_id")
        )
        cobertura.to_csv(
            execucao.tabelas / "cvm_ipe_cobertura_lider.csv",
            index=False,
            encoding="utf-8",
        )
        gravar_manifesto(
            execucao,
            cfg_manifesto,
            arquivos_dados=[REFERENCIA_LIDERES, *artefatos],
            status="concluida",
            extras={
                "arquivo_processado": str(FATOS_PROCESSADOS.relative_to(RAIZ)),
                "documentos": int(len(todos)),
                "lideres": int(todos["emissor_id"].nunique()),
            },
        )
        log.info("%d documentos salvos em %s", len(todos), FATOS_PROCESSADOS)
        return 0
    except Exception as exc:
        log.exception("coleta IPE falhou: %s", exc)
        gravar_manifesto(
            execucao,
            cfg_manifesto,
            arquivos_dados=[REFERENCIA_LIDERES, *artefatos],
            status="falhou",
            extras={"erro": str(exc)},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
