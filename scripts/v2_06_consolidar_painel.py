"""Consolida o painel multi-avaliador e mede a IA contra o gold resultante."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402
from src.v2 import validacao_ia  # noqa: E402

PAINEL_PADRAO = RAIZ / "outputs/validacao_humana"
CHAVE_PADRAO = RAIZ / "data/processed/cvm_ipe/chave_validacao_humana.csv"
DESEMPATE = "humano"
AVALIADORES = ("qwen_max", "kimi_26", "gemini", "opus", "humano")


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--painel", type=Path, default=PAINEL_PADRAO)
    parser.add_argument("--chave-interna", type=Path, default=CHAVE_PADRAO)
    return parser.parse_args(argv)


def _ler_avaliador(diretorio: Path, nome: str, ordem: list[str]) -> pd.Series:
    caminho = diretorio / f"painel_{nome}.csv"
    if not caminho.is_file():
        raise FileNotFoundError(f"avaliador ausente: {caminho}")
    tabela = pd.read_csv(
        caminho, encoding="utf-8-sig", dtype=object, keep_default_na=False
    )
    faltando = set(ordem) - set(tabela["id_anonimo"])
    if faltando:
        raise ValueError(f"{nome}: faltam {len(faltando)} documentos")
    tabela = tabela.set_index("id_anonimo").loc[ordem]
    especifico = tabela["especifico_empresa"].str.strip().str.casefold().eq("true")
    direcao = tabela["direcao"].str.strip().str.casefold()
    invalidas = sorted(set(direcao) - set(validacao_ia.DIRECOES))
    if invalidas:
        raise ValueError(f"{nome}: direcao invalida {invalidas}")
    if (~especifico & direcao.ne("neutra")).any():
        raise ValueError(f"{nome}: evento nao especifico com direcao nao neutra")
    return pd.Series(
        [
            f"especifico_{d}" if e else "nao_especifico"
            for e, d in zip(especifico, direcao, strict=True)
        ],
        index=ordem,
        name=nome,
    )


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    execucao = criar_execucao("v2_painel")
    log = configurar_log(execucao, "v2_06_consolidar_painel")
    config = {
        "painel": str(args.painel),
        "avaliadores": list(AVALIADORES),
        "desempate": DESEMPATE,
        "suporte_minimo_classe": validacao_ia.SUPORTE_MINIMO_CLASSE,
        "limiar_macro_f1": validacao_ia.LIMIAR_MACRO_F1,
        "limiar_kappa": validacao_ia.LIMIAR_KAPPA,
    }
    try:
        chave = pd.read_csv(args.chave_interna, encoding="utf-8-sig")
        ordem = chave["id_anonimo"].tolist()
        votos = pd.DataFrame(
            {nome: _ler_avaliador(args.painel, nome, ordem) for nome in AVALIADORES}
        )
        votos.index.name = "id_anonimo"

        fleiss = validacao_ia.calcular_fleiss_kappa(votos)
        gold = validacao_ia.consolidar_painel(votos, desempate=DESEMPATE)

        predicoes = pd.DataFrame(
            {
                "id_anonimo": chave["id_anonimo"],
                "especifico_empresa": chave["especifico_empresa_ia"].astype(bool),
                "direcao": chave["direcao_ia"].astype(str),
                "abster": chave["abster_ia"].astype(bool),
            }
        )
        gold_ia = pd.DataFrame(
            {
                "id_anonimo": gold.index,
                "especifico_empresa": gold["especifico_empresa"].to_numpy(),
                "direcao": gold["direcao"].to_numpy(),
            }
        )
        avaliacao = validacao_ia.avaliar_ia_contra_gold(
            predicoes, gold_ia, fleiss, coluna_id="id_anonimo"
        )

        destino = execucao.tabelas
        destino.mkdir(parents=True, exist_ok=True)
        votos.to_csv(destino / "painel_votos.csv", encoding="utf-8-sig")
        gold.to_csv(destino / "painel_gold.csv", encoding="utf-8-sig")
        avaliacao.matriz_confusao.to_csv(
            destino / "matriz_confusao.csv", encoding="utf-8-sig"
        )
        avaliacao.metricas_por_classe.to_csv(
            destino / "metricas_por_classe.csv", index=False, encoding="utf-8-sig"
        )
        concordancia = pd.DataFrame(
            {
                x: {y: float(votos[x].eq(votos[y]).mean()) for y in AVALIADORES}
                for x in AVALIADORES
            }
        )
        concordancia.to_csv(destino / "concordancia_par_a_par.csv", encoding="utf-8-sig")

        origens = gold["origem_rotulo"].value_counts().to_dict()
        resumo = pd.DataFrame(
            [
                {
                    "avaliadores": len(AVALIADORES),
                    "desempate": DESEMPATE,
                    "fleiss_kappa": fleiss,
                    "limiar_kappa": validacao_ia.LIMIAR_KAPPA,
                    "aprovado_kappa": avaliacao.aprovado_kappa,
                    "macro_f1": avaliacao.macro_f1,
                    "macro_f1_com_suporte": avaliacao.macro_f1_com_suporte,
                    "classes_subdimensionadas": ";".join(
                        avaliacao.classes_subdimensionadas
                    ),
                    "limiar_macro_f1": validacao_ia.LIMIAR_MACRO_F1,
                    "aprovado_macro_f1": avaliacao.aprovado_macro_f1,
                    "cobertura": avaliacao.cobertura,
                    "taxa_abstencao": avaliacao.taxa_abstencao,
                    "gold_unanime": origens.get("unanime", 0),
                    "gold_maioria": origens.get("maioria", 0),
                    "gold_desempate": origens.get("desempate", 0),
                    "aprovado": avaliacao.aprovado,
                }
            ]
        )
        resumo.to_csv(destino / "resumo_painel.csv", index=False, encoding="utf-8-sig")

        gravar_manifesto(
            execucao,
            config,
            arquivos_dados=[
                args.painel / f"painel_{nome}.csv" for nome in AVALIADORES
            ],
            status="concluida",
            extras={
                "fleiss_kappa": float(fleiss),
                "macro_f1": float(avaliacao.macro_f1),
                "macro_f1_com_suporte": float(avaliacao.macro_f1_com_suporte),
                "aprovado": bool(avaliacao.aprovado),
                "origens_gold": origens,
                "diretorio_tabelas": str(destino),
            },
        )
        log.info(
            "Fleiss %.4f | macro-F1 %.4f (gate %.4f) | gate %s",
            fleiss,
            avaliacao.macro_f1,
            avaliacao.macro_f1_com_suporte,
            "aprovado" if avaliacao.aprovado else "reprovado",
        )
        return 0 if avaliacao.aprovado else 2
    except Exception as exc:
        log.exception("falha na consolidacao do painel")
        gravar_manifesto(execucao, config, status="falhou", extras={"erro": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
