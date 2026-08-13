"""Backtest exploratório da V2: eventos classificados roteados pela rede V1.

Exploratório por decisão registrada em docs/ESPECIFICACAO_V2.md: o gate da
validação reprovou. O resultado não sustenta afirmação sobre a estratégia.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import dados, metricas, qualidade_dados  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402
from src.v2 import backtest_eventos, eventos  # noqa: E402

CLASSIFICACOES = RAIZ / "data/processed/cvm_ipe/classificacoes_ia.parquet"
HORIZONTES = (3, 1, 5)
SEED_PLACEBO = 20260812


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rede", type=Path, required=True)
    parser.add_argument("--classificacoes", type=Path, default=CLASSIFICACOES)
    return parser.parse_args(argv)


def _sinais_da_ia(caminho: Path) -> pd.DataFrame:
    """Documentos válidos viram um sinal por líder-dia; conflito abstém."""
    base = pd.read_parquet(caminho)
    ok = base[base["status_ia"].eq("ok")].copy()
    ok["classificacao"] = [
        direcao if bool(especifico) else "neutra"
        for especifico, direcao in zip(
            ok["especifico_empresa"], ok["direcao"], strict=True
        )
    ]
    agregado = eventos.agregar_sinais_eventos(
        ok, coluna_classificacao="classificacao", coluna_data="Sessao_Disponivel"
    )
    return agregado.rename(columns={"sinal": "direcao"})


def _pares_estruturais(caminho: Path) -> pd.DataFrame:
    """Todos os pares líder→seguidora da rede V1, sem filtro de beta ou FDR."""
    rede = pd.read_csv(caminho)
    estruturais = rede[rede["direcao"].eq("lider_para_seguidora")]
    return estruturais[["janela", "lider", "seguidora", "setor", "subsetor"]]


def _pares_placebo(pares: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Troca cada seguidora por outra do mesmo subsetor na mesma janela."""
    rng = np.random.default_rng(seed)
    linhas = []
    for (janela, subsetor), grupo in pares.groupby(["janela", "subsetor"]):
        candidatas = sorted(set(grupo["seguidora"]))
        for _, linha in grupo.iterrows():
            alternativas = [c for c in candidatas if c != linha["seguidora"]]
            if not alternativas:
                continue
            linhas.append(
                {
                    "janela": janela,
                    "lider": linha["lider"],
                    "seguidora": alternativas[rng.integers(len(alternativas))],
                }
            )
    return pd.DataFrame(linhas, columns=["janela", "lider", "seguidora"])


def _sinais_embaralhados(sinais: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Mantém datas e líderes, permuta as direções: rede sem informação da IA."""
    embaralhado = sinais.copy()
    rng = np.random.default_rng(seed)
    embaralhado["direcao"] = rng.permutation(embaralhado["direcao"].to_numpy())
    return embaralhado


def _sinais_no_proprio_lider(sinais: pd.DataFrame) -> pd.DataFrame:
    """Cada líder é sua própria seguidora: IA sem rede."""
    return (
        sinais[["janela", "lider"]]
        .drop_duplicates()
        .assign(seguidora=lambda t: t["lider"])
    )


def _resumo(nome: str, resultado, holding: int) -> dict[str, object]:
    pnl = resultado.pnl_diario["pnl_liquido"]
    linha: dict[str, object] = {"braco": nome, "h": holding}
    linha.update(metricas.resumo_estrategia(pnl))
    if len(pnl.dropna()) > 30:
        boot = metricas.block_bootstrap(pnl, holding_dias=holding)
        linha["ic_inferior"] = boot.get("ic_inferior")
        linha["ic_superior"] = boot.get("ic_superior")
        linha["cauda_inferior"] = boot.get("p_unilateral")
        nw = metricas.newey_west(pnl)
        linha["media_diaria"] = nw.get("media")
        linha["t_newey_west"] = nw.get("t")
    linha.update(
        {
            "operacoes": int(len(resultado.operacoes)),
            "posicao_dias": int(len(resultado.posicoes)),
        }
    )
    return linha


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    execucao = criar_execucao("v2_backtest")
    log = configurar_log(execucao, "v2_07_backtest_eventos")
    cfg = CONFIG_PADRAO
    config = {
        "rede": str(args.rede),
        "classificacoes": str(args.classificacoes),
        "horizontes": list(HORIZONTES),
        "seed_placebo": SEED_PLACEBO,
        "carater": "exploratorio_gate_reprovado",
    }
    try:
        sinais = _sinais_da_ia(args.classificacoes)
        ativos = sinais[~sinais["abstencao"].astype(bool)]
        log.info(
            "sinais: %d lider-dia, %d ativos, %d abstencoes",
            len(sinais),
            len(ativos),
            len(sinais) - len(ativos),
        )

        pares = _pares_estruturais(args.rede)
        log.info("pares estruturais: %d em %d janelas", len(pares), pares["janela"].nunique())

        cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
        calendario = qualidade_dados.calendario_pregoes(cot)

        pares_rede = pares[["janela", "lider", "seguidora"]].drop_duplicates()
        bracos = {
            "ia_mais_rede": (ativos, pares_rede),
            "ia_sem_rede": (ativos, _sinais_no_proprio_lider(ativos)),
            "rede_sem_ia": (_sinais_embaralhados(ativos, SEED_PLACEBO), pares_rede),
            "seguidora_aleatoria": (ativos, _pares_placebo(pares, SEED_PLACEBO)),
        }

        resumos: list[dict[str, object]] = []
        destino = execucao.tabelas
        destino.mkdir(parents=True, exist_ok=True)
        for nome, (sinal, par) in bracos.items():
            for holding in HORIZONTES:
                if nome != "ia_mais_rede" and holding != HORIZONTES[0]:
                    continue
                resultado = backtest_eventos.rodar_backtest_eventos(
                    sinal, par, calendario, cotahist=cot, h=holding
                )
                resumos.append(_resumo(nome, resultado, holding))
                if nome == "ia_mais_rede" and holding == HORIZONTES[0]:
                    resultado.pnl_diario.to_csv(
                        destino / "pnl_diario_principal.csv",
                        encoding="utf-8-sig",
                    )
                    resultado.operacoes.to_csv(
                        destino / "operacoes_principal.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    resultado.diagnosticos.to_csv(
                        destino / "diagnosticos_principal.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                log.info(
                    "%s h=%d | retorno %.4f | operacoes %d",
                    nome,
                    holding,
                    resumos[-1]["retorno_total"],
                    resumos[-1]["operacoes"],
                )

        tabela = pd.DataFrame(resumos)
        tabela.to_csv(destino / "resumo_bracos.csv", index=False, encoding="utf-8-sig")
        sinais.to_csv(destino / "sinais_agregados.csv", index=False, encoding="utf-8-sig")

        principal = tabela[
            tabela["braco"].eq("ia_mais_rede") & tabela["h"].eq(HORIZONTES[0])
        ].iloc[0]
        gravar_manifesto(
            execucao,
            config,
            arquivos_dados=[args.rede, args.classificacoes],
            status="concluida",
            extras={
                "sinais_ativos": int(len(ativos)),
                "retorno_total_principal": float(principal["retorno_total"]),
                "sharpe_principal": float(principal["sharpe"]),
                "diretorio_tabelas": str(destino),
            },
        )
        log.info("tabelas em %s", destino)
        return 0
    except Exception as exc:
        log.exception("falha no backtest da V2")
        gravar_manifesto(execucao, config, status="falhou", extras={"erro": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
