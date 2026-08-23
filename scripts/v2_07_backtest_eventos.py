"""Backtest exploratorio da V2: eventos classificados roteados pela rede V1.

Exploratorio por decisao registrada em docs/ESPECIFICACAO_V2.md: o gate da
validacao reprovou. O resultado nao sustenta afirmacao sobre a estrategia.
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

from src import dados, metricas, qualidade_dados, retornos  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402
from src.v2 import backtest_eventos, eventos, universo as universo_v2  # noqa: E402

CLASSIFICACOES = RAIZ / "data/processed/cvm_ipe/classificacoes_ia.parquet"
HORIZONTES = (3, 1, 5)
SEED_PLACEBO = 20260812


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rede", type=Path, required=True)
    parser.add_argument("--universo", type=Path, required=True)
    parser.add_argument("--classificacoes", type=Path, default=CLASSIFICACOES)
    parser.add_argument(
        "--n-placebos",
        type=int,
        default=CONFIG_PADRAO.placebo.n_embaralhamentos,
    )
    return parser.parse_args(argv)


def _sinais_da_ia(caminho: Path) -> pd.DataFrame:
    """Documentos validos viram um sinal por lider-dia; conflito abstem."""
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
    """Pares estruturais da faixa top 20 pre-registrada na V1."""
    rede = universo_v2.carregar_rede_top20(caminho)
    estruturais = rede[rede["direcao"].eq("lider_para_seguidora")]
    return estruturais[["janela", "lider", "seguidora", "setor", "subsetor"]]


def _pares_placebo(
    pares: pd.DataFrame,
    universo_top20: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Randomiza seguidoras top 20 no setor, preservando todas as arestas."""
    rng = np.random.default_rng(seed)
    linhas: list[dict[str, str]] = []
    base = pares.sort_values(["janela", "lider", "seguidora"], kind="stable")
    for (janela, lider), grupo in base.groupby(["janela", "lider"], sort=True):
        setores = grupo["setor"].dropna().astype(str).unique().tolist()
        if len(setores) != 1:
            raise ValueError(f"setor ambiguo para {janela}/{lider}: {setores}")
        candidatas = sorted(
            set(
                universo_top20.loc[
                    universo_top20["janela"].eq(janela)
                    & universo_top20["setor"].astype(str).eq(setores[0]),
                    "CODNEG",
                ].astype(str)
            )
            - {str(lider)}
        )
        n = len(grupo)
        if len(candidatas) < n:
            raise ValueError(
                f"placebo sem {n} candidatas unicas para {janela}/{lider} "
                f"no setor {setores[0]} (encontradas {len(candidatas)})"
            )
        sorteadas = rng.choice(candidatas, size=n, replace=False)
        for seguidora in sorteadas:
            linhas.append(
                {
                    "janela": str(janela),
                    "lider": str(lider),
                    "seguidora": str(seguidora),
                }
            )
    placebo = pd.DataFrame(linhas, columns=["janela", "lider", "seguidora"])
    if len(placebo) != len(base) or placebo.duplicated().any():
        raise RuntimeError("placebo nao preservou uma aresta unica por par original")
    if placebo["lider"].eq(placebo["seguidora"]).any():
        raise RuntimeError("placebo gerou autorrelacao lider=seguidora")
    return placebo


def _sinais_rede_sem_ia(
    pares: pd.DataFrame,
    retornos_lideres: pd.DataFrame,
    calendario: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aplica diariamente a regra de sinal da V1, sem usar Fato Relevante."""
    linhas: list[dict[str, object]] = []
    cal = pd.DatetimeIndex(calendario)
    proxima = dict(zip(cal[:-1], cal[1:], strict=True))
    lideres_janela = pares[["janela", "lider"]].drop_duplicates()
    for janela, grupo in lideres_janela.groupby("janela", sort=True):
        inicio = pd.Timestamp(f"{janela}-01")
        fim = inicio + pd.DateOffset(months=3)
        datas = cal[(cal >= inicio) & (cal < fim)]
        for lider in sorted(grupo["lider"].astype(str)):
            if lider not in retornos_lideres.columns:
                continue
            serie = retornos_lideres.loc[datas, lider].dropna()
            serie = serie[serie.ne(0.0)]
            for data_sinal, retorno in serie.items():
                sessao = proxima.get(data_sinal)
                if sessao is None or sessao >= fim:
                    continue
                linhas.append(
                    {
                        "janela": str(janela),
                        "Data_Entrega": data_sinal,
                        "Sessao_Disponivel": sessao,
                        "lider": lider,
                        "direcao": 1 if float(retorno) > 0.0 else -1,
                    }
                )
    return pd.DataFrame(
        linhas,
        columns=[
            "janela",
            "Data_Entrega",
            "Sessao_Disponivel",
            "lider",
            "direcao",
        ],
    )


def _sinais_no_proprio_lider(sinais: pd.DataFrame) -> pd.DataFrame:
    """Cada lider e sua propria seguidora: IA sem rede."""
    return (
        sinais[["janela", "lider"]]
        .drop_duplicates()
        .assign(seguidora=lambda tabela: tabela["lider"])
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
            "dias_ativos": int(resultado.pnl_diario["n_operacoes_ativas"].gt(0).sum()),
            "posicao_dias": int(resultado.posicoes.ne(0.0).sum().sum()),
        }
    )
    return linha


def _distribuicao_placebo(
    *,
    sinais: pd.DataFrame,
    pares: pd.DataFrame,
    universo_top20: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    cotahist: pd.DataFrame,
    n_placebos: int,
    retorno_principal: float,
) -> tuple[pd.DataFrame, float]:
    if n_placebos <= 0:
        raise ValueError("n-placebos deve ser positivo")
    linhas = []
    for indice in range(n_placebos):
        seed = SEED_PLACEBO + indice
        pares_placebo = _pares_placebo(pares, universo_top20, seed)
        resultado = backtest_eventos.rodar_backtest_eventos(
            sinais,
            pares_placebo,
            calendario,
            cotahist=cotahist,
            h=HORIZONTES[0],
        )
        linhas.append(
            {
                "replicacao": indice + 1,
                "seed": seed,
                "retorno_total": float(resultado.pnl_diario["pnl_liquido"].sum()),
                "operacoes": int(len(resultado.operacoes)),
            }
        )
    tabela = pd.DataFrame(linhas)
    extremos = int(tabela["retorno_total"].ge(retorno_principal).sum())
    p_randomizacao = (extremos + 1) / (n_placebos + 1)
    return tabela, float(p_randomizacao)


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    execucao = criar_execucao("v2_backtest")
    log = configurar_log(execucao, "v2_07_backtest_eventos")
    cfg = CONFIG_PADRAO
    config = {
        "rede": str(args.rede),
        "universo": str(args.universo),
        "classificacoes": str(args.classificacoes),
        "horizontes": list(HORIZONTES),
        "seed_placebo": SEED_PLACEBO,
        "n_placebos": args.n_placebos,
        "carater": "exploratorio_gate_reprovado",
    }
    try:
        sinais = _sinais_da_ia(args.classificacoes)
        ativos = sinais[~sinais["abstencao"].astype(bool)].copy()
        log.info(
            "sinais: %d lider-dia, %d ativos, %d abstencoes",
            len(sinais),
            len(ativos),
            len(sinais) - len(ativos),
        )

        pares = _pares_estruturais(args.rede)
        universo_top20 = universo_v2.carregar_universo_top20(args.universo)
        log.info(
            "pares estruturais top20: %d em %d janelas",
            len(pares),
            pares["janela"].nunique(),
        )

        cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
        calendario = qualidade_dados.calendario_pregoes(cot)
        volume = retornos.painel_volume_financeiro(cot)
        retornos_lideres = retornos.retornos_preco_bruto_cotahist(
            cot,
            calendario,
            volume,
            mascarar_dia_seguinte=cfg.dados.mascarar_pregao_seguinte_ao_evento,
        )

        pares_rede = pares[["janela", "lider", "seguidora"]].drop_duplicates()
        sinais_sem_ia = _sinais_rede_sem_ia(pares, retornos_lideres, calendario)
        pares_placebo = _pares_placebo(pares, universo_top20, SEED_PLACEBO)
        bracos = {
            "ia_mais_rede": (ativos, pares_rede),
            "ia_sem_rede": (ativos, _sinais_no_proprio_lider(ativos)),
            "rede_sem_ia": (sinais_sem_ia, pares_rede),
            "seguidora_aleatoria": (ativos, pares_placebo),
        }

        resumos: list[dict[str, object]] = []
        resultados: dict[tuple[str, int], object] = {}
        destino = execucao.tabelas
        destino.mkdir(parents=True, exist_ok=True)
        for nome, (sinal, par) in bracos.items():
            for holding in HORIZONTES:
                if nome != "ia_mais_rede" and holding != HORIZONTES[0]:
                    continue
                resultado = backtest_eventos.rodar_backtest_eventos(
                    sinal, par, calendario, cotahist=cot, h=holding
                )
                resultados[(nome, holding)] = resultado
                resumos.append(_resumo(nome, resultado, holding))
                if holding == HORIZONTES[0]:
                    resultado.pnl_diario[["pnl_liquido"]].to_csv(
                        destino / f"pnl_diario_{nome}.csv", encoding="utf-8-sig"
                    )
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
                    resultado.pnl_operacao_dia.to_csv(
                        destino / "pnl_operacao_dia_principal.csv",
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
        principal = tabela[
            tabela["braco"].eq("ia_mais_rede") & tabela["h"].eq(HORIZONTES[0])
        ].iloc[0]
        distribuicao, p_randomizacao = _distribuicao_placebo(
            sinais=ativos,
            pares=pares,
            universo_top20=universo_top20,
            calendario=calendario,
            cotahist=cot,
            n_placebos=args.n_placebos,
            retorno_principal=float(principal["retorno_total"]),
        )
        tabela["p_randomizacao_seguidora"] = np.nan
        tabela.loc[
            tabela["braco"].eq("ia_mais_rede") & tabela["h"].eq(HORIZONTES[0]),
            "p_randomizacao_seguidora",
        ] = p_randomizacao

        sensibilidade = []
        for taxa in cfg.custos.aluguel_cenarios_anual:
            if taxa == cfg.custos.aluguel_cenario_base:
                resultado = resultados[("ia_mais_rede", HORIZONTES[0])]
            else:
                resultado = backtest_eventos.rodar_backtest_eventos(
                    ativos,
                    pares_rede,
                    calendario,
                    cotahist=cot,
                    h=HORIZONTES[0],
                    taxa_aluguel_anual=taxa,
                )
            sensibilidade.append(
                {
                    "taxa_aluguel_anual": taxa,
                    "retorno_total": float(resultado.pnl_diario["pnl_liquido"].sum()),
                    "operacoes": int(len(resultado.operacoes)),
                }
            )

        tabela.to_csv(destino / "resumo_bracos.csv", index=False, encoding="utf-8-sig")
        distribuicao.to_csv(
            destino / "placebo_seguidora_distribuicao.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(sensibilidade).to_csv(
            destino / "sensibilidade_aluguel.csv",
            index=False,
            encoding="utf-8-sig",
        )
        sinais.to_csv(destino / "sinais_agregados.csv", index=False, encoding="utf-8-sig")
        sinais_sem_ia.to_csv(
            destino / "sinais_rede_sem_ia.csv", index=False, encoding="utf-8-sig"
        )
        pares_placebo.to_csv(
            destino / "pares_seguidora_aleatoria.csv", index=False, encoding="utf-8-sig"
        )

        gravar_manifesto(
            execucao,
            config,
            arquivos_dados=[args.rede, args.universo, args.classificacoes],
            status="concluida",
            extras={
                "arestas_top20": int(len(pares_rede)),
                "sinais_ativos": int(len(ativos)),
                "sinais_rede_sem_ia": int(len(sinais_sem_ia)),
                "retorno_total_principal": float(principal["retorno_total"]),
                "sharpe_principal": float(principal["sharpe"]),
                "p_randomizacao_seguidora": p_randomizacao,
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
