"""Gera as figuras da V2 para as páginas 3 e 4 do relatório."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402
from src.v2 import graficos_v2  # noqa: E402

PAINEL = RAIZ / "outputs/validacao_humana"
CHAVE = RAIZ / "data/processed/cvm_ipe/chave_validacao_humana.csv"
AVALIADORES = {
    "painel_qwen_max.csv": "Qwen 3.8 Max",
    "painel_gemini.csv": "Gemini 3.1 Pro",
    "painel_humano.csv": "leitura humana",
    "painel_kimi_26.csv": "Kimi 2.6",
    "painel_opus.csv": "Opus 5",
}
CLASSIFICADOR = "Qwen3 14B local"
ROTULOS_BRACOS = {
    "ia_mais_rede": "IA + rede",
    "ia_sem_rede": "IA sem rede",
    "rede_sem_ia": "rede sem IA (placebo)",
    "seguidora_aleatoria": "seguidora aleatória",
}


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--painel-run", type=Path, required=True)
    parser.add_argument("--backtest-run", type=Path, required=True)
    return parser.parse_args(argv)


def _contagens_direcionais() -> pd.Series:
    contagens: dict[str, int] = {}
    for arquivo, rotulo in AVALIADORES.items():
        tabela = pd.read_csv(PAINEL / arquivo, encoding="utf-8-sig", dtype=object)
        direcao = tabela["direcao"].str.strip().str.casefold()
        contagens[rotulo] = int(direcao.isin(["positiva", "negativa"]).sum())
    chave = pd.read_csv(CHAVE, encoding="utf-8-sig")
    contagens[CLASSIFICADOR] = int(
        chave["direcao_ia"].str.casefold().isin(["positiva", "negativa"]).sum()
    )
    return pd.Series(contagens)


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    execucao = criar_execucao("v2_figuras")
    log = configurar_log(execucao, "v2_08_figuras")
    config = {
        "painel_run": str(args.painel_run),
        "backtest_run": str(args.backtest_run),
    }
    try:
        destino = execucao.figuras
        destino.mkdir(parents=True, exist_ok=True)

        matriz = pd.read_csv(
            args.painel_run / "matriz_confusao.csv", index_col=0, encoding="utf-8-sig"
        )
        graficos_v2.fig_matriz_confusao(matriz, destino / "v2_01_matriz_confusao.png")

        graficos_v2.fig_escada_conservadorismo(
            _contagens_direcionais(),
            destaque=CLASSIFICADOR,
            caminho=destino / "v2_02_escada_conservadorismo.png",
        )

        series = {}
        for braco in ROTULOS_BRACOS:
            caminho = args.backtest_run / f"pnl_diario_{braco}.csv"
            tabela = pd.read_csv(caminho, index_col=0, parse_dates=[0],
                                 encoding="utf-8-sig")
            series[braco] = tabela["pnl_liquido"]
        graficos_v2.fig_curvas_bracos(
            series,
            principal="ia_mais_rede",
            placebo="rede_sem_ia",
            caminho=destino / "v2_03_curvas_bracos.png",
        )

        resumo = pd.read_csv(
            args.backtest_run / "resumo_bracos.csv", encoding="utf-8-sig"
        )
        graficos_v2.fig_intervalos(
            resumo, destino / "v2_04_intervalos.png", ROTULOS_BRACOS
        )

        arquivos = sorted(p.name for p in destino.glob("*.png"))
        gravar_manifesto(
            execucao,
            config,
            status="concluida",
            extras={"figuras": arquivos, "diretorio_figuras": str(destino)},
        )
        log.info("figuras em %s: %s", destino, ", ".join(arquivos))
        return 0
    except Exception as exc:
        log.exception("falha ao gerar figuras")
        gravar_manifesto(execucao, config, status="falhou", extras={"erro": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
