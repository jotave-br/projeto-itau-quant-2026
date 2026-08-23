"""
Pipeline completo, na ordem do metodo.

Cria uma pasta de execucao e compartilha com todas as etapas: cada script grava
tabelas, figuras e logs na mesma run, e o manifesto final descreve o pipeline
inteiro. Codigo de retorno diferente de zero em qualquer etapa interrompe tudo e
marca a execucao como falha, para nao sobrar numero gerado depois de erro.

As auditorias 1b-1e dependem do cache do yfinance; `--sem-auditorias-yf` as
pula quando o objetivo e so o caminho principal COTAHIST.

Uso:
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --rotulo oficial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))  # raiz, para `import src.<modulo>`
sys.path.insert(0, str(_AQUI))  # scripts/, para importar as etapas

from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

ETAPAS = [
    ("Download, cache e auditoria dos dados", "01_baixar_e_auditar_dados", False),
    ("Precos, retornos e validacao de fontes", "01b_precos_e_retornos", True),
    ("Diagnostico de ausencia no Yahoo", "01c_diagnostico_ausencia_yf", True),
    ("Auditoria das mascaras de evento", "01d_auditoria_mascaras", True),
    ("Validacao da classificacao setorial", "01e_lista_curadoria_setorial", False),
    ("Universos e pares point-in-time", "02_construir_universos", False),
    ("Redes lead-lag, nao-sincronia, placebos e FDR", "03_estimar_redes", False),
    ("Backtest walk-forward com custos", "04_rodar_backtests", False),
    ("Tabelas e figuras do pre-relatorio", "05_gerar_relatorio", False),
    ("Checagens integradas", "99_checagens_integradas", False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline V1 lead-lag B3")
    ap.add_argument("--rotulo", default="v1", help="sufixo do id da execucao")
    ap.add_argument("--sem-auditorias-yf", action="store_true",
                    help="pula as auditorias que dependem do cache yfinance")
    args = ap.parse_args()

    cfg = CONFIG_PADRAO
    execucao = criar_execucao(args.rotulo)
    log = configurar_log(execucao)

    log.info("Execucao: %s", execucao.id)
    log.info("Pasta:    %s", execucao.pasta)
    log.info(
        "Config:   %s | faixa principal top-%d | treino %dm / teste %dm | holding %dd",
        cfg.periodo.inicio,
        cfg.liquidez.top_n,
        cfg.walk_forward.treino_meses,
        cfg.walk_forward.teste_meses,
        cfg.estrategia.holding_dias,
    )

    gravar_manifesto(execucao, cfg.to_dict(), status="iniciada")

    executadas, puladas = [], []
    for rotulo, modulo, e_auditoria in ETAPAS:
        if e_auditoria and args.sem_auditorias_yf:
            log.info("PULADA    | %s (--sem-auditorias-yf)", rotulo)
            puladas.append(rotulo)
            continue
        try:
            mod = __import__(modulo)
            rc = mod.main(execucao)
        except Exception:
            log.exception("FALHOU    | %s", rotulo)
            gravar_manifesto(execucao, cfg.to_dict(), status="falhou",
                             extras={"etapa_que_falhou": rotulo,
                                     "etapas_executadas": executadas})
            return 1
        if rc != 0:
            log.error("FALHOU    | %s (codigo %s)", rotulo, rc)
            gravar_manifesto(execucao, cfg.to_dict(), status="falhou",
                             extras={"etapa_que_falhou": rotulo,
                                     "etapas_executadas": executadas})
            return int(rc)
        log.info("OK        | %s", rotulo)
        executadas.append(rotulo)

    gravar_manifesto(execucao, cfg.to_dict(), status="concluida", extras={
        "etapas_executadas": executadas,
        "etapas_puladas": puladas,
    })
    log.info("Status final: concluida (%d etapa(s), %d pulada(s))",
             len(executadas), len(puladas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
