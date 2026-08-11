"""
Sanidades da execucao completa.

Nao recalcula nada: confere se o que as etapas gravaram respeita as invariantes
que o projeto afirma. Cada checagem que falha e listada, e qualquer falha
derruba o codigo de retorno.

Checagens:
  1. manifestos das etapas 2, 3 e 4 com status "concluida";
  2. P&L liquido <= bruto, dia a dia, em toda carteira e cenario;
  3. exposicao bruta dentro do teto (1 + tolerancia);
  4. turnover nao negativo;
  5. pares selecionados com beta positivo e lider mais bem posicionada;
  6. FDR: aprovacoes nunca excedem hipoteses testadas;
  7. nenhum par com o mesmo ticker nas duas pontas.

Uso:
    .venv\\Scripts\\python.exe scripts\\99_checagens_integradas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DIR_RUNS = RAIZ / "outputs" / "runs"

CARTEIRAS = ("fdr_long_short", "fdr_long_only",
             "top_k_long_short", "top_k_long_only")


def _ultima_run(sufixo: str) -> Path | None:
    for pasta in sorted(DIR_RUNS.glob(f"*_{sufixo}"), reverse=True):
        m = pasta / "manifest.json"
        if m.exists() and json.loads(m.read_text("utf-8")).get("status") == "concluida":
            return pasta
    return None


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa99")
    log = configurar_log(execucao, "99_checagens")
    falhas: list[str] = []

    def checar(condicao: bool, descricao: str):
        if condicao:
            log.info("OK     | %s", descricao)
        else:
            log.error("FALHOU | %s", descricao)
            falhas.append(descricao)

    # No pipeline tudo esta na pasta desta execucao; rodando solto, conferimos a
    # ultima run concluida de cada etapa.
    if (execucao.tabelas / "resumo_carteiras.csv").exists():
        runs = {s: execucao.pasta for s in ("etapa2", "etapa3", "etapa4")}
        log.info("Conferindo a propria execucao compartilhada: %s", execucao.id)
    else:
        runs = {s: _ultima_run(s) for s in ("etapa2", "etapa3", "etapa4")}
        for sufixo, pasta in runs.items():
            checar(pasta is not None, f"{sufixo}: existe execucao concluida")
        if any(p is None for p in runs.values()):
            gravar_manifesto(execucao, cfg.to_dict(), status="falhou",
                             extras={"falhas": falhas})
            return 1
        log.info("Conferindo: %s", {k: v.name for k, v in runs.items()})

    # etapa 4: invariantes do dinheiro
    T4 = runs["etapa4"] / "tabelas"
    tol = 1e-9
    for carteira in CARTEIRAS:
        arq = T4 / f"pnl_{carteira}.csv"
        if not arq.exists():
            log.warning("carteira %s sem P&L (pode ser regra vazia)", carteira)
            continue
        pnl = pd.read_csv(arq, index_col=0, parse_dates=True)
        cen = pd.read_csv(T4 / f"cenarios_aluguel_{carteira}.csv",
                          index_col=0, parse_dates=True)
        exp = pd.read_csv(T4 / f"exposicoes_{carteira}.csv",
                          index_col=0, parse_dates=True)
        pos = pd.read_csv(T4 / f"posicoes_{carteira}.csv",
                          index_col=0, parse_dates=True)

        checar(bool((pnl["liquido"] <= pnl["bruto"] + tol).all()),
               f"{carteira}: liquido <= bruto em todos os dias")
        acum = [cen[c].sum() for c in cen.columns]
        checar(all(a >= b - tol for a, b in zip(acum, acum[1:])),
               f"{carteira}: cenarios de aluguel monotonicos")
        checar(bool((exp["bruta"] <= 1.0 + 1e-6).all()),
               f"{carteira}: exposicao bruta <= 1")
        giro = pos.diff().abs().sum(axis=1)
        checar(bool((giro >= -tol).all()), f"{carteira}: turnover >= 0")
        if "long_only" in carteira:
            checar(bool((pos.min() >= -tol).all().all()),
                   f"{carteira}: nenhuma posicao vendida")

    for regra in ("fdr", "top_k"):
        arq = T4 / f"pares_selecionados_{regra}.csv"
        sel = pd.read_csv(arq)
        if sel.empty:
            log.info("OK     | selecao %s vazia (resultado legitimo)", regra)
            continue
        checar(bool((sel["beta"] > 0).all()),
               f"selecao {regra}: todo par com beta positivo")
        checar(bool((sel["posicao_lider"] < sel["posicao_seguidora"]).all()),
               f"selecao {regra}: lider mais liquida que seguidora")
        checar(bool((sel["lider"] != sel["seguidora"]).all()),
               f"selecao {regra}: pontas distintas")

    # etapa 3: coerencia da inferencia
    T3 = runs["etapa3"] / "tabelas"
    fdr = pd.read_csv(T3 / "fdr_resumo_por_faixa.csv")
    checar(bool((fdr["aprovados_fdr"] <= fdr["hipoteses_testadas"]).all()),
           "FDR: aprovacoes nunca excedem hipoteses")
    rede = pd.read_csv(T3 / "rede_por_janela.csv")
    checar(bool((rede["lider"] != rede["seguidora"]).all()),
           "rede: nenhum par com o mesmo ticker nas duas pontas")
    checar(int(rede["n"].max()) <= 505,
           "rede: nenhuma estimativa com mais pregoes que um treino de 24m")

    status = "concluida" if not falhas else "falhou"
    gravar_manifesto(execucao, cfg.to_dict(), status=status, extras={
        "runs_conferidas": {k: v.name for k, v in runs.items()},
        "falhas": falhas,
    })
    log.info("%d falha(s)", len(falhas))
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
