"""
Quem precisa de setor, e por qual via.

A classificacao oficial da B3 so tem quem esta listado hoje. Os papeis que
sairam da bolsa, preservados pelo COTAHIST e identificados na etapa 1c como
ausentes no Yahoo, nao estao la. Montar o setor so com a fonte atual traria o
vies de sobrevivencia de volta.

O script separa duas filas e nao preenche nenhuma:

    candidato_via_oficial_b3   negociou ate o fim da amostra, entao
                               provavelmente esta na tabela oficial
    candidato_via_documental   encerrado ou transformado, provavelmente exige
                               pesquisa caso a caso

O nome diz "candidato" porque sao hipoteses: ter negociado nos ultimos pregoes
nao prova presenca na classificacao da B3, so a importacao daquela fonte
confirma. Ate la, a fila dimensiona trabalho, nao afirma procedencia.

Nao inventa setor, nao consulta fonte externa e nao herda de sucessor. Grava um
esqueleto para revisao que nao substitui `data/reference/setores_b3.csv`;
qualquer incorporacao a tabela de referencia exige conferencia manual.

Uso:
    .venv\\Scripts\\python.exe scripts\\01e_lista_curadoria_setorial.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import backtest, dados, qualidade_dados, setores, universo  # noqa: E402
from src.config import CONFIG_PADRAO  # noqa: E402
from src.execucao import Execucao, configurar_log, criar_execucao, gravar_manifesto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SETORES = RAIZ / "data" / "reference" / "setores_b3.csv"

# Papel que negociou nos ultimos N dias da amostra ainda esta na classificacao
# oficial. O resto precisa da via documental.
DIAS_VIVO_NO_FIM = 30


def main(execucao: Execucao | None = None) -> int:
    cfg = CONFIG_PADRAO
    execucao = execucao or criar_execucao("etapa1e")
    log = configurar_log(execucao, "01e_setores")
    T = execucao.tabelas

    log.info("Carregando COTAHIST...")
    cot, _ = dados.carregar_periodo(cfg.periodo.inicio, somente_acoes=True)
    calendario = qualidade_dados.calendario_pregoes(cot)
    janelas = backtest.janelas_do_periodo(cot, cfg.periodo, cfg.walk_forward)

    log.info("Quem alcanca alguma faixa em %d janelas...", len(janelas))
    melhor = universo.melhor_posicao_por_ticker(cot, janelas, cfg.liquidez)
    faixa_max = max(cfg.liquidez.faixas)
    necessarios = melhor[melhor["melhor_posicao"] <= faixa_max]
    log.info("Tickers que precisam de setor: %d", len(necessarios))

    g = cot.groupby("CODNEG")
    perfil = pd.DataFrame({
        "primeira_data": g["DATA"].min(),
        "ultima_data": g["DATA"].max(),
        "nome": g["NOMRES"].first(),
        "tipo_papel": g["TIPO_PAPEL"].first(),
        "codisi": g["CODISI"].first(),
    })
    perfil["emissor_id"] = perfil["codisi"].map(setores.emissor_do_isin)

    alertas = setores.auditar_emissores(cot)
    alertas.to_csv(T / "emissores_alertas_de_revisao.csv", encoding="utf-8")
    if len(alertas):
        log.warning("%d ticker(s) com alerta de revisao no emissor (nao e "
                    "defeito confirmado):\n%s", len(alertas),
                    alertas["alerta"].value_counts().to_string())
        log.warning("nomes_divergentes_no_emissor tanto pode ser colisao real "
                    "quanto grafia - BIDI alterna entre BANCO INTER e INTER "
                    "BANCO, que e a mesma empresa.")

    corte = calendario.max() - pd.Timedelta(days=DIAS_VIVO_NO_FIM)
    fila = necessarios.join(perfil)
    fila["negociou_ate_o_fim"] = fila["ultima_data"] >= corte
    fila["via_hipotese"] = fila["negociou_ate_o_fim"].map(
        {True: "candidato_via_oficial_b3", False: "candidato_via_documental"})

    # Uma classificacao serve o emissor inteiro, entao a unidade de trabalho e o
    # emissor e nao o ticker: PETR3 e PETR4 sao uma pesquisa so.
    fila["tickers_do_emissor"] = fila["emissor_id"].map(
        fila.groupby("emissor_id").size())

    fila = fila.sort_values(["via_hipotese", "melhor_posicao"])
    fila.to_csv(T / "curadoria_setorial_fila.csv", encoding="utf-8")

    log.info("--- fila por via (hipotese, nao procedencia confirmada) ---")
    for via, n in fila["via_hipotese"].value_counts().items():
        emissores = fila.loc[fila["via_hipotese"] == via, "emissor_id"].nunique()
        log.info("  %-26s %3d ticker(s) | %d emissor(es)", via, n, emissores)

    # Emissor nas duas filas quer dizer que uma classe sobreviveu e a outra nao,
    # normalmente por mudanca de ticker ou conversao. Pesquisar as duas em
    # separado daria classificacoes contraditorias para a mesma empresa.
    por_emissor = fila.groupby("emissor_id")["via_hipotese"].nunique()
    divididos = sorted(por_emissor[por_emissor > 1].index)
    if divididos:
        tab = fila[fila["emissor_id"].isin(divididos)].sort_values(
            ["emissor_id", "melhor_posicao"])
        tab.to_csv(T / "curadoria_emissores_nas_duas_filas.csv", encoding="utf-8")
        log.warning("--- %d emissor(es) nas duas filas: %s ---",
                    len(divididos), ", ".join(divididos))
        log.warning("\n%s", tab[["emissor_id", "via_hipotese", "melhor_posicao",
                                 "primeira_data", "ultima_data", "nome"]]
                    .to_string())

    doc = fila[fila["via_hipotese"] == "candidato_via_documental"]
    log.info("--- candidatos via documental, por melhor posicao ---\n%s",
             doc[["melhor_posicao", "janelas_elegivel", "primeira_data",
                  "ultima_data", "nome", "emissor_id"]]
             .head(45).to_string())

    # Estado atual da tabela curada. O periodo em que cada ticker precisa de
    # setor e a propria janela de vida: exigir cobertura fora dela criaria
    # lacuna onde o papel nem existia.
    periodos = pd.DataFrame({
        "ticker_observado": fila.index,
        "inicio": fila["primeira_data"].values,
        "fim": fila["ultima_data"].values,
    })

    if SETORES.exists():
        tab = setores.carregar(SETORES)
        erros = setores.validar(tab)
        if erros:
            log.error("setores_b3.csv invalido:")
            for e in erros:
                log.error("  - %s", e)
            return 1
        log.info("setores_b3.csv: %d linha(s)", len(tab))
    else:
        tab = setores.tabela_vazia()
        log.warning("data/reference/setores_b3.csv nao existe")

    lac = setores.lacunas(tab, periodos)
    lac = lac.merge(
        fila[["via_hipotese", "nome", "emissor_id", "melhor_posicao"]],
        left_on="ticker_observado", right_index=True, how="left")
    lac.to_csv(T / "curadoria_setorial_lacunas.csv", index=False, encoding="utf-8")
    log.info("--- lacunas (%d) ---\n%s", len(lac),
             lac["situacao"].value_counts().to_string())

    # Esqueleto para preenchimento humano, gravado na pasta da execucao e nao em
    # data/reference. Incorporar uma linha a metodologia exige revisao manual.
    # O emissor vai junto para nao duplicar pesquisa entre classes.
    esqueleto = pd.DataFrame(
        {c: "" for c in setores.COLUNAS}, index=range(len(fila)))
    esqueleto["ticker_observado"] = list(fila.index)
    esqueleto["emissor_id"] = fila["emissor_id"].fillna("").values
    esqueleto["status_revisao"] = "pendente"
    esqueleto = esqueleto.sort_values(["emissor_id", "ticker_observado"])
    esqueleto.to_csv(T / "setores_b3_esqueleto.csv", index=False, encoding="utf-8")
    log.info("Esqueleto com %d linha(s) 'pendente', agrupado por emissor, em %s",
             len(esqueleto), T / "setores_b3_esqueleto.csv")
    log.info("Nenhum setor foi preenchido; revise antes de incorporar a "
             "data/reference.")

    gravar_manifesto(execucao, cfg.to_dict(), status="concluida", extras={
        "tickers_que_precisam_de_setor": int(len(necessarios)),
        "candidato_via_oficial_b3": int(
            (fila["via_hipotese"] == "candidato_via_oficial_b3").sum()),
        "candidato_via_documental": int(
            (fila["via_hipotese"] == "candidato_via_documental").sum()),
        "via_e_hipotese_nao_procedencia_confirmada": True,
        "emissores_distintos": int(fila["emissor_id"].nunique()),
        "emissores_nas_duas_filas": divididos,
        "tickers_com_alerta_no_emissor": int(len(alertas)),
        "lacunas": int(len(lac)),
        "setores_preenchidos_automaticamente": False,
        "rede_estimada": False,
    })
    log.info("Tabelas em %s", T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
