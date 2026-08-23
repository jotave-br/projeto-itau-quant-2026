from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.config import CustosConfig, EstrategiaConfig
from src.custos import custo_por_ponta_bps
from src.v2 import backtest_eventos


def _calendario(n: int = 8) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-05", periods=n)


def _sinais(
    calendario: pd.DatetimeIndex,
    *,
    posicao: int = 2,
    direcao: int = 1,
    lider: str = "LID3",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "janela": ["2026-01"],
            "Sessao_Disponivel": [calendario[posicao]],
            "lider": [lider],
            "direcao": [direcao],
        }
    )


def _pares(lider: str = "LID3", seguidoras=("SEG3",)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "janela": "2026-01",
            "lider": lider,
            "seguidora": list(seguidoras),
        }
    )


def _paineis(
    calendario: pd.DatetimeIndex,
    tickers=("SEG3",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    abertura = pd.DataFrame(100.0, index=calendario, columns=list(tickers))
    fechamento = pd.DataFrame(100.0, index=calendario, columns=list(tickers))
    return abertura, fechamento


def _rodar(
    sinais: pd.DataFrame,
    pares: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    abertura: pd.DataFrame,
    fechamento: pd.DataFrame,
    **kwargs,
) -> backtest_eventos.ResultadoBacktestEventos:
    return backtest_eventos.rodar_backtest_eventos(
        sinais,
        pares,
        calendario,
        painel_preabe=abertura,
        painel_preult=fechamento,
        **kwargs,
    )


def test_timing_usa_open_close_na_entrada_e_ignora_precos_fora_do_holding():
    cal = _calendario()
    sinais = _sinais(cal)
    abertura, fechamento = _paineis(cal)
    entrada = cal[2]
    fechamento.loc[entrada, "SEG3"] = 110.0
    fechamento.loc[cal[3], "SEG3"] = 121.0
    fechamento.loc[cal[4], "SEG3"] = 133.1

    base = _rodar(sinais, _pares(), cal, abertura, fechamento, h=3)
    adulterada_abertura = abertura.copy()
    adulterado_fechamento = fechamento.copy()
    adulterada_abertura.loc[cal[0], "SEG3"] = 1e9
    adulterado_fechamento.loc[cal[0], "SEG3"] = 1e9
    adulterada_abertura.loc[cal[5]:, "SEG3"] = 1e-6
    adulterado_fechamento.loc[cal[5]:, "SEG3"] = 1e-6
    depois = _rodar(
        sinais, _pares(), cal, adulterada_abertura, adulterado_fechamento, h=3
    )

    pd.testing.assert_frame_equal(base.operacoes, depois.operacoes)
    pd.testing.assert_frame_equal(base.pnl_diario, depois.pnl_diario)
    fluxo = base.pnl_operacao_dia
    assert fluxo["tipo_retorno"].tolist() == [
        "abertura_fechamento",
        "fechamento_fechamento",
        "fechamento_fechamento",
    ]
    assert fluxo["retorno_ativo"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert base.operacoes.iloc[0]["data_entrada"] == cal[2]
    assert base.operacoes.iloc[0]["data_saida"] == cal[4]


def test_pnl_intraholding_usa_quantidade_fixa_sem_rebalanceamento_diario():
    cal = _calendario()
    abertura, fechamento = _paineis(cal)
    entrada = cal[2]
    fechamento.loc[entrada, "SEG3"] = 110.0
    fechamento.loc[cal[3], "SEG3"] = 121.0
    fechamento.loc[cal[4], "SEG3"] = 133.1
    cfg = replace(EstrategiaConfig(), peso_maximo_por_posicao=1.0)

    resultado = _rodar(
        _sinais(cal),
        _pares(),
        cal,
        abertura,
        fechamento,
        h=3,
        cfg_estrategia=cfg,
        taxa_aluguel_anual=0.0,
    )

    fluxo = resultado.pnl_operacao_dia
    assert fluxo["retorno_ativo"].tolist() == pytest.approx([0.10, 0.10, 0.10])
    assert fluxo["retorno_incremental_entrada"].tolist() == pytest.approx(
        [0.10, 0.11, 0.121]
    )
    peso = 1 / 3
    assert resultado.operacoes.iloc[0]["pnl_bruto"] == pytest.approx(
        peso * (133.1 / 100.0 - 1.0)
    )


@pytest.mark.parametrize("h", [1, 3, 5])
def test_horizontes_saem_no_fechamento_da_h_esima_sessao(h):
    cal = _calendario(10)
    sinais = _sinais(cal, posicao=1)
    abertura, fechamento = _paineis(cal)

    resultado = _rodar(sinais, _pares(), cal, abertura, fechamento, h=h)

    operacao = resultado.operacoes.iloc[0]
    assert operacao["data_entrada"] == cal[1]
    assert operacao["data_saida"] == cal[h]
    assert operacao["peso"] == pytest.approx(0.1 / h)
    assert len(resultado.pnl_operacao_dia) == h


def test_horizonte_fora_do_pre_registro_e_recusado():
    cal = _calendario()
    abertura, fechamento = _paineis(cal)
    with pytest.raises(ValueError, match="h deve ser"):
        _rodar(_sinais(cal), _pares(), cal, abertura, fechamento, h=2)


@pytest.mark.parametrize("direcao,multiplicador", [(1, 1.0), (-1, -1.0)])
def test_long_e_short_aplicam_a_direcao_ao_mesmo_retorno(
    direcao, multiplicador
):
    cal = _calendario()
    abertura, fechamento = _paineis(cal)
    fechamento.loc[cal[2], "SEG3"] = 110.0
    cfg = replace(EstrategiaConfig(), peso_maximo_por_posicao=1.0)

    resultado = _rodar(
        _sinais(cal, direcao=direcao),
        _pares(),
        cal,
        abertura,
        fechamento,
        h=1,
        cfg_estrategia=cfg,
        taxa_aluguel_anual=0.0,
    )

    assert resultado.operacoes.iloc[0]["peso"] == multiplicador
    assert resultado.operacoes.iloc[0]["pnl_bruto"] == pytest.approx(
        multiplicador * 0.10
    )


def test_fanout_divide_unidade_e_aplica_teto_sem_renormalizar():
    cal = _calendario()
    abertura, fechamento = _paineis(cal, ("A3", "B3", "C3"))
    pares = _pares(seguidoras=("C3", "A3", "B3"))

    sem_corte = _rodar(
        _sinais(cal),
        pares,
        cal,
        abertura,
        fechamento,
        h=1,
        cfg_estrategia=replace(
            EstrategiaConfig(), peso_maximo_por_posicao=1.0
        ),
    )
    com_corte = _rodar(_sinais(cal), pares, cal, abertura, fechamento, h=1)

    assert sem_corte.operacoes["seguidora"].tolist() == ["A3", "B3", "C3"]
    assert sem_corte.operacoes[
        "contribuicao_liquida_pre_normalizacao"
    ].tolist() == pytest.approx(
        [1 / 3] * 3
    )
    assert sem_corte.operacoes["peso"].abs().sum() == pytest.approx(1.0)
    assert com_corte.operacoes["peso"].tolist() == pytest.approx([0.1] * 3)
    assert com_corte.operacoes["peso"].abs().sum() == pytest.approx(0.3)


def test_coorte_neta_normaliza_aplica_teto_e_so_entao_divide_por_h():
    cal = _calendario()
    abertura, fechamento = _paineis(cal, ("A3", "B3"))
    sinais = pd.concat(
        [
            _sinais(cal, lider="L1"),
            _sinais(cal, lider="L2"),
        ],
        ignore_index=True,
    )
    pares = pd.concat(
        [
            _pares("L1", ("A3", "B3")),
            _pares("L2", ("A3",)),
        ],
        ignore_index=True,
    )
    cfg = replace(EstrategiaConfig(), peso_maximo_por_posicao=0.60)

    resultado = _rodar(
        sinais,
        pares,
        cal,
        abertura,
        fechamento,
        h=3,
        cfg_estrategia=cfg,
    )
    operacoes = resultado.operacoes.set_index("seguidora")

    # A/B: 1,5/0,5 -> 0,75/0,25; o teto limita A antes do 1/H.
    assert operacoes.loc["A3", "contribuicao_liquida_pre_normalizacao"] == \
        pytest.approx(1.5)
    assert operacoes.loc["B3", "contribuicao_liquida_pre_normalizacao"] == \
        pytest.approx(0.5)
    assert operacoes["exposicao_bruta_coorte_pre_normalizacao"].tolist() == \
        pytest.approx([2.0, 2.0])
    assert operacoes.loc["A3", "peso_normalizado"] == pytest.approx(0.75)
    assert operacoes.loc["B3", "peso_normalizado"] == pytest.approx(0.25)
    assert operacoes.loc["A3", "peso_pos_teto"] == pytest.approx(0.60)
    assert operacoes.loc["B3", "peso_pos_teto"] == pytest.approx(0.25)
    assert operacoes.loc["A3", "peso"] == pytest.approx(0.60 / 3)
    assert operacoes.loc["B3", "peso"] == pytest.approx(0.25 / 3)
    assert operacoes.loc["A3", "lideres_origem"] == ["L1", "L2"]
    assert operacoes.loc["A3", "n_sinais_origem"] == 2
    assert operacoes.loc["A3", "contribuicoes_origem"] == [0.5, 1.0]


def test_net_zero_na_coorte_nao_opera_e_preserva_origens_no_diagnostico():
    cal = _calendario()
    abertura, fechamento = _paineis(cal)
    sinais = pd.concat(
        [
            _sinais(cal, direcao=-1, lider="AID3"),
            _sinais(cal, direcao=1, lider="ZID3"),
        ],
        ignore_index=True,
    )
    pares = pd.concat([_pares("AID3"), _pares("ZID3")], ignore_index=True)

    resultado = _rodar(sinais, pares, cal, abertura, fechamento, h=3)

    assert resultado.operacoes.empty
    assert resultado.diagnosticos["motivo_diagnostico"].tolist() == [
        backtest_eventos.CONTRIBUICAO_LIQUIDA_ZERO
    ]
    diagnostico = resultado.diagnosticos.iloc[0]
    assert diagnostico["lideres_origem"] == ["AID3", "ZID3"]
    assert diagnostico["ids_sinal_origem"] == [
        "sinal_000001",
        "sinal_000002",
    ]
    assert diagnostico["n_sinais_origem"] == 2
    assert resultado.resumo["n_net_zero"] == 1


def test_custos_cobram_duas_pontas_e_aluguel_so_no_short():
    cal = _calendario()
    abertura, fechamento = _paineis(cal)
    cfg_e = replace(EstrategiaConfig(), peso_maximo_por_posicao=1.0)
    cfg_c = CustosConfig(aluguel_cenario_base=0.05)

    long = _rodar(
        _sinais(cal, direcao=1),
        _pares(),
        cal,
        abertura,
        fechamento,
        h=3,
        cfg_estrategia=cfg_e,
        cfg_custos=cfg_c,
    )
    short = _rodar(
        _sinais(cal, direcao=-1),
        _pares(),
        cal,
        abertura,
        fechamento,
        h=3,
        cfg_estrategia=cfg_e,
        cfg_custos=cfg_c,
    )

    custo_ponta = custo_por_ponta_bps(cfg_c) / 1e4
    peso = 1 / 3
    assert long.operacoes.iloc[0]["custo_entrada"] == pytest.approx(
        peso * custo_ponta
    )
    assert long.operacoes.iloc[0]["custo_saida"] == pytest.approx(
        peso * custo_ponta
    )
    assert long.operacoes.iloc[0]["custo_aluguel"] == 0.0
    assert short.operacoes.iloc[0]["custo_aluguel"] == pytest.approx(
        peso * 0.05 * 3 / 252
    )
    assert short.operacoes.iloc[0]["pnl_liquido"] == pytest.approx(
        -2 * peso * custo_ponta - peso * 0.05 * 3 / 252
    )
    assert short.pnl_diario["custo_giro"].sum() == pytest.approx(
        2 * peso * custo_ponta
    )


def test_cotahist_longo_produz_o_mesmo_resultado_dos_paineis():
    cal = _calendario()
    abertura, fechamento = _paineis(cal)
    fechamento.loc[cal[2], "SEG3"] = 105.0
    por_painel = _rodar(
        _sinais(cal), _pares(), cal, abertura, fechamento, h=1
    )
    cotahist = pd.DataFrame(
        {
            "DATA": cal,
            "CODNEG": "SEG3",
            "PREABE": abertura["SEG3"].to_numpy(),
            "PREULT": fechamento["SEG3"].to_numpy(),
        }
    )

    longo = backtest_eventos.rodar_backtest_eventos(
        _sinais(cal), _pares(), cal, cotahist=cotahist, h=1
    )

    pd.testing.assert_frame_equal(por_painel.operacoes, longo.operacoes)
    pd.testing.assert_frame_equal(por_painel.pnl_diario, longo.pnl_diario)


def test_faltantes_suspensao_e_fim_do_calendario_sao_diagnosticados():
    cal = _calendario(5)
    abertura, fechamento = _paineis(cal, ("SEMOPEN3", "SEMCLOSE3"))
    abertura.loc[cal[1], "SEMOPEN3"] = 0.0
    fechamento.loc[cal[2], "SEMCLOSE3"] = pd.NA
    sinais = _sinais(cal, posicao=1)
    pares = _pares(seguidoras=("SEMOPEN3", "SEMCLOSE3"))

    faltantes = _rodar(sinais, pares, cal, abertura, fechamento, h=3)
    fim = _rodar(
        _sinais(cal, posicao=4), pares, cal, abertura, fechamento, h=3
    )

    assert faltantes.operacoes.empty
    assert set(faltantes.diagnosticos["motivo_diagnostico"]) == {
        backtest_eventos.PRECO_ENTRADA_AUSENTE,
        backtest_eventos.PRECO_FECHAMENTO_AUSENTE,
    }
    assert fim.diagnosticos["motivo_diagnostico"].tolist() == [
        backtest_eventos.FIM_CALENDARIO
    ]


def test_safras_sobrepostas_somam_ate_o_teto_e_sao_deterministicas():
    cal = _calendario()
    abertura, fechamento = _paineis(cal)
    sinais = pd.concat(
        [
            _sinais(cal, posicao=3, direcao=1, lider="ZID3"),
            _sinais(cal, posicao=2, direcao=1, lider="ZID3"),
            _sinais(cal, posicao=4, direcao=1, lider="ZID3"),
        ],
        ignore_index=True,
    )
    pares = _pares("ZID3")

    primeiro = _rodar(sinais, pares, cal, abertura, fechamento, h=3)
    segundo = _rodar(
        sinais.iloc[::-1].reset_index(drop=True),
        pares.iloc[::-1].reset_index(drop=True),
        cal,
        abertura,
        fechamento,
        h=3,
    )

    pd.testing.assert_frame_equal(primeiro.operacoes, segundo.operacoes)
    pd.testing.assert_frame_equal(primeiro.pnl_diario, segundo.pnl_diario)
    # O teto/H por coorte mantem o agregado no teto da V1.
    assert primeiro.posicoes.loc[cal[2], "SEG3"] == pytest.approx(0.1 / 3)
    assert primeiro.posicoes.loc[cal[3], "SEG3"] == pytest.approx(0.2 / 3)
    assert primeiro.posicoes.loc[cal[4], "SEG3"] == pytest.approx(0.1)
    assert primeiro.pnl_diario.loc[cal[2], "n_operacoes_ativas"] == 1
    assert primeiro.pnl_diario.loc[cal[3], "n_operacoes_ativas"] == 2
    assert primeiro.pnl_diario.loc[cal[4], "n_operacoes_ativas"] == 3
    assert primeiro.resumo["n_operacoes"] == 3


def test_data_entrega_precisa_anteceder_sessao_disponivel():
    cal = _calendario()
    sinais = _sinais(cal)
    sinais["Data_Entrega"] = sinais["Sessao_Disponivel"]
    abertura, fechamento = _paineis(cal)

    with pytest.raises(ValueError, match="estritamente anterior"):
        _rodar(sinais, _pares(), cal, abertura, fechamento, h=1)
