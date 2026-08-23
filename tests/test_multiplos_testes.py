"""
Testes da correcao de multiplas comparacoes.

O BH e verificado contra o exemplo do artigo original de Benjamini e Hochberg
(1995), que tem resposta conhecida: nao ha como a implementacao estar errada e
acertar aquele gabarito por coincidencia.
"""

import numpy as np
import pandas as pd
import pytest

from src import multiplos_testes as mt
from src.config import MultiplosTestesConfig

# Os 15 p-valores do exemplo do artigo original. Com q = 0,05, o BH rejeita
# exatamente as 4 primeiras hipoteses.
P_BH_1995 = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
             0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]


def _rede(pvalores, janela="2017-01", betas=None, ts=None):
    n = len(pvalores)
    return pd.DataFrame({
        "janela": janela,
        "lider": [f"L{i:02d}" for i in range(n)],
        "seguidora": [f"S{i:02d}" for i in range(n)],
        "p_valor": pvalores,
        "beta": betas if betas is not None else [0.1] * n,
        "estat_t": ts if ts is not None else list(range(n, 0, -1)),
    })


def test_bh_reproduz_o_exemplo_do_artigo_original():
    r = mt.benjamini_hochberg(pd.Series(P_BH_1995), q=0.05)
    assert r["aprovado"].sum() == 4
    assert list(r.index[r["aprovado"]]) == [0, 1, 2, 3]


def test_bh_p_ajustado_e_monotonico():
    r = mt.benjamini_hochberg(pd.Series(P_BH_1995), q=0.05)
    ajustados = r["p_ajustado"].to_numpy()
    assert (np.diff(ajustados) >= -1e-12).all()
    assert (ajustados <= 1.0).all()


def test_bh_nao_conta_par_nao_estimado_como_hipotese():
    """
    NaN e par que nao foi testado. Se ele entrasse em m, a existencia de pares
    nao estimados puniria os estimados - hipotese fantasma apertando a barra.
    """
    com_nan = pd.Series(P_BH_1995 + [np.nan] * 10)
    r_com = mt.benjamini_hochberg(com_nan, q=0.05)
    r_sem = mt.benjamini_hochberg(pd.Series(P_BH_1995), q=0.05)

    assert r_com["aprovado"].sum() == r_sem["aprovado"].sum() == 4
    assert not r_com.loc[15:, "aprovado"].any()
    assert r_com.loc[15:, "p_ajustado"].isna().all()


def test_mais_hipoteses_na_mesma_janela_apertam_a_barra():
    """
    A prova de que m importa - e portanto de que aplicar BH na janela errada
    muda o resultado. Dois p-valores que passam sozinhos deixam de passar
    quando 18 testes ruins entram na mesma correcao.
    """
    cfg = MultiplosTestesConfig()
    sozinhos, _ = mt.aplicar_fdr_janela(_rede([0.01, 0.02]), cfg)
    diluidos, _ = mt.aplicar_fdr_janela(_rede([0.01, 0.02] + [0.9] * 18), cfg)

    assert sozinhos["aprovado_fdr"].sum() == 2
    assert diluidos["aprovado_fdr"].sum() == 0


def test_fdr_recusa_tabela_com_mais_de_uma_janela():
    """A protecao point-in-time: BH atravessando janelas e erro, nao ajuste."""
    duas = pd.concat([_rede([0.01], janela="2017-01"),
                      _rede([0.02], janela="2017-04")], ignore_index=True)
    with pytest.raises(ValueError, match="janelas"):
        mt.aplicar_fdr_janela(duas)


def test_resumo_registra_hipoteses_q_e_aprovacoes():
    rede, resumo = mt.aplicar_fdr_janela(
        _rede(P_BH_1995 + [np.nan]), MultiplosTestesConfig())
    assert resumo["hipoteses_testadas"] == 15
    assert resumo["pares_nao_estimados"] == 1
    assert resumo["q_fdr"] == MultiplosTestesConfig().q_fdr
    assert resumo["aprovados_fdr"] == int(rede["aprovado_fdr"].sum())


def test_selecao_top_k_limita_ordena_e_exige_beta_positivo():
    cfg = MultiplosTestesConfig()
    n = 30
    rede = _rede(list(np.linspace(0.001, 0.5, n)),
                 betas=[0.2] * (n - 5) + [-0.3] * 5,
                 ts=list(np.linspace(6, 0.5, n)))
    sel = mt.selecionar_pares(rede, "top_k", cfg)

    assert len(sel) == cfg.top_k_pares
    assert (sel["beta"] > 0).all()
    assert (sel["estat_t"].diff().dropna() <= 0).all()


def test_selecao_fdr_exige_correcao_previa_e_beta_positivo():
    cfg = MultiplosTestesConfig()
    rede = _rede([0.001, 0.002, 0.9], betas=[0.2, -0.2, 0.2])

    with pytest.raises(ValueError, match="aplicar_fdr_janela"):
        mt.selecionar_pares(rede, "fdr", cfg)

    corrigida, _ = mt.aplicar_fdr_janela(rede, cfg)
    sel = mt.selecionar_pares(corrigida, "fdr", cfg)
    assert list(sel["lider"]) == ["L00"]


def test_regra_desconhecida_e_erro():
    with pytest.raises(ValueError, match="regra"):
        mt.selecionar_pares(_rede([0.01]), "melhor_sharpe")
