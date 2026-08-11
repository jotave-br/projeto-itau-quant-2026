"""
Testes de nao-sincronia, o controle central do projeto.

Atraso entre duas acoes pode vir de difusao de informacao, que e a hipotese, ou
de negociacao nao-sincrona, que e o artefato: a seguidora quase nao negociou,
seu ultimo preco registrado e velho e, quando alguem finalmente negocia, o
preco salta e parece reacao atrasada. Atraso de registro nao rende dinheiro -
na hora de executar voce paga o preco ja corrigido - e se disfarca
perfeitamente de difusao numa regressao defasada. Dai a bateria:

1. estratificacao por liquidez, nas faixas top 20/40/60/100 (a coluna
   `faixa_minima` dos pares ja carrega isso);
2. filtro de dias de preco velho, com reestimacao (`filtrar_dias_preco_velho`,
   `reestimar_sem_preco_velho`);
3. subconjunto so com quem negociou todo dia (`subconjunto_sempre_negociado`);
4. placebo de embaralhamento (`rodar_placebos`);
5. placebo de direcao invertida, em leadlag.estimar_direcao_invertida;
6. lags alternativos, lag 0 incluido, em leadlag.estimar_lags.

Como ler: efeito que so aparece nos mais iliquidos e artefato potencial ou
previsibilidade nao implementavel, nao achado. E "suspeito" nunca vira exclusao
silenciosa - toda reestimativa reporta o que perdeu pelo caminho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import leadlag
from src.backtest import Janela
from src.config import LeadLagConfig, NaoSincroniaConfig, PlaceboConfig


def _recorte(painel: pd.DataFrame, janela: Janela) -> pd.DataFrame:
    return painel.loc[(painel.index >= janela.treino_inicio)
                      & (painel.index < janela.treino_fim)]


def filtrar_dias_preco_velho(
    volume: pd.DataFrame,
    negocios: pd.DataFrame,
    precos: pd.DataFrame,
    janela: Janela,
    cfg: NaoSincroniaConfig | None = None,
) -> pd.DataFrame:
    """
    Mascara (datas x tickers) dos dias de preco provavelmente velho no treino.

    Marca o dia se qualquer criterio pegar:
      - menos de `min_negocios_dia` negocios;
      - volume financeiro no percentil inferior do proprio ticker, medido so
        dentro da janela, porque comparar com a amostra inteira usaria dias que
        ainda nao aconteceram;
      - fechamento identico aos anteriores em sequencia de pelo menos
        `max_precos_repetidos` precos iguais.

    Marcar nao e excluir: quem decide o destino do dia marcado e a
    reestimativa, e a perda de amostra fica registrada la.
    """
    cfg = cfg or NaoSincroniaConfig()
    vol = _recorte(volume, janela)
    neg = _recorte(negocios, janela).reindex(columns=vol.columns)
    pre = _recorte(precos, janela).reindex(columns=vol.columns)

    poucos_negocios = neg < cfg.min_negocios_dia
    sem_volume = ~(vol > 0)

    # Estritamente menor que o percentil: com <=, um ticker de volume constante
    # teria todos os dias marcados, ja que o limiar coincide com o volume dele.
    # Dia igual ao limiar nao e rarefeito, e o normal do papel.
    limiar = vol.where(vol > 0).quantile(cfg.percentil_volume_baixo)
    volume_raro = vol.lt(limiar, axis=1)

    repetido = pre.eq(pre.shift(1)) & pre.notna()
    # tamanho da sequencia de fechamentos identicos que termina em cada dia: o
    # acumulado de repeticoes menos o acumulado congelado na ultima quebra.
    acumulado = repetido.cumsum()
    desde_quebra = acumulado - acumulado.where(~repetido).ffill().fillna(0)
    travado = (desde_quebra + 1) >= cfg.max_precos_repetidos

    return (poucos_negocios.fillna(False) | sem_volume.fillna(False)
            | volume_raro.fillna(False) | travado.fillna(False))


def reestimar_sem_preco_velho(
    retornos: pd.DataFrame,
    mascara_preco_velho: pd.DataFrame,
    pares: pd.DataFrame,
    janela: Janela,
    cfg: LeadLagConfig | None = None,
) -> pd.DataFrame:
    """
    A mesma rede, sem os dias suspeitos. Queda material do beta e evidencia de
    sensibilidade a negociacao rarefeita; persistencia do beta, sozinha, nao
    prova que o fechamento estava atualizado.

    Um retorno liga dois pregoes, entao dia marcado invalida o retorno do
    proprio dia e o do seguinte, que usa aquele fechamento como base.
    """
    m = mascara_preco_velho.reindex(index=retornos.index,
                                    columns=retornos.columns, fill_value=False)
    contaminado = m | m.shift(1, fill_value=False)
    limpos = retornos.where(~contaminado)
    rede = leadlag.estimar_rede(limpos, pares, janela, cfg)
    rede["reestimativa"] = "sem_preco_velho"
    return rede


def subconjunto_sempre_negociado(
    volume: pd.DataFrame,
    janela: Janela,
) -> list[str]:
    """
    Tickers que negociaram em todos os pregoes do treino.

    Esse recorte elimina dias sem negocio, mas nao garante que o ultimo negocio
    ocorreu perto do fechamento. E uma robustez de liquidez mais exigente, com
    perda de amostra e de pares reportada junto.
    """
    vol = _recorte(volume, janela)
    sempre = (vol > 0).all()
    return sorted(sempre.index[sempre])


def rodar_placebos(
    retornos: pd.DataFrame,
    pares: pd.DataFrame,
    janela: Janela,
    cfg_placebo: PlaceboConfig | None = None,
    cfg_leadlag: LeadLagConfig | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Distribuicao nula por embaralhamento de seguidoras, para uma janela.

    Cada rodada re-sorteia as seguidoras dentro de (setor, subsetor) e estima
    os betas com `leadlag.betas_em_lote`. O sorteio preserva o grupo, mas quebra
    a orientacao por liquidez; grupos pequenos podem permanecer inalterados.

    Devolve a distribuicao (uma linha por embaralhamento) e o resumo com o
    p-valor empirico. Fracao alta de embaralhamentos batendo a mediana real
    indica que ela nao e excepcional sob este embaralhamento especifico.
    """
    from src import pares as mod_pares

    cfg_placebo = cfg_placebo or PlaceboConfig()
    cfg_leadlag = cfg_leadlag or LeadLagConfig()
    rng = np.random.default_rng(seed)
    treino = retornos.loc[(retornos.index >= janela.treino_inicio)
                          & (retornos.index < janela.treino_fim)]
    lag = cfg_leadlag.lag_principal
    min_obs = cfg_leadlag.min_observacoes_par

    reais = leadlag.betas_em_lote(
        treino, list(pares["lider"]), list(pares["seguidora"]), lag, min_obs)
    beta_real_mediano = float(reais["beta"].median())
    beta_real_medio = float(reais["beta"].mean())

    linhas = []
    for i in range(cfg_placebo.n_embaralhamentos):
        emb = mod_pares.pares_placebo_embaralhados(pares, rng)
        b = leadlag.betas_em_lote(
            treino, list(emb["lider"]), list(emb["seguidora"]), lag, min_obs)
        linhas.append({"embaralhamento": i,
                       "beta_mediano": float(b["beta"].median()),
                       "beta_medio": float(b["beta"].mean()),
                       "pares_estimados": int(b["beta"].notna().sum())})
    dist = pd.DataFrame(linhas)

    medianas = dist["beta_mediano"].dropna()
    resumo = {
        "janela": janela.rotulo,
        "n_embaralhamentos": int(cfg_placebo.n_embaralhamentos),
        "pares_reais_estimados": int(reais["beta"].notna().sum()),
        "beta_real_mediano": beta_real_mediano,
        "beta_real_medio": beta_real_medio,
        "placebo_mediana_p50": float(medianas.median()) if len(medianas) else float("nan"),
        "placebo_mediana_p95": float(medianas.quantile(0.95)) if len(medianas) else float("nan"),
        # frequencia de medianas placebo iguais ou maiores que a observada
        "p_empirico_mediana": float((medianas >= beta_real_mediano).mean())
        if len(medianas) else float("nan"),
    }
    return dist, resumo
