"""
Parametros do projeto V1 (rede lead-lag na B3).

So parametros: nenhuma regra de negocio, nenhum calculo, nenhum acesso a disco.
Tudo frozen para que ninguem mude um valor no meio da execucao e o resultado
passe a depender da ordem em que os modulos rodaram.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Literal

# Uma semente para o projeto inteiro (bootstrap, sorteio dos placebos).
SEED: int = 42


@dataclass(frozen=True)
class PeriodoConfig:
    """
    Recorte temporal do estudo.

    2015 em diante, e nao os ~2 anos do plano original: com 2 anos, o
    walk-forward de 24m/3m deixaria 3 ou 4 janelas fora da amostra, pouco para
    separar efeito de sorte depois da correcao de multiplos testes. O preco e
    mais troca de regime (2016, Covid, ciclo de juros) e mais vies de
    sobrevivencia.
    """

    inicio: date = date(2015, 1, 1)
    fim: date | None = None  # None = ate a ultima data disponivel nos dados

    # Robustez: repetir tudo so nos ultimos N anos, para ver se o efeito e
    # estavel ou so existe no passado distante.
    subamostra_recente_anos: int = 2


@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Como a janela caminha para frente no tempo.

    Tudo que decide o que sera operado na janela de teste (universo, pares,
    rede, FDR) sai apenas da janela de treino que a precede.
    """

    treino_meses: int = 24
    teste_meses: int = 3  # reestimacao trimestral

    # Robustez: mesmas janelas de teste, treino mais curto e mais longo.
    treinos_robustez_meses: tuple[int, ...] = (12, 36)

    # Sinal observado no fechamento de t, posicao aberta no fechamento de t+1.
    # Com 0 aqui o backtest negocia a um preco que ainda nao tinha visto.
    defasagem_execucao_dias: int = 1


@dataclass(frozen=True)
class LiquidezConfig:
    """
    Como o universo operavel e reconstruido dentro de cada janela de treino.

    O ranking e point-in-time: calculado ate o fim do treino e congelado
    durante o teste seguinte. Calculado sobre a amostra inteira, "as 40 mais
    liquidas" ja seria lookahead disfarcado de definicao de universo.
    """

    # Faixa principal da analise.
    top_n: int = 40

    # Faixas rodadas em paralelo no teste de nao-sincronia: efeito que so
    # aparece nas faixas mais amplas (menos liquidas) e suspeito.
    faixas: tuple[int, ...] = (20, 40, 60, 100)

    # Fracao minima de pregoes com dado valido para o ticker ser elegivel.
    cobertura_minima: float = 0.95

    # Fracao minima de pregoes em que o ticker realmente negociou (volume > 0)
    # dentro da janela de treino.
    proporcao_minima_dias_negociados: float = 0.95

    min_pregoes_treino: int = 400

    # Mediana e nao media: um unico dia de leilao gigante nao deve promover um
    # papel iliquido para o topo do ranking.
    estatistica_liquidez: Literal["mediana", "media"] = "mediana"

    # Um ticker por emissor, o mais liquido. PETR3 contra PETR4 como par de
    # lead-lag seria artefato de preco velho na forma mais pura.
    um_ticker_por_emissor: bool = True


@dataclass(frozen=True)
class NaoSincroniaConfig:
    """
    Controle central do projeto: separar difusao de informacao de atraso
    mecanico por falta de negociacao.
    """

    # Dia "sem negociacao" = volume financeiro zero. O numero de negocios
    # identifica sessoes rarefeitas, mas nao prova que o fechamento ficou velho:
    # o COTAHIST diario nao informa o horario do ultimo negocio.
    min_negocios_dia: int = 10

    # Abaixo deste percentil do volume do proprio ticker, o dia conta como
    # negociacao rarefeita.
    percentil_volume_baixo: float = 0.10

    # Numero minimo de fechamentos iguais em sequencia para marcar o dia. Com 2,
    # o segundo fechamento igual ja e sinalizado. A regra marca dias, nunca
    # reprova um ticker inteiro.
    max_precos_repetidos: int = 2

    # Acima desta fracao de pregoes com fechamento repetido a serie e tratada
    # como quebrada, e nao apenas iliquida. Limiar frouxo de proposito: a taxa
    # cai de 17,2% no decil menos liquido para 1,2% no mais liquido, entao 50%
    # so pega papel inutilizavel (RJCP3 tem 99,4%, com 155 fechamentos iguais
    # seguidos). Excluir os iliquidos aqui destruiria o teste central, que
    # precisa deles para mostrar se o efeito depende de preco velho - quem
    # controla exposicao a iliquidez e a faixa de liquidez, nao este filtro.
    max_frac_fechamento_repetido: float = 0.50


@dataclass(frozen=True)
class LeadLagConfig:
    """
    Regressao defasada: retorno_seguidora(t) ~ alpha + beta * retorno_lider(t-k).

    E o metodo mais simples e interpretavel que responde a pergunta. Granger
    fica como reforco opcional.
    """

    lag_principal: int = 1

    # Robustez. O lag 0 e diagnostico: se o contemporaneo domina, o que a
    # regressao defasada captura pode ser vazamento de correlacao comum.
    lags_robustez: tuple[int, ...] = (0, 2, 3)

    cov_type: Literal["HAC", "HC3", "nonrobust"] = "HAC"
    hac_maxlags: int = 5

    # Minimo de observacoes pareadas (lider defasada e seguidora validas no
    # mesmo dia) para estimar um par. Abaixo disso a estimativa e o HAC ficam
    # instaveis. O par reprovado continua na tabela com n
    # e sem beta, para a exclusao ficar contada.
    min_observacoes_par: int = 100


@dataclass(frozen=True)
class PlaceboConfig:
    """
    Dois testes que tentam derrubar o proprio resultado.

    - Embaralhamento: sorteia de novo quem segue quem dentro do mesmo grupo
      setorial. O teste verifica se a mediana real se destaca sob essa quebra
      especifica da orientacao lider-seguidora.
    - Direcao invertida: seguidora(t-1) -> lider(t). Difusao e assimetrica por
      definicao; se a iliquida "preve" a liquida com a mesma forca, nao ha
      direcionalidade nenhuma.
    """

    ativar: bool = True
    n_embaralhamentos: int = 500
    testar_direcao_invertida: bool = True


@dataclass(frozen=True)
class MultiplosTestesConfig:
    """
    Benjamini-Hochberg, aplicado dentro de cada janela de treino. Sobre a
    amostra inteira seria selecionar pares com informacao do futuro.
    """

    q_fdr: float = 0.10

    # Quais pares entram de fato na estrategia:
    #   fdr    so os que passam BH; conservador, pode nao sobrar nenhum
    #   top_k  os k melhores por estatistica t na janela de treino
    # Rodamos as duas: o FDR e a afirmacao inferencial, o top_k garante serie
    # de P&L para analisar mesmo quando o FDR zera.
    regras_selecao: tuple[str, ...] = ("fdr", "top_k")
    top_k_pares: int = 20


@dataclass(frozen=True)
class EstrategiaConfig:
    """Transformacao do sinal em posicoes."""

    holding_dias: int = 3

    # Safras sobrepostas: com holding de k dias, cada safra diaria recebe peso
    # 1/k. Da uma unica serie diaria de P&L sem multiplicar capital, e evita
    # tratar operacoes sobrepostas como observacoes independentes.
    safras_sobrepostas: bool = True

    # Menos dinheiro em papel agitado, mais em papel calmo, para uma acao so
    # nao dominar o resultado.
    sizing: Literal["vol_target", "igual"] = "vol_target"
    janela_vol_dias: int = 60
    vol_alvo_anual: float = 0.10

    # Teto de peso por posicao, como fracao da exposicao bruta.
    peso_maximo_por_posicao: float = 0.10

    # Long-short e o teste limpo do sinal; long-only mede implementabilidade
    # caso o aluguel inviabilize a ponta vendida. As pernas tambem saem
    # separadas, para revelar assimetria do efeito.
    modos: tuple[str, ...] = ("long_short", "long_only")


@dataclass(frozen=True)
class CustosConfig:
    """
    Custos de execucao. Valores em bps (1 bp = 0,01%).

    O resultado que interessa e o liquido destes custos: e ele que diz se a
    difusao encontrada e negociavel.
    """

    # Metade do spread de compra e venda, paga em cada ponta.
    meio_spread_bps: float = 5.0

    slippage_bps: float = 5.0

    # Corretora com taxa baixa.
    corretagem_bps: float = 0.0

    # Emolumentos + liquidacao da B3, por ponta.
    emolumentos_bps: float = 3.25

    # Aluguel (BTC), so na perna vendida. Sao cenarios, nao taxas historicas:
    # o projeto nao dispoe de serie point-in-time de taxa de aluguel.
    aluguel_cenarios_anual: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.20)
    aluguel_cenario_base: float = 0.05

    dias_uteis_ano: int = 252

    # IR entra como aproximacao ilustrativa, fora do resultado principal.
    ir_aliquota_ilustrativa: float = 0.15
    ir_incluir_no_principal: bool = False


@dataclass(frozen=True)
class MetricasConfig:
    """Avaliacao e teste de significancia."""

    benchmark_ticker: str = "^BVSP"  # Ibovespa
    taxa_livre_risco_anual: float = 0.0  # Sharpe sobre excesso simples

    # Bootstrap por blocos e a inferencia principal. Bootstrap iid trataria
    # dias vizinhos como independentes, o que e falso com operacoes
    # sobrepostas, e inflaria a significancia.
    bootstrap_n: int = 10_000
    # Bloco maior que o holding, para preservar a autocorrelacao que as safras
    # sobrepostas induzem.
    bootstrap_bloco_dias: int = 10
    bootstrap_alpha: float = 0.05

    # Newey-West entra como robustez, nao como numero principal.
    hac_maxlags: int = 10


@dataclass(frozen=True)
class DadosConfig:
    """
    De onde vem cada coisa.

    O COTAHIST da existencia do instrumento, volume financeiro em R$, numero de
    negocios e dias sem negociacao, e e point-in-time por construcao, porque
    inclui quem deixou de existir. So nao tem preco ajustado. O yfinance tem o
    ajuste por proventos, mas apaga a serie de quem saiu da bolsa: a cobertura
    do universo point-in-time cai de 99% em 2026 para 65% em 2017, ou seja, o
    vies de sobrevivencia entrava pela fonte de precos.

    Por isso estimacao e P&L usam paineis diferentes:

      estimacao  COTAHIST com as fronteiras de evento removidas. Mede retorno
                 de preco entre eventos, sem o degrau do provento contaminando
                 a regressao.
      P&L        COTAHIST bruto, sem mascara, com marcacao a ultimo preco
                 negociado. Cobre o universo inteiro, deslistados incluidos. Em
                 troca, o degrau do dia ex entra no retorno de preco sem o
                 fluxo do provento. A contagem de posicao-dias que cruzam
                 fronteiras mede a incidencia, nao a magnitude desse efeito.

    O yfinance fica como robustez da estimacao, em painel separado e restrito
    ao suporte comum. Num painel unico, a lider (liquida, sobrevivente, tem
    Yahoo) teria dado de qualidade diferente da seguidora (iliquida, pode ter
    morrido), e o lead-lag medido poderia ser so a diferenca de fonte.
    """

    # Preferencia para o universo: composicao oficial do IBrX 100
    # (BVBG.087.01), depois universo dinamico por liquidez do COTAHIST, e como
    # ultimo recurso a lista curada atual, com vies de sobrevivencia declarado.
    fonte_universo: Literal["ibrx_oficial", "cotahist", "lista_curada"] = "cotahist"

    fonte_retornos_estimacao: Literal["cotahist", "yfinance"] = "cotahist"
    fonte_retornos_estimacao_robustez: Literal["yfinance", "nenhuma"] = "yfinance"

    fonte_retornos_pnl: Literal["cotahist_bruto_marcacao"] = "cotahist_bruto_marcacao"

    # Mascarar tambem o pregao seguinte ao evento foi testado e descartado:
    # custa 0,87% dos retornos candidatos e nao muda erro mediano, p95 nem a
    # contagem de retornos extremos. So sai o retorno que atravessa a fronteira.
    mascarar_pregao_seguinte_ao_evento: bool = False

    sufixo_yfinance: str = ".SA"

    # COTAHIST: so mercado a vista, lote padrao.
    cotahist_tipo_mercado: tuple[str, ...] = ("010",)
    cotahist_codbdi: tuple[str, ...] = ("02",)

    # Timebox da tentativa de obter o IBrX historico oficial. Estourou, seguimos
    # pelo universo dinamico em vez de travar o MVP.
    timebox_ibrx_horas: float = 2.0


@dataclass(frozen=True)
class Config:
    """Configuracao de uma execucao, serializada para `config.json` na run."""

    seed: int = SEED
    periodo: PeriodoConfig = field(default_factory=PeriodoConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    liquidez: LiquidezConfig = field(default_factory=LiquidezConfig)
    nao_sincronia: NaoSincroniaConfig = field(default_factory=NaoSincroniaConfig)
    leadlag: LeadLagConfig = field(default_factory=LeadLagConfig)
    placebo: PlaceboConfig = field(default_factory=PlaceboConfig)
    multiplos_testes: MultiplosTestesConfig = field(default_factory=MultiplosTestesConfig)
    estrategia: EstrategiaConfig = field(default_factory=EstrategiaConfig)
    custos: CustosConfig = field(default_factory=CustosConfig)
    metricas: MetricasConfig = field(default_factory=MetricasConfig)
    dados: DadosConfig = field(default_factory=DadosConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# Para variar um parametro, use `dataclasses.replace` sobre esta copia - o
# objeto e frozen justamente para nao ser mutado no meio do caminho.
CONFIG_PADRAO = Config()
