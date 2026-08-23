<p align="center">
  <img src="assets/relatorio/mimir_mascote_v3.png" width="140" alt="Mascote do projeto Mímir">
</p>

# Mímir: lead-lag setorial na B3

A ideia do Mímir nasceu de uma observação do dia a dia: na natureza e na
sociedade, indivíduos observam e imitam aqueles que são bem-sucedidos ou
particularmente relevantes. Será que algo parecido acontece no mercado
financeiro? Quando um evento afeta uma empresa líder, as empresas relacionadas
a ela, as seguidoras, reagem? Existe um atraso? E, se existe, é possível
transformá-lo em uma oportunidade de investimento?

O projeto foi finalizado sem uma estratégia para implementar. Encontrei associação
entre ações no mesmo pregão, mas não um atraso persistente, robusto e capaz de
produzir retorno depois dos custos. A segunda etapa, condicionada a Fatos
Relevantes da CVM, também não passou pelos critérios definidos antes do
backtest. Para mim, esse é o resultado central do trabalho: uma hipótese
plausível pode sobreviver à intuição e ainda assim falhar quando precisa
enfrentar relógio, dados, controles estatísticos e custos reais.

Ainda assim, eu voltaria a essa pergunta se tivesse acesso a horários
intradiários auditáveis. Com os dados disponíveis, um evento datado em `D` só
podia ser considerado conhecido na primeira abertura seguinte. Isso evita
look-ahead, mas pode deixar a reação inicial para trás. Com timestamps
confiáveis, seria possível testar a sequência real entre líder e seguidora sem
impor esse atraso por construção.

O [relatório final](docs/relatorio_final.pdf) reúne o desenho, os gráficos e os
resultados usados na conclusão.

## A pergunta

A hipótese econômica era que as ações mais líquidas incorporariam informação
antes das menos líquidas do mesmo setor. Chamei a ação mais líquida de líder e
a menos líquida de seguidora. Como os dados não informavam o horário dos
eventos, eu só poderia negociar depois que a informação estivesse seguramente
disponível. Por isso, o atraso precisaria sobreviver até o pregão seguinte,
reaparecer fora da amostra e produzir retorno depois dos custos.

O nome vem da mitologia nórdica: Mímir guarda o poço da sabedoria. No projeto,
os movimentos das ações mais líquidas seriam sinais para as seguidoras. Esses
sinais só teriam utilidade se superassem os custos de negociação.

A criação de um mascote era um dos requisitos da competição. O grupo decidiu
usar a história de Mímir para representar o projeto, e a imagem do mascote foi
criada com o GPT a partir dessa ideia.

## O que foi testado

### V1: rede agregada

A V1 reconstrói o universo a cada trimestre usando apenas os 24 meses
anteriores. Dentro de cada setor e subsetor, os pares são direcionados da ação
mais líquida para a menos líquida. A rede fica congelada durante os três meses
de teste.

Foram 38 janelas fora da amostra, de 2017-Q1 a 2026-Q2. O universo principal da
V1 é o top 40 por liquidez, com 1.247 observações par-janela e 131 pares únicos.
Para cada par, uma regressão independente estima os lags de zero a três
pregões. O teste principal está em `k=1`; `k=0` é diagnóstico, porque pode
captar uma notícia ou um fator comum às duas empresas no mesmo dia.

O desenho também inclui:

- universo e classificação setorial point-in-time;
- erros HAC(5) e mínimo de 100 observações por regressão;
- Benjamini-Hochberg a 10%, aplicado separadamente em cada janela;
- filtros de não sincronia e preço stale;
- redes placebo dentro do mesmo setor;
- execução walk-forward com spread, slippage, emolumentos e aluguel.

### V2: eventos e classificação textual

A V2 pergunta se um Fato Relevante sobre a líder ajuda a encontrar difusão que
o teste agregado não conseguiu identificar. Ela usa os pares estruturais do
top 20, sem escolher relações pelo retorno ou pelo valor-p observado na V1.

Os documentos vêm da CVM IPE. Como a fonte informa o dia de entrega, mas não um
horário auditável, um documento datado em `D` só é considerado conhecido na
primeira abertura da B3 depois de `D`. A regra pode perder parte da reação
inicial, mas evita inventar um timestamp e introduzir look-ahead.

Um Qwen3:14B local recebeu somente o texto dos documentos e classificou cada
evento como positivo, negativo, neutro ou não específico. Preços, retornos e
P&L não entraram no prompt. Antes de olhar o backtest, uma amostra cega de 90
documentos foi comparada com um painel de referência multi-modelo e uma leitura
humana.

O gate reprovou: o macro-F1 das classes com suporte foi 0,688, abaixo do limite
de 0,70, e apenas três das quatro classes tiveram suporte. Uma emenda registrada
antes do primeiro P&L permitiu rodar o backtest apenas como diagnóstico
exploratório, sem promovê-lo a evidência confirmatória.

## Resultado

| Teste | Resultado principal |
|---|---:|
| V1, beta mediano contemporâneo (`k=0`) | +0,617 no top 40; +0,679 no top 20 |
| V1, beta mediano com um pregão de atraso (`k=1`) | -0,015 no top 40; -0,020 no top 20 |
| V1, placebo setorial | p mediano 0,632; nenhuma das 38 janelas abaixo de 5% |
| V1, FDR long-only | +0,15% bruto; -1,21% líquido |
| V1, FDR long-short | -5,64% bruto; -8,63% líquido |
| V2, gate do classificador | macro-F1 0,688; reprovado |
| V2, horizonte principal H=3 | 99 operações; -0,08% líquido; Sharpe -0,013 |
| V2 contra 500 redes aleatórias | p de randomização 0,319 |

Na V1, as aprovações estatísticas apareceram apenas a partir de 2023, ficaram
concentradas em poucas relações e foram, em sua maioria, negativas. O efeito
forte está em `k=0`; nos pregões seguintes, ele desaparece.

Na V2, todos os intervalos de 95% cruzam zero. O horizonte H=1 ficou positivo,
mas era apenas um teste de robustez. Substituir H=3 por H=1 depois de ver o
resultado seria data snooping. A decisão, portanto, foi encerrar o
desenvolvimento do sinal e não implementá-lo.

Isso não prova que lead-lag não exista na B3. Mostra apenas que ele não foi
verificável nem negociável com os dados, o relógio e a estratégia utilizados.

## Dados

- **B3 COTAHIST:** identidade do ativo, calendário, volume, número de negócios e
  preços brutos. É a fonte que define o universo.
- **Yahoo Finance:** conferência auxiliar com preços ajustados. Nunca define
  identidade ou elegibilidade.
- **B3 e CVM/FCA:** ISIN, histórico setorial e tratamento de empresas
  deslistadas.
- **CVM IPE:** apresentações originais dos Fatos Relevantes usados na V2.

Os arquivos brutos e derivados não são versionados. As pequenas tabelas de
referência que fazem parte da metodologia ficam em `data/reference/`, com fonte
e validade temporal documentadas. Cada execução cria uma pasta própria em
`outputs/runs/`, contendo parâmetros, manifesto, tabelas, figuras e logs.

## Como executar

O ambiente final usou Python 3.12.
Em PowerShell:

```powershell
git clone https://github.com/jotave-br/projeto-itau-quant-2026.git
cd projeto-itau-quant-2026
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

A V1 tem um orquestrador único:

```powershell
python scripts/run_pipeline.py --rotulo oficial
```

As auditorias que dependem do Yahoo podem ser omitidas com
`--sem-auditorias-yf`, sem alterar o caminho principal baseado no COTAHIST.

A V2 é deliberadamente separada em etapas, porque há coleta documental,
classificação local e composição do painel entre o corpus e o backtest:

```powershell
python scripts/v2_01_coletar_ipe.py --ano-final 2026
python scripts/v2_02_preparar_documentos.py --run-v1 outputs/runs/<run_v1>
python scripts/v2_03_classificar_eventos.py --modelo qwen3:14b
python scripts/v2_04_preparar_validacao_humana.py
python scripts/v2_06_consolidar_painel.py
python scripts/v2_07_backtest_eventos.py `
  --rede outputs/runs/<run_v1>/tabelas/rede_por_janela.csv `
  --universo outputs/runs/<run_v1>/tabelas/universo_por_janela.csv
```

O classificador requer uma instalação local do Ollama com o modelo congelado.
A consolidação também depende das fichas do painel preenchidas conforme o
[protocolo de rotulagem](docs/PROTOCOLO_ROTULAGEM_V2.md). A
[especificação da V2](docs/ESPECIFICACAO_V2.md) registra as hipóteses, o gate e
a emenda que tornou o backtest exploratório.

O relatório pode ser reconstruído por `scripts/gerar_relatorio_final.py`. O
script aponta para as execuções canônicas usadas na entrega; ao gerar uma nova
rodada, esses caminhos precisam ser atualizados para os novos manifestos.

## Organização do repositório

- `src/`: regras de dados, universo, regressões, controles e backtest da V1;
- `src/v2/`: coleta documental, classificação, validação e backtest de eventos;
- `scripts/`: entradas executáveis das duas etapas;
- `data/reference/`: decisões curadas e versionadas;
- `docs/`: relatório, especificação e protocolo de rotulagem;
- `tests/`: testes unitários, anti-lookahead e invariantes do backtest;
- `outputs/runs/`: resultados locais, ignorados pelo Git.

## Limitações que mudam a interpretação

Os dados diários não permitem saber quem reagiu primeiro dentro do mesmo
pregão. O beta contemporâneo não identifica causalidade e pode refletir um
fator comum. As regressões não controlam por mercado, setor ou lags próprios da
seguidora. O P&L usa retorno de preço, não retorno anormal ou retorno total
com proventos.

Na V2, o gate do classificador falhou, as 99 posições não são independentes e
não foi calculado poder estatístico ou efeito mínimo detectável. A saída por
reversão descrita no protocolo não foi implementada. Além disso, concordância
entre modelos de linguagem mede convergência, não verdade: eles podem
compartilhar dados de treino e erros correlacionados.

Essas limitações não foram usadas para salvar a estratégia; são parte do motivo
para abandoná-la.

## Uso de IA

O Qwen3:14B local foi o classificador textual da V2. Outros modelos auxiliaram
no parsing dos documentos, na composição cega do painel, na criação do mascote
e na conferência independente de cálculos. Uma leitura humana participou do
painel e dos desempates. Depois do congelamento dos rótulos, as etapas
quantitativas foram determinísticas.

## Autoria e contribuições

As responsabilidades foram divididas de acordo com a especialidade de cada
integrante. Eu fiquei responsável pela parte técnica do projeto: busca e seleção
das fontes, obtenção e tratamento dos dados, regras point-in-time, controles
contra look-ahead, implementação do código, testes estatísticos, backtests e
análise dos resultados.

O relatório final e a preparação da apresentação foram trabalhos conjuntos. Os
demais integrantes contribuíram especialmente para adequar o conteúdo aos
requisitos da competição e organizar a comunicação para a banca.

## Licença e citação

O projeto está sob a licença [MIT](LICENSE). Os metadados para citação estão em
[`CITATION.cff`](CITATION.cff).
