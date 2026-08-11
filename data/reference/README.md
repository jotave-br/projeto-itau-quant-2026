# Tabelas de referência

Arquivos **de entrada** da metodologia: decisões curadas que o pipeline lê, não
resultados que ele produz. Por isso são versionados no git — ao contrário de
`data/raw/` e `data/processed/`, que são baixados/derivados e ficam fora.

Resultados derivados (tabelas de frequência, auditoria por ticker, contagem de
instrumentos filtrados) são **regerados a cada execução** dentro de
`outputs/runs/<id>/tabelas/`. Não congelamos essas tabelas aqui de propósito:
elas envelheceriam em silêncio, e um arquivo congelado que não corresponde mais
aos dados é pior que nenhum arquivo.

---

## `classificacao_instrumentos.csv`

Quais instrumentos entram no universo e quais ficam de fora, com a justificativa
de cada decisão.

**Data da decisão:** 2026-07-28
**Base empírica:** COTAHIST anual da B3, 2015 a 2026 (12 arquivos)
**Fonte dos campos:** layout oficial do COTAHIST (registro tipo 01, 245 colunas)

Um instrumento só é aceito se **duas fontes independentes concordarem**: o
primeiro token normalizado do campo `ESPECI` precisa estar na allowlist **e** o
tipo codificado no ISIN (posições 7-9) precisa confirmar que é ação. Se
discordarem, o instrumento fica de fora — preferimos perder um papel a
contaminar a amostra.

**Ordem de classificação aplicada em `src/dados.py`:**

1. ISIN diz que não é ação (`BDR`, `CTF`, `IND`) → bloqueia
2. `ESPECI` na allowlist **e** ISIN confirma ação → aceita
3. `ESPECI` na denylist → bloqueia
4. Nenhum dos anteriores → **erro**, pipeline para e pede classificação humana

O passo 4 não é enfeite. Foi ele que revelou duas categorias que não estavam
previstas: `DRE` (BDR de ETF, criada pela B3 em 2022) e `SML)` (índice small
cap, 2024). Sem ele, ambas teriam entrado na amostra sem ninguém notar.

### Por que não filtrar por sufixo do ticker

É a armadilha mais tentadora e está errada:

| Ticker | Termina em | O que é |
|---|---|---|
| `SANB11`, `TAEE11`, `ALUP11` | 11 | **Unit** — ação legítima |
| `BRGE11` | 11 | **Preferencial classe E** — ação legítima |
| `BRGE12` | 12 | **Preferencial classe F** — ação legítima |
| `BOVA11` | 11 | ETF |
| `IBOV11`, `SMLL11` | 11 | Instrumento de índice |

Uma regra "termina em 11 é ETF" jogaria fora ações brasileiras legítimas.

### Por que o filtro de `CODBDI` não basta

BDR é negociado em **lote padrão** (`CODBDI = 02`) até 2022 — passa inteiro
pelo filtro. Em 2015 são ~90 dos 555 tickers do pool, 16% do universo
candidato. Em 2023 a B3 migrou os BDRs para os códigos 34 e 36.

**Nenhum campo isolado é estável ao longo de doze anos.** Por isso os filtros
coexistem: `ESPECI`/ISIN carrega o peso de 2015 a 2022, `CODBDI` cobre de 2023
em diante.

### Normalização do `ESPECI`

O campo é composto (`tipo` + `marcador de evento` + `segmento de governança`) e
muda para o mesmo ticker no dia ex-dividendo:

```
"ON      NM"  ->  ON      ordinária, Novo Mercado
"ON  ED  NM"  ->  ON      mesma ação, dia ex-dividendo
"UNT     N2"  ->  UNT     unit, Nível 2
"IBO/"        ->  IBO     índice (a B3 escreveu o rótulo truncado)
"SML)"        ->  SML     índice small cap (idem)
```

Usamos o **primeiro token, sem pontuação e em maiúsculas**. Isso evita tanto a
quebra intermitente (comparar a string inteira falharia só nos dias de evento)
quanto o jogo de catalogar cada pontuação que a bolsa resolver usar.

---

## `mudancas_ticker.csv`

Camada de **mapeamento** entre códigos de negociação da mesma empresa antes e
depois de um evento societário. **Não altera os dados brutos** — o COTAHIST
preserva sempre o ticker original, e esta tabela é consultada por cima.

**Status: 28 linhas pendentes de revisão manual.** Nenhuma está autorizada.

### Por que existe

A mesma empresa muda de código no meio da amostra e a série quebra em dois
pedaços que o pipeline trataria como empresas diferentes. Casos reais na
amostra, todos entre os 100 mais líquidos de alguma janela:

```
KROT3 -> COGN3   Kroton vira Cogna          (posição 11)
BRFS3 -> MBRF3   BRF incorporada à Marfrig  (posição 11)
ELET3 -> AXIA3   Eletrobras vira Axia       (posição 11)
EMBR3 -> EMBJ3   Embraer troca de código    (posição 16)
CCRO3 -> MOTV3   CCR vira Motiva            (posição 18)
```

O código de emissor do ISIN, que resolve PETR3 × PETR4, **não resolve isto**:
`EMBR` e `EMBJ` são emissores distintos para ele.

### Dois identificadores, não um

Um único id não pode representar ao mesmo tempo continuidade de instrumento e
linhagem societária — são conceitos diferentes e confundi-los autorizaria
concatenações erradas.

| Campo | O que identifica | Quem preenche |
|---|---|---|
| `canonical_instrument_id` | Série econômica que **pode ser continuada** | **Humano.** Nasce em branco — é ele que autoriza concatenar preços |
| `suggested_corporate_lineage_id` | Pista automática, do grafo **completo** de candidatos | Detector. Não autoriza nada |
| `corporate_lineage_id` | Linhagem societária confirmada, **incluindo fusões** | **Reconstruída automaticamente** a partir de arestas aprovadas |

Como isso resolve os casos:

```
ELET3 -> AXIA3            mesmo instrument_id, mesma linhagem
VVAR3 -> VIIA3 -> BHIA3   mesmo instrument_id nos três, mesma linhagem
BRFS3 -> MBRF3            linhagem compartilhada com MRFG3 -> MBRF3,
MRFG3 -> MBRF3            mas instrument_id SEPARADOS: é fusão, não renomeação
falso positivo            ids separados nos dois campos
```

**Não reatribua linhagem à mão.** A linhagem sugerida propaga através de falsos
positivos — `ESTC3→YDUQ3`, `ESTC3→ALSO3` e `ALSO3→ALOS3` compartilham a
sugestão `ALOS3` porque o falso positivo do meio conecta os dois grupos. Basta
marcar `ESTC3→ALSO3` como `falso_positivo`: na execução seguinte o script
reconstrói os componentes conectados **usando somente arestas aprovadas**, e a
linhagem confirmada sai separada (`ESTC3` de um lado, `ALOS3` do outro).

Uma aresta entra na linhagem confirmada quando tem `status` decidido **e**
`tipo_evento` em `rename_1to1`, `conversao_classe` ou `fusao_incorporacao`.
Falso positivo e não confirmado não geram aresta.

### Proteção da revisão

| Campo | Para quê |
|---|---|
| `detector_version` | Versão do detector que gerou os diagnósticos (hoje `1.0.0`, **congelado**) |
| `diagnostic_hash` | Impressão digital dos números que sustentam a decisão |
| `reviewed_detector_version` | Versão vigente quando a linha foi revisada |
| `needs_revalidation` | `sim` quando o diagnóstico mudou depois da revisão |

Três garantias, todas testadas:

1. **Campos manuais nunca são sobrescritos** — qualquer valor preenchido é
   preservado, mesmo em linha ainda pendente.
2. **Decisão órfã não é apagada** — linha já decidida que saia da detecção
   permanece no arquivo, marcada para revalidação.
3. **Diagnóstico desatualizado é sinalizado, não escondido** — se os números
   mudarem depois da revisão, a linha recebe `needs_revalidation=sim`, mas o
   `status` decidido **não** é alterado automaticamente. Quem decide continua
   sendo a pessoa.

### `tratar_retorno_fronteira`

Define o que fazer com o retorno **entre o último pregão do código antigo e o
primeiro do novo** — o único ponto onde a emenda pode inventar um número.

| Valor | Quando |
|---|---|
| `pendente_revisao` | **Valor inicial.** Ainda não decidido |
| `calcular_1to1` | `rename_1to1` confirmado por fonte oficial |
| `ajustar_por_conversao` | `conversao_classe` **com** `proporcao_conversao` confirmada |
| `definir_nan` | `conversao_classe` sem proporção confirmada |
| `nao_aplicavel` | **Decisão explícita**: falso positivo, relação rejeitada, ou evento que não autoriza continuidade |

`nao_aplicavel` é uma decisão, não um estado inicial — por isso as linhas
nascem em `pendente_revisao`.

**Política da V1:** só `rename_1to1` oficialmente confirmado calcula o retorno
de fronteira. `conversao_classe` sem proporção vira `NaN` — perder uma
observação é muito melhor que fabricar um retorno falso. `fusao_incorporacao`
nunca concatena mecanicamente.

Exemplo concreto do risco: `STBP11 → STBP3` tem razão de preços de **0,19**,
perto de 1/5, porque a unit continha cinco ações. Concatenar direto criaria uma
queda de 80% que nunca existiu.

### Prioridade e status

`prioridade` ordena a pesquisa documental: **alta** (alguma ponta no top-50, ou
ponta com histórico curto), **media** (top-100), **baixa** (só a faixa de
margem).

`status` distingue o que ainda não foi olhado do que foi pesquisado sem
sucesso:

| Status | Significado |
|---|---|
| `pendente` | Ainda não pesquisado, prioridade alta ou média |
| `pendente_prioridade_baixa` | Ainda não pesquisado, pode esperar |
| `nao_confirmado` | **Efetivamente pesquisado**, sem confirmação suficiente |
| `confirmado` | Fonte oficial registrada em `fonte` |

### Classificação obrigatória (`tipo_evento`)

| Valor | Significado | Pode concatenar? |
|---|---|---|
| `rename_1to1` | Mudança de nome/código, mesma classe, continuidade 1:1 | **Sim**, se confirmado por fonte oficial |
| `conversao_classe` | Troca de classe (unit→ON, PN→UNT). Exige proporção e direitos explícitos | Não, até haver regra documentada |
| `fusao_incorporacao` | Duas empresas viram uma | **Nunca** como troca simples |
| `falso_positivo` | Adjacência por acaso | Não |
| `nao_confirmado` | Ainda sem fonte oficial | Não |

**Somente `rename_1to1` com `status=confirmado`, `continuidade_autorizada=sim`
e `fonte` preenchida pode ser concatenado** para retorno e cobertura. Nos
demais, as séries permanecem separadas.

Continuidade de preço serve para **priorizar** candidatos. Não substitui
confirmação societária.

### Como atualizar

```bash
.venv\Scripts\python.exe scripts\gerar_mapeamento_ticker.py
```

O script **preserva revisão humana**: uma linha com `status` diferente de
`pendente*` nunca é sobrescrita, e linhas já decididas que saírem da detecção
permanecem no arquivo. Só as colunas de diagnóstico (posições, preços,
liquidez) são regeradas, e apenas nas linhas ainda pendentes.

### Regra anti-lookahead

O mapeamento pode ser conhecido retrospectivamente para corrigir a
**identidade** do ativo. Ele **não** pode fazer a liquidez futura do código
novo ajudar a selecionar o código antigo.

Concretamente: o ranking de liquidez de cada janela usa apenas os dados que já
existiam até aquela data, e **volumes de tickers distintos nunca são somados**
no ranking. Depois de confirmada uma troca 1:1, o `canonical_asset_id` serve
para medir cobertura contínua, impedir que a mesma empresa seja tratada como
duas entidades e escolher a classe mais líquida do emissor — nunca para
antecipar liquidez.

### Preservação da informação original

Nos dados processados, a série canônica **não apaga** o observado. São
mantidos `ticker_observado`, `canonical_asset_id`, `data`, `fonte` e o eventual
fator de conversão.

### Campos

`canonical_asset_id`, `ticker_antigo`, `ticker_novo`, `nome_antigo`,
`nome_novo`, `isin_antigo`, `isin_novo`, `especie_antiga`, `especie_nova`,
`ultimo_pregao_antigo`, `primeiro_pregao_novo`, `intervalo_pregoes`,
`razao_precos`, `melhor_posicao_antigo`, `melhor_posicao_novo`,
`criterio_relevancia`, `metodo_deteccao`, `tipo_evento`,
`proporcao_conversao`, `continuidade_autorizada`, `fonte`, `data_confirmacao`,
`revisor`, `status`, `observacoes`

### Como os 28 foram selecionados

De 55 candidatos brutos, ficaram os **materialmente relevantes** por critério
*point-in-time* — nunca por liquidez média dos 12 anos, que mistura épocas e
favorece quem listou recentemente num período de volume alto. Entra na revisão
quem: (a) entrou no top-100 de alguma janela de treino; (b) alcançou o top-120,
como margem; ou (c) tem uma ponta com histórico curto demais para ser rankeada
enquanto a outra é relevante — provavelmente elegível se a série fosse
corretamente continuada.

O critério (c) cobre o caso mais perigoso: o código novo com poucos pregões
reprova por cobertura e **some do universo** sem que ninguém note que a empresa
continua ali.

---

## `setores_b3.csv` — classificação setorial point-in-time

Tabela curada em 07/08/2026, com **192 linhas para 189 tickers**. Duas linhas
extras registram mudanças reais de grupo em HYPE3 e RAIZ4; a terceira preserva
explicitamente a única lacuna de evidência, ITSA4 entre 24 e 27/07/2026. A
unidade preservada é o ticker observado; fusão, incorporação, migração de unit
e mudança de código não herdam setor silenciosamente.

### Fontes e reconciliação

- **117 tickers da fila atual:** ligação exata pelo código de emissor à
  [classificação setorial da B3](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/acoes/consultas/classificacao-setorial/),
  acessada em 07/08/2026. Arquivo `ClassifSetorial.xlsx`, SHA-256
  `AA2595618AD433E663F7EB66AAA44EDD9035BA2AED2108980BB46C21BC44D069`.
- **41 tickers históricos:** ligação exata ao arquivo oficial legado
  [`ClassifSetorial.zip`](https://bvmf.bmfbovespa.com.br/InstDados/InformacoesEmpresas/ClassifSetorial.zip),
  planilha `Setorial B3 15-09-2022 (português).xlsx`, SHA-256
  `6127B9F03E2D123F821D2816E535E776620A80EAEA73602EF11D22C7632DDC21`.
- **31 tickers ausentes dos dois recortes aplicáveis:** curadoria histórica
  com fotografias da lista B3/BM&FBovespa, páginas de companhias, editais de
  OPA e eventos societários. Cada linha traz a URL do documento que sustenta
  aquela classificação, reconciliado com os
  [FCA anuais da CVM](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/)
  e com o cadastro de companhias abertas.

Os arquivos-fonte brutos não são versionados. O CSV registra em cada linha a
fonte, a data, a evidência, a confiança e o intervalo de validade; os hashes
acima permitem conferir as duas fotografias oficiais usadas. Em
`data_fonte`, 07/08/2026 significa data de acesso/revisão quando a publicação
não traz uma data própria.

As principais fotografias históricas auxiliares foram a
[lista B3 de 21/12/2016](https://pco.uem.br/0-arquivos/dissertacoes/2018_mara-cristina-piovesan-cortezia.pdf),
a [lista BM&FBovespa de 24/11/2014](https://repositorio.ufpb.br/jspui/bitstream/123456789/1880/1/GKBG30082017.pdf)
e o [histórico oficial de adequações da taxonomia](https://www.b3.com.br/data/files/2C/83/3A/8A/15CB7610F157B776AC094EA8/Historico-Adequacoes-Metodologias-Nov2018.pdf).
A classificação segue o
[critério econômico declarado pela B3](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/acoes/consultas/criterio-de-classificacao/);
nada foi inferido de retorno, correlação ou P&L.

### Decisões temporais especiais

- **HYPE3:** Produtos Diversos até 31/12/2017 e Saúde a partir de 01/01/2018.
  A mudança está interval-censored: uma fotografia B3 de janeiro–abril de 2017
  ainda mostra o grupo antigo, enquanto o foco exclusivamente farmacêutico já
  era público em dezembro de 2017 e fotografias posteriores mostram o novo.
  A fronteira anual foi fixada sem consultar resultados e tem confiança média.
- **RAIZ4:** Agricultura até 31/12/2023 e Petróleo a partir de 01/01/2024.
  Agricultura está documentada em 10/07/2023 e a classificação de Petróleo em
  base B3 de 27/03/2024. A fronteira anual, também interval-censored, tem
  confiança média.
- **ITSA4:** Bancos até 23/07/2026. A primeira fotografia disponível como
  holding é de 07/08/2026; como o dia exato da reclassificação não foi
  localizado, 24–27/07 permanece `evidencia_insuficiente`. São dois pregões no
  fim da amostra em que ITSA4 fica fora dos pares, em vez de receber uma classe
  não demonstrada.
- **BBTG11:** permanece como observação própria até 18/08/2017. A migração
  automática posterior para BPAC11/BBTG12 não apaga nem transfere o histórico.

Alterações apenas nominais da taxonomia foram harmonizadas para o vocabulário
atual — por exemplo, `Comércio` → `Comércio Varejista`, `Transporte Aéreo` →
`Linhas Aéreas de Passageiros` e o rótulo abreviado de serviços de saúde. Isso
não cria mudança econômica. Os grupos da V1 usam `(setor, subsetor)`;
`segmento` é informativo. `validade_inicio` e `validade_fim` delimitam somente
o período observado no COTAHIST, não toda a existência jurídica da companhia.

## Pendentes

- `emissores_b3.csv` — **provavelmente desnecessária.** O código de emissor está
  nas posições 3-6 do ISIN e agrupa corretamente as classes da mesma empresa
  (`BRPETR…` → PETR3 e PETR4; `BRSANB…` → SANB3, SANB4 e SANB11). Sendo campo
  oficial, é preferível a uma tabela curada à mão.
- `composicoes_indices/` — composição histórica oficial do IBrX 100. Seria a
  solução direta para o viés de sobrevivência, mas não foi obtida. O universo
  dinâmico por liquidez reconstruído do COTAHIST é o proxy point-in-time
  adotado. Limitação declarada no README principal.
