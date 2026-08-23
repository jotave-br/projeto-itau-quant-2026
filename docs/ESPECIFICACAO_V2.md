# Especificação V2 — lead-lag condicionado a Fato Relevante

## Status e pergunta

Este documento é um pré-registro: as regras abaixo devem ser congeladas antes
de calcular o primeiro resultado financeiro da V2.

**Pergunta:** um Fato Relevante direcional sobre a líder revela difusão
condicional de informação para a seguidora nos pares estruturais *point-in-time*
do top 20 da V1?

A hipótese principal é que, após um evento positivo ou negativo na líder, a
carteira de seguidoras se mova na mesma direção e produza retorno líquido médio
positivo no horizonte de três pregões. A hipótese nula é retorno líquido médio
menor ou igual a zero. Um resultado nulo é compatível com a conclusão da V1 e
não autoriza mudar universo, prompt, horizonte ou regra de seleção.

### Registro de execução — 13/08/2026

O gate de IA reprovou e o backtest foi executado apenas como exploração, sob a
exceção já documentada. A execução canônica usa exclusivamente as 452 arestas
top 20, H=3 como principal e H=1/H=5 como robustez. O controle “rede sem IA” usa
o sinal diário do retorno da líder da V1; o placebo randomiza 500 redes entre
seguidoras top 20 do mesmo setor, sem self-pairs, e aplica a correção finita do
valor-p abaixo. A saída por reversão, embora prevista como diagnóstico, não foi
executada e não integra os resultados ou o relatório. Este registro descreve o
que foi rodado sem alterar a hipótese principal pré-registrada.

## Rede preservada da V1

Em cada janela, o universo continua sendo reconstruído apenas com informação
anterior ao teste. Entram os 20 emissores mais líquidos, uma classe por emissor.
Dentro de cada setor e subsetor, a ação mais líquida é a líder e a menos líquida
é a seguidora.

Serão usados **todos os pares estruturais elegíveis**, sem filtro por retorno,
beta, valor-p, top-k ou FDR. A rede é formada antes dos eventos do período de
teste e fica congelada. Ela é um mapa econômico de transmissão, não uma coleção
de relações já comprovadas pela V1.

Um evento da líder pode alcançar várias seguidoras. Nesse caso, uma unidade de
exposição é dividida igualmente entre elas, antes dos limites de posição da V1,
para que uma líder com mais pares não receba mais capital por construção.
Quando mais de uma líder gera sinal para a mesma abertura, as contribuições são
somadas por seguidora, direções opostas são compensadas e a safra resultante é
normalizada para exposição bruta um. Aplica-se então o teto por ação e o peso
`1/H` usado nas safras sobrepostas da V1, sem renormalizar depois do teto.

## Eventos e classificação por IA

A única fonte textual será a base de Fatos Relevantes da CVM. Não entram notícias,
redes sociais ou fatos posteriores. A coleta preserva todas as entregas para
auditoria, mas o sinal usa somente a apresentação original; uma retificação não
é retroagida nem contada como um novo choque. O vínculo entre companhia, ticker
e data deve ser verificável.

O modelo generativo recebe somente o texto disponível no documento. Não recebe
preços, retornos, volume, seguidoras, P&L ou qualquer informação posterior. Com
prompt, modelo, versão e parâmetros congelados, ele devolve uma saída estruturada:

- efeito fundamental esperado da informação sobre o valor das ações da líder:
  `positiva`, `negativa` ou `neutra/incerta`;
- uma justificativa curta, apoiada em trecho do próprio documento.

Documentos neutros ou incertos geram abstenção. Vários documentos da mesma
líder na mesma `Data_Entrega` formam um único líder-dia. Se os rótulos
direcionais concordarem, há um único sinal; se houver conflito, o sistema se
abstém. Não será usada pontuação de confiança ajustável.

## Relógio, entrada e saída

A CVM fornece `Data_Entrega` sem horário. Para não supor que o documento estava
disponível antes da abertura daquele dia, um fato datado em `d` só é considerado
conhecido imediatamente antes da primeira abertura da B3 com data estritamente
posterior a `d`. A entrada ocorre nessa abertura; nunca em `d`.

O sinal positivo compra as seguidoras e o negativo as vende. O horizonte
principal é **H = 3 pregões**: entrada na abertura do primeiro pregão elegível e
saída no fechamento do terceiro. **H = 1** e **H = 5** são robustezes
pré-declaradas e não podem substituir o resultado principal.

Haverá ainda uma saída exploratória por reversão da líder. Depois da entrada,
se o retorno diário da líder fechar com sinal contrário ao rótulo do evento, a
posição é encerrada na abertura seguinte. Se não houver reversão, a posição é
encerrada, no máximo, no fechamento do quinto pregão. Essa regra será reportada
separadamente e não sustentará a conclusão principal.

Posições simultâneas são agregadas antes dos limites de exposição. Spread,
slippage, emolumentos, corretagem, aluguel, dimensionamento e limites serão os
mesmos da V1, sem recalibração a partir dos resultados da V2.

## Comparações que precisam sobreviver

O teste principal é **IA + rede estrutural**. Três comparações separam os dois
componentes:

1. **IA sem rede:** o mesmo sinal textual é aplicado à própria líder, como teste
   de continuação do choque sem usar as ligações líder-seguidora.
2. **Rede sem IA:** os mesmos pares seguem a regra de sinal por retorno da líder
   já usada na V1, sem condicionar a Fato Relevante.
3. **Seguidora aleatória:** preservam-se evento, líder, setor, quantidade de
   posições e faixa de liquidez, mas a seguidora é sorteada entre candidatas
   top 20. O valor-p de randomização usará a correção `(extremos + 1) / (B + 1)`.

O número de eventos, líderes, líder-dias, pares alcançados, abstenções e conflitos
será informado. O desfecho principal é o P&L líquido diário em H = 3, com
inferência por blocos para acomodar posições sobrepostas. Retorno bruto, pernas,
H = 1, H = 5 e a saída por reversão são diagnósticos ou robustezes.

Só haverá evidência de difusão condicional se a estratégia principal for
positiva após custos, tiver intervalo de 95% acima de zero e superar a seguidora
aleatória. Caso contrário, a conclusão será nula ou inconclusiva, sem busca por
subgrupos vencedores.

## Validação da IA e limite retrospectivo

Antes do P&L, dois avaliadores humanos rotularão independentemente uma amostra
cega de 90 documentos, com 30 por direção prevista; casos não específicos terão
prioridade dentro da cota neutra. Divergências serão
adjudicadas por uma terceira leitura cega. Serão publicados matriz de confusão,
macro-F1, concordância entre humanos, sucesso técnico e taxa de abstenção. A
amostra é um teste de estresse das decisões, não uma estimativa da
prevalência no corpus. A classificação é utilizável com macro-F1 de pelo menos
0,70 nas quatro classes conjuntas, presença das quatro classes na previsão e no
gold, e kappa de Cohen de pelo menos 0,60. Retornos não podem orientar mudanças
no prompt.

Os pesos do LLM podem ter sido treinados depois de parte dos documentos e ter
memorizado fatos históricos. Impedir preços no prompt reduz, mas não elimina,
esse risco. Por isso a análise histórica será descrita como retrospectiva, e não
como prova de uma decisão que poderia ter sido executada em tempo real. Uma
validação genuinamente prospectiva exige eventos posteriores ao corte de
conhecimento documentado do modelo.

## Efeito fundamental não é retorno anormal

Registrado em 12/08/2026, antes de qualquer backtest da V2 ter sido executado.

O rótulo pede o efeito fundamental esperado da informação sobre o valor das
ações. A estratégia precisa de retorno anormal. As duas coisas só coincidem
quando a informação ainda não está no preço, e nada no desenho garante isso.

O caso clássico é o resultado que vem forte e derruba a ação porque o mercado
esperava mais: o efeito fundamental é positivo e o retorno é negativo. Um
rotulador que leia só o documento não tem como saber a expectativa — a regra 1
do protocolo proíbe consultá-la —, então ele acerta o rótulo e erra o sinal do
trade. É por isso que as regras 4 e 5 empurram para neutro em vez de deixar a
leitura de manchete decidir.

O corpus reduz parte dessa exposição. São 882 Fatos Relevantes e nenhum
demonstrativo de resultado: earnings entram na CVM por outra categoria e ficaram
fora. Mas o mecanismo continua ativo em `Guidance` e `Projeções`, que são
revisão de expectativa por definição, e no tamanho de dividendo ou JCP frente ao
que o mercado projetava.

Isso também explica a neutralidade dominante sem precisar culpar o protocolo:
os assuntos mais frequentes do corpus são remuneração ao acionista, JCP,
recompra e reorganização societária — exatamente as operações que a regra 5
trata como neutras porque realocam valor em vez de criar. A escassez de eventos
direcionais é uma propriedade do que é um Fato Relevante, não um artefato da
régua.

Consequência para a leitura do resultado: um backtest nulo na V2 admite ao menos
três explicações que este desenho não separa — rotulagem ruim, rotulagem boa
medindo a grandeza errada, ou ausência de efeito lead-lag condicionado. O
relatório precisa declarar as três em vez de atribuir o nulo à primeira.

## Emenda: backtest exploratório após reprovação do gate

Escrita em 12/08/2026, antes de qualquer retorno da V2 ter sido calculado e
antes de a orquestração do backtest existir.

O gate reprovou. O painel de referência produziu kappa de Fleiss de 0,7032, que
passa, e macro-F1 de 0,6876 nas classes com suporte, contra limiar de 0,70. A
classe `nao_especifico` ficou com suporte zero no gold, o que também reprova a
exigência de presença das quatro classes. O limiar de 0,70 não é alterado.

A especificação original proíbe rodar P&L antes da aprovação. Esta emenda abre
uma exceção declarada e limitada:

- o backtest a seguir é **exploratório**, não confirmatório;
- qualquer que seja o resultado, ele não constitui evidência de que a estratégia
  funciona, e não pode ser apresentado como tal;
- ele será reportado sempre ao lado da reprovação do gate, nunca isolado;
- nenhuma mudança de prompt, limiar, holding, top N, modelo ou desenho de
  amostra será feita com base no que ele mostrar.

Motivo de rodar mesmo reprovado: o diagnóstico é específico e vale medir. A IA
recuperou 16 de 16 eventos positivos e 18 de 18 negativos do gold, sem nenhuma
inversão de sinal, e errou apenas por excesso — 31 dos 56 documentos neutros do
gold receberam direção. O erro é diluição, não inversão. Saber se essa diluição
zera o alfa ou apenas o reduz informa se uma V3 vale a pena, e um nulo medido é
conclusão mais forte que uma pergunta em aberto.

Hipótese registrada antes da execução, formulada pelo autor: dada a estrutura de
erro por diluição, o alfa esperado é zero ou levemente positivo. Um resultado
fortemente positivo seria surpresa e exigiria suspeita de erro de implementação
antes de qualquer comemoração.

Limite de poder, registrado antes da execução: o corpus tem 76 eventos
direcionais em 882 documentos, ao longo de dez anos e cerca de vinte papéis, e o
painel sugere que perto de metade dessas direções não se sustenta. Com essa
contagem efetiva, um alfa pequeno e verdadeiro provavelmente não é distinguível
de zero pela inferência por blocos. Um resultado nulo deve ser lido como
"não foi possível medir", e não como "não existe efeito".

## Fora do escopo

Não serão adicionados notícias gerais, RAG, embeddings, fine-tuning, classificação
de regimes, outros tamanhos de universo, otimização de holding ou estratégias
específicas por categoria de evento. Essas extensões ficam para depois da V2,
independentemente do resultado.
