# Protocolo de rotulagem da V2

Versão: `rotulagem-eventos-1.0.0`

O objetivo é classificar o efeito fundamental esperado da informação sobre o
valor das ações da companhia emissora. Não é uma previsão da reação observada
no pregão.

## Regras

1. Leia somente a coluna `texto`. Não abra links, pesquise a empresa ou consulte
   preços, retornos e notícias. Se reconhecer o caso, ignore o que aconteceu
   depois. Não complete trechos cortados.
2. Marque `especifico_empresa=true` quando o texto ligar o fato à emissora, a uma
   controlada, a um ativo, contrato ou processo com efeito declarado sobre ela.
   Use `false` quando o texto for genérico, tratar apenas de terceiros ou não
   permitir estabelecer esse vínculo.
3. Use `direcao=positiva` somente quando o texto sustentar benefício líquido
   claro para o valor econômico das ações. Use `negativa` para prejuízo líquido
   claro.
4. Use `neutra` quando o efeito for incerto, condicional, misto, meramente
   factual ou depender de expectativa e informação externa. Alta ou queda de um
   número não determina o sinal sozinha.
5. A mera aprovação ou o pagamento de dividendos e JCP é neutra. Financiamento,
   aquisição, venda de ativo, recompra ou troca de gestão também são neutros sem
   uma implicação líquida clara no próprio texto.
6. Se `especifico_empresa=false`, a direção deve ser `neutra`. Um evento pode ser
   específico e ainda assim ter direção neutra.

Preencha somente `true` ou `false` em `especifico_empresa` e `positiva`,
`negativa` ou `neutra` em `direcao`.

## Divergências

Cada avaliador trabalha sem ver a resposta do outro ou da IA. Divergências são
enviadas a um terceiro avaliador, que recebe apenas o mesmo texto e preenche um
novo rótulo antes de conhecer as respostas anteriores.

## Interpretação da amostra

A amostra é uma auditoria cega com 30 documentos de cada direção prevista:
positiva, negativa e neutra. Casos marcados como não específicos entram primeiro
na cota neutra. O desenho testa as três decisões direcionais e inclui os filtros
de especificidade disponíveis, mas não estima a prevalência das classes no
corpus. A taxa de sucesso técnico e a taxa de abstenção do corpus completo são
reportadas separadamente.

O macro-F1 usa as classes conjuntas `não específico`, `específico positivo`,
`específico negativo` e `específico neutro`. O gate só pode passar se as quatro
classes aparecerem tanto nas previsões da amostra quanto no gold humano.

## Classes com suporte insuficiente

Uma classe com menos de 10 documentos no gold humano tem F1 dominado por ruído
amostral: um ou dois desacordos movem a métrica de 0 a 1. Por isso o macro-F1
que decide o gate é calculado apenas sobre as classes com suporte igual ou maior
que 10, e exige pelo menos duas delas. O macro-F1 das quatro classes continua
sendo calculado e reportado sempre, junto da lista de classes deixadas de fora e
das métricas individuais de cada uma.

O suporte é medido no **gold humano**, não nas previsões da IA, e isso é
deliberado. Se a IA subdisparar uma classe — prever poucos casos de algo que os
humanos veem com frequência —, o suporte no gold sobe, a classe volta a entrar
no macro-F1 do gate e a falha é penalizada normalmente. A exclusão só acontece
quando os próprios humanos também encontram poucos casos, isto é, quando a
classe genuinamente não tem poder estatístico na amostra.

Regra decidida em 12/08/2026, antes de existir qualquer rótulo humano e sem
nenhum retorno ter sido calculado. O motivo é a distribuição observada do lote
`ia-eventos-1.2.5`: 5 documentos não específicos em 795 classificações válidas
(0,6%). Com 90 documentos na amostra, essa classe carregaria 25% do gate com um
punhado de observações. O limiar de 0,70 e o gate de kappa ≥ 0,60 não mudam.
