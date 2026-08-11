# Diretrizes do relatório final — Desafio Quant AI

Resumo do edital oficial. Serve para orientar o que o pipeline precisa
**produzir** — a redação só começa quando a V1 estiver fechada.

---

## Restrições que eliminam

| Regra | Valor |
|---|---|
| Formato | PDF, **16:9 horizontal** |
| Páginas | **Máximo 5** — 6 ou mais **elimina** |
| Nome do arquivo | `[chave de envio].pdf`, fornecida após o pré-relatório |
| Anonimato | **Obrigatório** — sem nome de participante, equipe, universidade ou logo. Descumprir pode eliminar |
| Língua | Português (termos técnicos em inglês são aceitos) |

Capa, referências e apêndices contam dentro das 5 páginas. Pode usar de 1 a 5.

## Formato esperado

Documento **visual, em formato de apresentação** — não artigo acadêmico nem
texto corrido. A referência informal é **~750 palavras**; acima disso
provavelmente há texto demais. Cada página deve funcionar como unidade completa
de comunicação, legível em tela cheia sem zoom.

Não há apresentação oral. Os avaliadores **não terão acesso a código, links ou
qualquer material externo** — só ao PDF. Links e QR codes são desaconselhados.

Fórmulas, pseudocódigo e trechos de código são permitidos quando ajudam a
explicar; nada disso é obrigatório. A proposta é simular a apresentação de um
projeto para um gestor sênior.

---

## Critérios de avaliação

| Peso | Critério | O que avalia |
|---|---|---|
| **20%** | Conceito da estratégia | Hipótese central, ineficiência explorada, lógica econômica, justificativa do retorno |
| **20%** | Modelagem | Estrutura quantitativa, dados, construção dos sinais, geração das decisões |
| **15%** | Backtest | Metodologia, consistência, coerência com a proposta, **tratamento de vieses e limitações** |
| **15%** | Análise de resultados | Clareza, interpretação crítica, pontos fortes e fracos, cenários |
| **15%** | **Uso de IA generativa** | Como foi usada, valor agregado, **exemplos concretos**, limitações encontradas |
| **10%** | Conclusão e próximos passos | Viabilidade, limitações, melhorias, evolução |
| **5%** | Apresentação do robô | Coerência do nome, identidade visual, integração com a estratégia |

Também pesam: clareza, coerência entre ideia/modelo/resultados, rigor
metodológico, capacidade crítica e objetividade.

> Complexidade **não** é critério. Estratégia simples e bem executada pode
> superar estratégia complexa mal justificada.

---

## Identidade do robô (obrigatório)

Três elementos, em qualquer página, desde que fáceis de identificar:

- **Nome** da estratégia
- **Identidade visual**
- **Explicação do nome escolhido**

Imagem gerada por IA para a identidade visual é permitida e conta como uso de
IA generativa — mas não deve ser o único uso apresentado, dado que o critério
vale 15%.

---

## Implicações para o projeto

Pontos em que o edital e o desenho atual da V1 se encontram:

1. **"Tratamento adequado de vieses e limitações" é critério explícito do
   backtest.** O trabalho de universo *point-in-time*, não-sincronia, viés de
   sobrevivência e FDR é exatamente o que essa linha pontua — precisa estar
   visível, não escondido em apêndice.

2. **Os três desfechos possíveis são todos publicáveis.** "Efeito real mas não
   negociável após custos" pontua em interpretação crítica e capacidade
   crítica. Não há prêmio por retorno alto.

3. **15% para uso de IA generativa exige exemplos concretos.** Vale registrar
   ao longo do desenvolvimento onde a IA foi de fato usada e onde falhou —
   reconstruir isso no fim é pior.

4. **Restrição de espaço é severa.** ~750 palavras em 5 páginas significa que a
   maior parte do que o pipeline produz **não** entra. As figuras precisam ser
   escolhidas para caber nesse orçamento — decidir isso cedo evita gerar
   material que não será usado.

5. **Anonimato afeta o repositório.** Nenhuma figura ou tabela gerada pode
   carregar nome, e-mail ou instituição em rodapé ou metadado.
