# Calibração da validação numérica — yfinance × COTAHIST

**Versão: 1.0.0 — CONGELADA em 2026-07-30**

Este documento registra como os limiares foram escolhidos. **Não recalibrar
olhando os resultados dos 41 candidatos** — isso transformaria a validação em
ajuste retrospectivo, onde os cortes passam a descrever os dados em vez de
testá-los.

Alterar qualquer limiar exige subir `CALIBRACAO_VERSAO` em
[`src/qualidade_dados.py`](../src/qualidade_dados.py), o que marca
`needs_revalidation=sim` em toda linha já revisada.

---

## O que a validação testa

O yfinance emenda tickers renomeados por conta própria: `COGN3.SA` devolve
histórico desde 2014, embora `COGN3` só exista no COTAHIST desde out/2019. A
emenda é conveniente e **opaca**.

Como o COTAHIST é o registro oficial da bolsa, dá para auditar: no período em
que a empresa negociava sob o código antigo, o preço que o Yahoo reporta sob o
código novo deveria reproduzir o registro da B3.

**Dois testes separados:**

| Teste | O que compara | Por quê |
|---|---|---|
| Preço bruto | `Close` não ajustado × `PREULT` | Razão **constante** ≠ 1 indica reajuste retroativo (split, bonificação); razão **errática** indica séries sem relação. O que importa é a **dispersão**, não a distância de 1 |
| Retornos | Variação percentual das duas séries | É o que o projeto usa, e fator constante some nele |

O teste de retornos exclui: datas de transição (3 pregões em cada borda), datas
de evento corporativo (provento ou desdobramento, e o dia seguinte) e dias sem
negociação no COTAHIST. Nesses pontos a divergência é esperada e não diz nada
sobre identidade da série.

---

## Controles

Classificados por **conhecimento societário público, antes de olhar os
números**. O contrário tornaria a calibração circular.

### Positivos — 11 renomeações

`ESTC3→YDUQ3`, `EMBR3→EMBJ3`, `CCRO3→MOTV3`, `TIMP3→TIMS3`, `ARZZ3→AZZA3`,
`KROT3→COGN3`, `BRDT3→VBBR3`, `DTEX3→DXCO3`, `BVMF3→B3SA3`, `ELET3→AXIA3`,
`BTOW3→AMER3`

### Negativos — 3 eventos societariamente distintos

| Par | Natureza |
|---|---|
| `BRFS3→MBRF3` | Fusão — `MBRF3` carrega o histórico da Marfrig, não o da BRF |
| `SUZB5→SUZB3` | Conversão de classe PNA → ON |
| `RUMO3→RAIL3` | Reestruturação com troca de ações |

---

## Distribuição observada

| Métrica | Positivos (n=11) | Negativos (n=3) |
|---|---|---|
| `erro_retorno_mediano` | ≤ 8e-08 | 3e-08 · 0,0148 · 0,0160 |
| `corr_retornos` | ≥ 0,9827 | 0,4422 · 0,9967 · indefinida |
| `erro_retorno_p95` | ≤ 0,0094 | 0,0031 · 0,0529 · 0,0593 |
| `dispersao_razao_precos` | ≤ 0,0151 | 0,1966 · 0,6532 · 3,3908 |

---

## Limiares adotados

```python
LIMIAR_ERRO_RETORNO_MEDIANO = 1e-05
LIMIAR_ERRO_RETORNO_P95     = 0.02
LIMIAR_CORR_RETORNOS        = 0.95
LIMIAR_DISPERSAO_DEGRAU     = 0.05
```

**Justificativa dos cortes:**

- **`1e-05` para o erro mediano.** É o discriminante mais forte: positivos
  ficam em ~1e-08 e os negativos divergentes em ~1,5e-02 — **cinco ordens de
  grandeza** de separação. O corte fica folgadamente no meio do vão, três
  ordens acima do pior positivo e três abaixo do melhor negativo.
- **`0,02` para o p95.** Acima do pior positivo (0,0094) e abaixo do melhor
  negativo divergente (0,0529).
- **`0,95` para a correlação.** Abaixo do pior positivo (0,9827), acima do
  `BRFS3→MBRF3` (0,4422). Métrica auxiliar: sozinha ela não separa, porque
  `RUMO3→RAIL3` tem correlação 0,9967 sendo negativo.
- **`0,05` para a dispersão.** ~3× a dispersão máxima observada nos positivos
  (0,0151). Não reprova: **separa em duas categorias**.

---

## Categorias

| Categoria | Significado |
|---|---|
| `retornos_consistentes_sem_quebra_escala` | Retornos reproduzem o COTAHIST com escala estável |
| `retornos_consistentes_com_quebra_escala` | Retornos batem, mas a razão de preços é instável — houve evento societário no período |
| `retornos_divergentes` | Os retornos não batem |
| `inconclusivo` | Amostra insuficiente, ou o Yahoo não emendou |
| `serie_ausente` | O Yahoo não tem série para o ticker |

**A palavra "aprovado" foi deliberadamente evitada.** Ela carrega autorização
mesmo quando o texto diz que não carrega.

---

## O que esta validação NÃO faz

Consistência numérica **não é** identidade econômica.

`RUMO3 → RAIL3` demonstra: erro mediano `3e-08`, correlação `0,9967` — os
retornos batem perfeitamente. Societariamente foi reestruturação com troca de
ações, não renomeação. Foi capturado apenas pela dispersão (3,39).

Por isso a validação **nunca preenche** `tipo_evento`,
`canonical_instrument_id`, `continuidade_autorizada`,
`tratar_retorno_fronteira` nem o `status` documental. Ela produz **evidência e
prioridade**; a autorização depende da categoria societária confirmada em
documento.

---

## Validação por cadeia

O Yahoo não ter mais o código **intermediário** não significa que o histórico
sumiu — ele costuma migrar tudo para o código terminal. Quando a validação
direta resulta em `serie_ausente` ou `inconclusivo`, o pipeline tenta o
terminal da cadeia.

Resolveu 2 casos: `VVAR3→VIIA3` validado via `BHIA3` (1.451 datas comuns), e
`PARC3→WIZS3` via `WIZC3`.

A busca **para em bifurcação**: `ESTC3` aparece indo para `YDUQ3` e para
`ALSO3`, então não há cadeia única a seguir, e adivinhar seria pior que não
responder.

**Limitação conhecida:** quando a cadeia inteira sai do Yahoo, não há o que
consultar. É o caso de `GETI4 → TIET11 → AESB3`, em que nem o terminal existe.
