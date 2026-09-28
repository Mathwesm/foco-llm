# Guia dos experimentos e dos 120 casos

## O que estamos investigando

Queremos medir se um modelo consegue responder usando as informações necessárias e manter o desempenho quando acrescentamos informações irrelevantes. Depois, comparar o modelo original com versões ajustadas e investigar se o aprendizado transfere entre tarefas. Não medimos diretamente pensamentos ou mecanismos internos: respostas e evidências declaradas são medidas de comportamento.

Os enunciados são **sintéticos e em inglês**. Não são provas de português, questões escolares oficiais nem um benchmark externo baixado da internet. Nosso código gera problemas de gramática controlada, calcula o gabarito e valida as relações. Isso permite controlar distrações, mas limita a conclusão a esse universo de problemas.

## De onde vêm os 120 casos

O conjunto completo tem 300 problemas-base: 100 por tarefa. Cada problema possui quatro versões, totalizando 1.200 exemplos. A separação mantém as quatro versões juntas, evitando que a mesma situação apareça no treino e na validação.

| Partição | Problemas-base | Versões por problema | Exemplos | Finalidade |
|---|---:|---:|---:|---|
| Treino | 240 | 4 | 960 | Atualizar adaptadores; não é medida independente de sucesso |
| Validação | 30 | 4 | 120 | Diagnosticar e escolher configurações |
| Teste reservado | 30 | 4 | 120 | Avaliar a configuração final congelada |

Cada avaliação de validação contém **10 problemas-base por tarefa × 3 tarefas × 4 condições = 120 respostas**. São 30 situações independentes na construção, não 120 problemas independentes: as variantes compartilham gabarito e conteúdo relevante. Intervalos futuros devem reamostrar problemas-base, mantendo as variantes juntas.

Os 97 testes automatizados registrados na etapa anterior são outra coisa: verificam o funcionamento do código, como integridade de checkpoints, máscaras de treinamento e métricas. Não são perguntas respondidas pela LLM e não entram em sua acurácia.

## Quais tarefas o modelo resolve

Exemplos abaixo são traduções didáticas simplificadas, não novos dados avaliados.

| Tarefa | Exemplo de problema | Resposta esperada | Capacidade exigida |
|---|---|---|---|
| Aritmética | Começar com 70 bolinhas prateadas, retirar 2, 6, 8 e 7 | 47 | Escolher operações do alvo e calcular |
| Dedução | copper tem P4939; P4939 implica P5909; a cadeia continua até P2134 | P2134 | Encadear regras referentes ao alvo |
| Acompanhamento | O objeto ivory sai de B3037 e passa por locais em tempos 1 a 4, terminando em B7771 | B7771 | Resolver a sequência temporal e identificar a posição final |

A tarefa exige compreensão do enunciado, mas não é somente compreensão textual. A aritmética exige cálculo; dedução exige aplicação de regras; acompanhamento exige atualização de estado. Uma falha pode ocorrer em qualquer uma dessas partes.

Na v2.1, treino usa cadeias de 2 ou 3 passos, validação de 4 e teste de 5. Os templates também mudam entre partições. Portanto, profundidade e redação variam simultaneamente; ainda não isolamos seus efeitos.

## As quatro versões de cada problema

| Condição | O que acrescentamos | Exemplo didático |
|---|---|---|
| Limpa (`clean`) | Nada irrelevante | Apenas os fatos necessários para responder |
| Não relacionada (`unrelated`) | Frases sem relação com a tarefa | A mochila é verde |
| Numérica (`numeric`) | Números sem relação com o cálculo ou alvo | Outro prédio tem 27 janelas |
| Semelhante (`similar`) | Uma cadeia independente do mesmo tipo, para outro alvo | Operações com bolinhas de outra cor |

O gabarito permanece igual nas quatro versões. Na v2.1, preservamos a ordem relativa dos fatos relevantes e igualamos o número de frases distratoras entre as três condições com ruído. Os tokens ainda diferem: frases semelhantes costumam ser maiores. Por isso não atribuímos automaticamente toda diferença ao significado do distrator.

## O que pedimos e como pontuamos

O prompt pede uma resposta em JSON, com `answer` como string e `evidence` como lista dos IDs dos fatos usados. O modelo vê fatos e pergunta, nunca o gabarito. Cada execução gera uma resposta por exemplo, com geração gulosa e limite fixado.

- **Formato estrito:** aceita somente JSON conforme o contrato. Bloco Markdown não passa.
- **Conteúdo:** protocolo fixo aceita JSON puro ou um único bloco Markdown completo, sem reparar valores ou extrair uma resposta de prosa adicional.
- **Resposta:** comparação com o gabarito. Ausências e rejeições ficam no denominador total.
- **Evidências:** IDs devem existir; medimos precisão, revocação e conjunto exatamente correto. Resposta correta e evidências corretas são critérios distintos.
- **Resistência a distrações:** contamos acerto→erro e erro→acerto entre versões do mesmo problema. Se o modelo erra tudo na condição limpa, a ausência de queda não indica robustez.
- **Recursos:** tokens, duração de geração, memória alocada/reservada e motivo de parada. Não são medidas de qualidade científica por si só.

## Histórico preservado

| Etapa | Dados e modelo | Resultado observado | Interpretação |
|---|---|---|---|
| Primeiro baseline | Piloto v1, Qwen2.5-0.5B, FP16 | 37/120 respostas corretas na análise de conteúdo | Referência inicial; saída em Markdown |
| Tentativa numérica | Piloto v1, Qwen2.5-1.5B, FP16 | Logits inválidos; execução preservada | Falha técnica, não desempenho válido do modelo |
| Comparação de tamanhos | Piloto v1, BF16 | 0,5B: 38/120; 1,5B: 74/120 | Comparação restrita ao piloto e às configurações registradas |
| Treino funcional | v2, 1,5B, LoRA, seis passos | Retomada e recarga verificadas; perda de uma sonda de treino caiu | Infraestrutura funciona; não prova melhora em dados não usados |
| Baseline revisado | v2.1, 1,5B, BF16, sem adaptador | 6/120; limpo: aritmética 0/10, dedução 3/10, acompanhamento 0/10 | Forte efeito de piso; exige diagnóstico de competência |

Os valores de versões diferentes não medem ganho ou perda causado pelo treinamento. O conjunto mudou. Todos os números acima têm artefatos e limitações nos relatórios vinculados abaixo.

## Quando teremos material suficiente para o artigo

O critério não é atingir uma acurácia desejada. Precisamos de pergunta delimitada, revisão bibliográfica, dados auditáveis, protocolo documentado, controles antes/depois comparáveis, repetições e incerteza, teste reservado e discussão das limitações. Resultados negativos também são válidos quando o desenho permite interpretar o que falhou.

Para afirmar transferência entre tarefas, é necessário treinar em tarefas-fonte e medir tarefas-alvo ausentes desse ajuste. Para afirmar efeitos de tamanho, comparar tamanhos da mesma família com protocolos equivalentes. Para afirmar mecanismos internos, precisaríamos de intervenções adicionais; nossos IDs de evidência não bastam. O título e a conclusão devem corresponder ao que efetivamente medirmos, com revisão do orientador.

A apresentação será preparada depois. Este guia e os relatórios são a base documental para explicar objetivo, tarefas, exemplos, método, resultados e limitações sem reconstruir o histórico de memória.

## Modelos maiores e nuvem

Já usamos a família Qwen, de origem chinesa. Não escolhemos modelos pelo país, e sim por licença, disponibilidade de pesos, tamanho, competência e viabilidade. Pesos gratuitos não tornam a GPU gratuita. Nenhuma instância foi contratada.

A ampliação usará primeiro uma única GPU, se suficiente: revisão fixada, mesmo conjunto, execução retomável, limites de tokens, amostra curta para medir memória e tempo, e orçamento máximo antes do lote completo. Repetir primeiro uma configuração local separa a mudança de hardware da mudança de modelo. Quantização, batching e outras otimizações devem ter seus efeitos medidos, sem sacrificar comparabilidade silenciosamente.

## Fontes internas verificáveis

- [Primeiro baseline](baseline-2026-09-27.md) e [análise de conteúdo](content-evaluation-2026-09-27.md).
- [Falha numérica](numerical-failure-2026-09-27.md) e [comparação de tamanhos](model-comparison-2026-09-27.md).
- [Primeiro treinamento funcional](training-smoke-2026-09-28.md).
- [Protocolo v2.1](benchmark-v21-protocol.md) e [resultados v2.1](baseline-v21-2026-09-28.md).
- [Controles, repetições e nuvem](training-and-cloud-plan.md).
