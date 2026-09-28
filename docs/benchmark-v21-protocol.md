# Revisão v2.1 e baseline — protocolo antes da medição

## Objetivo desta etapa

Medir Qwen2.5-1.5B-Instruct sem adaptador nos 120 exemplos de validação da revisão v2.1. Esse resultado será a referência para futuros ajustes avaliados **nos mesmos dados**. O teste LoRA anterior verificou infraestrutura na v2; sua perda de treino não é uma medida de eficácia na v2.1.

## Problemas identificados e correções

A auditoria automática da v2 encontrou 823 das 900 variantes com distratores em que a ordem relativa dos fatos relevantes diferia da versão limpa. Nesse desenho, uma diferença de acurácia poderia refletir tanto os distratores quanto a reorganização dos fatos. A v2.1 preserva a ordem relativa em todas as variantes e mantém as mesmas posições de evidência entre as três condições com distratores.

As condições não relacionada e numérica tinham apenas uma frase distratora, enquanto a semelhante tinha uma cadeia inteira. Agora todas têm a mesma quantidade de frases distratoras, igual à quantidade de fatos relevantes. O número exato de tokens ainda varia e será registrado; igualar frases não isola completamente o efeito semântico.

Foi corrigida a concordância de “1 marble is”. Atualizações aritméticas ganharam identificadores distintos para que duas operações com a mesma quantidade não pareçam uma frase duplicada. As letras são sorteadas tanto no ramo relevante quanto no distrator, sem uma sequência alfabética exclusiva do ramo-alvo. Nenhum número foi acrescentado ao nome dos objetos ou ao prefixo da atualização.

Os geradores v1/v2 e as saídas anteriores permanecem preservados. A v2.1 mantém IDs, gabaritos e partições da v2, mas modifica textos, posições e quantidade de ruído. A identidade científica é o hash do dataset, não apenas o ID do problema.

## Dados congelados para este baseline

- Dataset: `data/benchmark-v21/2026-09-28/b142fe60ba9cd6a5/dataset.json`.
- Seed 42; 300 problemas-base; 1.200 exemplos: 960 treino, 120 validação e 120 teste.
- Quatro condições: limpa, não relacionada, numérica e semelhante.
- Dez problemas-base por tarefa na validação, com quatro variantes dependentes por problema.
- Mesmo esquema de templates/profundidades da v2: treino 2/3, validação 4 e teste 5 passos. A mudança de template e dificuldade entre partições permanece combinada.
- Verificações: gabarito independente, remoção de fatos para conferir evidências necessárias, variantes agrupadas, preservação de respostas, concordância, ordem relativa, posições pareadas e quantidade de distratores.

A geração automatizada valida os dados de todas as partições. A auditoria textual e a inferência desta etapa usam somente treino/validação; o teste final não será usado para ajustar escolhas experimentais.

## Auditoria semântica da amostra

A amostra preservada contém 24 exemplos: primeiro problema-base de treino e de validação em cada tarefa, com quatro condições. Foram conferidos os alvos, as operações/cadeias e a independência dos ramos distratores. A checagem de IDs, posições e gabaritos em todas as variantes é automatizada. A seleção da amostra é deliberada, não aleatória ou representativa de todo o conjunto.

| Problema-base | Partição | Resolução conferida |
|---|---|---|
| aritmética 000000 | treino | azure: 58 − 6 − 9 + 1 = 44 |
| aritmética 000008 | validação | silver: 70 − 2 − 6 − 8 − 7 = 47 |
| dedução 000000 | treino | ivory: P5366 → P5753 → P8100 |
| dedução 000008 | validação | copper: P4939 → P5909 → P6156 → P1746 → P2134 |
| acompanhamento 000000 | treino | ivory: B7981 → B6653 → B5623 |
| acompanhamento 000008 | validação | ivory: B3037 → B7823 → B4576 → B7290 → B7771 |

Na aritmética de validação, o distrator amber contém duas remoções de uma unidade: são operações distintas e recebem identificadores distintos. Em acompanhamento, a cronologia é definida pelos tempos, mesmo quando a frase do último movimento aparece no início do texto. As cadeias de outros alvos não compartilham símbolos com a cadeia relevante.

[Manifesto e amostra auditada](../reports/2026-09-28/benchmark-v21/).

## Configuração fixada

Qwen2.5-1.5B-Instruct, revisão `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, sem LoRA, BF16, RTX 3060 Laptop de 6 GB, atenção eager, seed 42, lote 1, geração gulosa, máximo de 128 tokens novos, limite de entrada 1.024 e limite cooperativo de 60 segundos por exemplo. Prompt `evidence-json-v1`, avaliação estrita preservada e análise `content-envelope-v2` já definida.

Relatar acurácia de resposta, formato, evidências, transições limpo→distrator, tokens de entrada/saída, tempo e memória. Manter rejeições no denominador. A referência exploratória de competência limpa continua sendo pelo menos 8/10 acertos por tarefa; esse limiar não é teste estatístico. Não mudar prompt, parsing ou dados durante a execução por causa de uma pontuação baixa.

```powershell
poetry run python scripts/prepare_v21.py
poetry run python -m foco_llm.inference data/benchmark-v21/2026-09-28/b142fe60ba9cd6a5/dataset.json --model-id Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --precision bfloat16 --output data/inference-v21/2026-09-28
```

## Limitações restantes

Ainda são problemas sintéticos curtos, com uma família estrutural por tarefa, um template por partição e vocabulário pequeno. Frases “distant/another” tornam parte do ruído fácil de reconhecer. Os comprimentos em tokens não são idênticos. O desenho entre partições combina mudança de linguagem e profundidade; não permite separar esses dois fatores. A amostra de validação é pequena e as variantes são dependentes. O estudo completo ainda requer controles de supervisão, repetição de sementes, intervalos de confiança e avaliação de transferência. Não comparar acurácias v1/v2.1 como ganho do modelo.
