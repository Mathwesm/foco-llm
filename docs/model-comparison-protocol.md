# Comparação exploratória de competência inicial

## Decisão antes de observar a execução de 1,5B

Comparar Qwen2.5-0.5B-Instruct e Qwen2.5-1.5B-Instruct nos mesmos 120 exemplos de validação do piloto. O objetivo imediato é selecionar uma configuração com competência limpa suficiente para estudar distratores; não estimar um efeito causal isolado do número de parâmetros.

O modelo de 1,5B usa a revisão `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, fixada antes da execução. A [ficha oficial](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) informa 1,54 bilhão de parâmetros e licença Apache 2.0. A GPU local tinha aproximadamente 5 GiB livres antes do carregamento. A viabilidade será confirmada pela execução, não por uma promessa de memória.

## Controles mantidos

- Dataset e variantes: mesmos IDs, textos, respostas e evidências; seed de dados 42.
- Partição: somente validação; dez problemas-base por tarefa, quatro condições.
- Prompt: `evidence-json-v1`, mesmo texto e template de conversa oficial de cada checkpoint.
- Geração: gulosa, lote 1, seed 42, até 128 tokens novos e 1.024 tokens de entrada; limite cooperativo de 60 segundos por exemplo.
- Execução: mesma RTX 3060 Laptop de 6 GB; FP16, atenção eager, PyTorch 2.8.0+cu128 e Transformers 4.57.3.
- Avaliação: protocolo `content-envelope-v2` para ambos, mantendo também pontuação estrita e seleção de evidências.

O backend de inferência não foi alterado desde a execução de 0,5B. O hash global do código pode diferir porque foram acrescentados módulos de análise. Os checkpoints têm pré/pós-treinamento próprios; mesmo na mesma família, tamanho não é a única diferença possível. A análise adicional foi definida depois do baseline de 0,5B e antes desta execução de 1,5B.

### Emenda após falha numérica, antes da comparação válida

A tentativa de 1,5B em FP16 produziu 120 sequências idênticas de `!`, todas limitadas a 128 tokens. Um diagnóstico no primeiro prompt encontrou NaN em todos os 151.936 logits do próximo token. Essa execução é inválida para comparar competência e será preservada como falha técnica.

Foi acrescentada uma proteção que recusa NaN, infinito positivo e sequências sem nenhum candidato finito, sem substituir valores ou escolher um token arbitrariamente. A precisão passou a ser opção explícita. No diagnóstico de um prompt, BF16 produziu logits finitos e iniciou JSON normalmente. Isso não prova correção de respostas nem estabilidade em todos os prompts.

**Novo controle:** repetir 0,5B e 1,5B em BF16, com a proteção ativa para ambos. Manter o restante dos controles. O baseline antigo de 0,5B em FP16 não será usado como o par principal do modelo de 1,5B em BF16. A alteração é motivada pelo diagnóstico numérico, não pela seleção de uma pontuação melhor.

## Critério operacional de triagem

Usar pelo menos 8 acertos nos 10 problemas limpos de cada tarefa como sinal **exploratório** de competência suficiente para investigar degradação nessa tarefa. Esse limiar não é teste estatístico, não elimina incerteza e não justifica uma conclusão científica com dez exemplos. Resultados de piso ou teto orientam a revisão de dificuldade e diversidade antes do estudo final.

Relatar por tarefa/condição: acerto independente da resposta, seleção exata de evidências, saídas legíveis, identificadores válidos, transições pareadas e recursos medidos. Não mudar prompt ou parsing após observar as saídas desta comparação. Não treinar durante esta etapa.

## Auditoria qualitativa do piloto existente

Foi inspecionado o primeiro problema-base da validação em cada tarefa, com suas quatro variantes, totalizando 12 exemplos. É uma amostra deliberada para revisão de estrutura, não uma auditoria estatisticamente representativa.

| Problema-base | Verificação manual | Observação |
|---|---|---|
| `arithmetic-000008` | 34 + 29 − 15 = 48; distratores não alteram A | `token-8 tokens` introduz um número no nome do objeto e linguagem pouco natural |
| `deduction-000008` | P197 → P541 → C615 para object-8 | Categoria-alvo sempre começa com C, enquanto a categoria distratora começa com D; há pista lexical |
| `tracking-000008` | B614 → B576 → B110 | As frases relevantes têm ordem fixa; a frase “After these events” pode aparecer antes de uma frase de movimento no texto |

Nos casos inspecionados, os gabaritos permanecem coerentes nas quatro condições. Isso não prova que todos os exemplos são adequados. Em especial, mencionar todos os fatos corretos como evidências não demonstra uso desses fatos: o modelo de 0,5B selecionou F1/F2/F3 em exemplos limpos e ainda errou a resposta.

Para a próxima versão dos dados: nomes de objetos sem números em aritmética; distratores com o mesmo vocabulário e prefixos dos fatos-alvo; marcadores temporais explícitos e consistentes; cadeias/dificuldades variadas; separação de templates e estruturas entre splits. Os problemas antigos não serão alterados no lugar nem usados como teste final após inspeção.
