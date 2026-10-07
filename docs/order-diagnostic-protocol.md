# Diagnóstico de ordem — protocolo anterior à execução

Usar os 30 casos limpos da validação v2.1 já avaliados em `145ba1dd1550430d`. Inverter somente a sequência de apresentação dos fatos, preservando IDs, textos, pergunta e resposta. Não renumerar fatos. A inversão modifica também a ordem visual dos IDs; não permite separar efeitos do texto e desses rótulos. Não escolhemos exemplos pelo resultado anterior.

Gerar 30 respostas novas com o modelo original Qwen2.5-1.5B-Instruct, mesma revisão, BF16, seed, prompt e limites do baseline. Reutilizar as respostas originais verificando seus hashes de prompt. Não há treinamento, distratores, alteração de dificuldade ou nova tentativa de corrigir formato nesta etapa. O teste final permanece reservado.

O gabarito é invariável à apresentação: somas e subtrações se aplicam ao saldo inicial; dedução segue relações entre propriedades; acompanhamento segue tempos explícitos. A validação independente deve confirmar as respostas antes de gerar.

Pontuar com `content-envelope-v2`, mantendo rejeições no denominador. Publicar cada par, acertos por tarefa, ganhos e perdas. Não escolher a melhor ordem para cada item. Com dez casos por tarefa, o resultado é descritivo e exploratório. Ausência de diferença não prova invariância a outras permutações.

Este desenho mede a reversão exata da ordem existente, não compara necessariamente ordem cronológica com aleatória. Templates, profundidade e contrato de resposta ficam fixos; diagnósticos desses fatores exigirão etapas próprias. A nova execução tem outro momento de medição, portanto duração não é uma comparação controlada de velocidade.
