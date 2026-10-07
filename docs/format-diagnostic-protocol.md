# Diagnóstico de formato — protocolo de 29/09/2026

Comparar os mesmos 30 problemas limpos v2.1 do baseline `145ba1dd1550430d` em três braços: original já registrado; JSON com instrução explícita de string entre aspas; resposta simples, sem JSON ou evidências. Gerar apenas os 60 novos casos com o modelo original 1,5B, mesma revisão, BF16, seed 42, runtime e limites do baseline. Nenhum treinamento ou teste final nesta etapa.

Não alterar frases, ordem, pergunta ou dificuldade. Não selecionar itens pelos acertos. A nova instrução JSON modifica a redação das instruções, portanto o efeito não isola somente a palavra “inteiro”. O braço simples remove também a exigência de evidências: seu efeito combina contrato e carga de saída.

Preservar o parser `content-envelope-v2` para os braços JSON, inclusive a aceitação diagnóstica de um único bloco completo. No braço simples aceitar somente a resposta inteira após retirar espaços externos, com padrão de inteiro ou código P/B; não extrair números de explicações nem converter JSON. Falhas ficam no denominador. Comparar acurácia e legibilidade por tarefa, com todas as respostas brutas publicadas.

O experimento é exploratório: dez casos por tarefa e um modelo. Não selecionar o melhor braço por item. Se adotarmos outro prompt depois, congelar nova versão e repetir todos os controles; não substituir os baselines anteriores. Seguir para dificuldade/profundidade se o efeito de piso persistir. O protocolo não promete que mais treinamento resolverá o problema.
