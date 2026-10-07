# Ciclo funcional treino → validação — protocolo

## Objetivo

Verificar a avaliação de um adaptador recarregado sobre os mesmos 120 casos v2.1 do baseline `145ba1dd1550430d`. A competência inicial baixa permanece uma limitação. Esta execução não substitui a matriz C1/C2/C3 nem é teste de transferência.

## Configuração antes da execução

Usar dados `b142fe60ba9cd6a5`, Qwen2.5-1.5B-Instruct na revisão `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16, LoRA rank 4 em q_proj/v_proj, alpha 8, dropout zero, AdamW a 0,0001, seed 42. Executar seis atualizações, uma por exemplo: primeiro problema-base de treino de cada tarefa, nas condições limpa e similar. Não escolher exemplos pela acurácia de validação.

Esse orçamento pequeno mantém o objetivo funcional: integrar treino, recarga, proveniência do adaptador e avaliação. Uma melhora ocasional não valida a hipótese científica, e ausência de melhora não demonstra que LoRA seja ineficaz.

Manter na inferência o prompt `evidence-json-v1`, geração gulosa, seed 42, 128 tokens novos, limite cooperativo de 60 segundos, eager e BF16. Avaliar somente validação, preservando teste final. O carregador verifica integridade dos arquivos, modelo-base, revisão e hash dos dados; registra hashes do adaptador e do manifesto no novo resultado para impedir mistura de checkpoints.

## Critérios e registro

Relatar formato, respostas, evidências e transições antes/depois por exemplo, sem selecionar apenas saídas válidas. Preservar respostas brutas e falhas. Registrar perda de treino separadamente de acurácia de validação. Não ajustar prompt, parsing ou orçamento após ver o resultado desta execução.

Critério funcional: concluir treinamento com base congelada, adaptador atualizado e recarga verificada; concluir 120 tentativas de validação com identidade do adaptador registrada. O critério científico de competência limpa de 8/10 permanece inalterado e exploratório.

Próximas etapas científicas continuam sendo diagnóstico de dificuldade, controles de supervisão, repetições e avaliação independente. Antes de execuções longas ou pagas, dimensionar duração, memória, alerta e orçamento. Nenhum recurso de nuvem é necessário para esta execução curta.
