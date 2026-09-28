# Dados revisados e teste funcional de treinamento

## Protocolo antes da execução

Esta etapa revisa o piloto e verifica se um ajuste pequeno pode ser executado e retomado na RTX 3060 de 6 GB. Não testa ainda a hipótese científica de melhora de relevância. O artigo permanece reservado.

O gerador `benchmark-v2-candidate` cria 1.200 exemplos, sendo 960 de treino, 120 de validação e 120 de teste, com seed 42. As quatro variantes de cada problema ficam na mesma partição. O gerador antigo e seus resultados permanecem intactos.

Mudanças: recipientes sem números no nome; propriedades relevantes e distratoras com o mesmo prefixo P; cadeias independentes sem símbolos coincidentes; movimentações com tempos explícitos; ordem textual embaralhada; seleção aleatória dos nomes-alvo. O verificador resolve o texto sem consultar o gabarito e testa a necessidade de cada evidência removendo fatos.

Cada partição usa um template linguístico próprio. O treino tem cadeias de duas ou três operações/regras/movimentações, a validação tem quatro e o teste, cinco. Isso separa templates e comprimentos de cadeia, mas **confunde mudança linguística e dificuldade**: não permite atribuir uma diferença somente a um desses fatores. Ainda há uma única família estrutural por tarefa, vocabulário restrito e um template por partição. Não alegar generalização estrutural ampla.

O distrator semelhante contém uma cadeia inteira para outro alvo. Portanto, a condição semelhante também é mais longa que as condições não relacionada/numérica. Um futuro controle de comprimento é necessário antes de atribuir toda diferença à semelhança semântica. Os baselines da v1 não devem ser comparados diretamente com modelos treinados na v2 como se fossem os mesmos dados.

## Configuração do teste curto

- Qwen2.5-1.5B-Instruct na revisão já usada no baseline; pesos-base BF16, sem quantização.
- LoRA rank 4, alpha 8, dropout 0, somente `q_proj` e `v_proj`; pesos-base congelados.
- AdamW, taxa 0,0001, weight decay 0; seis atualizações, lote 1, sem acumulação.
- Seed 42, atenção eager, gradient checkpointing não reentrante, cache desligado, até 512 tokens sem truncar entradas.
- Seis exemplos exclusivamente de treino: um par limpo/semelhante do primeiro problema de cada tarefa. A ordem é aritmética, dedução, acompanhamento; não é uma amostra representativa.
- Supervisão de resposta e identificadores de evidência em JSON. O prefixo completo, inclusive o cabeçalho de assistente, recebe máscara -100; a conclusão e o encerramento do assistente recebem perda. A igualdade do prefixo tokenizado é conferida antes de mascarar.
- Pausa programada depois da terceira atualização, saída do processo e retomada em outro processo. Cada passo publica adaptador, otimizador, geradores aleatórios e histórico por renomeação atômica; hashes são conferidos antes da leitura.

A serialização é um teste do braço com resposta e evidências. Os controles C1/C2, ponderação da perda e orçamento correspondente ainda precisam ser implementados antes da comparação científica. Nenhum rótulo de evidência entra no prompt de inferência.

Critérios de sucesso: perdas/gradientes/pesos finitos, somente adaptadores treináveis, hash dos pesos-base idêntico antes/depois, hash dos adaptadores diferente, seis checkpoints íntegros e recarga em um modelo-base novo com logits equivalentes em uma posição de um exemplo de treino (tolerância absoluta 0,0001). Essa verificação é funcional, não uma auditoria de equivalência em todas as entradas.

O histórico registra a perda de cada exemplo diferente; não deve ser interpretado como curva de validação. A sonda antes/depois usa o mesmo exemplo de treino e mede apenas ajuste sobre esse exemplo. Memória e duração dos passos serão medidas. Não haverá execução paga nem agendamento. Alertas remotos e limites de duração para treinamentos longos continuam pendentes.

## Reprodução

```powershell
poetry install --with inference
poetry run python scripts/prepare_v2.py
# Use o caminho de dataset.json informado pelo gerador.
poetry run python -m foco_llm.train_smoke $datasetPath --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --stop-after 3 --output data/training/2026-09-28
poetry run python -m foco_llm.train_smoke $datasetPath --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --output data/training/2026-09-28
```

Manter o mesmo código, configuração, dados, runtime e `--output` para retomar. Não executar duas instâncias no mesmo destino. Checkpoints e pesos ficam em `data/`, fora do Git. O relatório público terá configuração, métricas, gráfico e limitações. O teste final não será usado para inferência ou treinamento.

Referências da implementação: [PEFT 0.17.1 — LoRA](https://huggingface.co/docs/peft/v0.17.1/en/package_reference/lora) e [salvamento dos adaptadores](https://huggingface.co/docs/peft/v0.17.1/en/package_reference/peft_model). LoRA é usada para limitar os parâmetros atualizados; sua presença não garante melhora na tarefa.
