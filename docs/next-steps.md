# Próximas etapas e critérios de conclusão

**Atualização — 30/09/2026, controles concluídos:** [nove treinos e resultados](local-controls-2026-09-30.md).
C2 atingiu média 34,3/120, C1 31,7/120, C3 19,0/120. Antes do Kaggle,
auditar uma amostra estratificada das respostas, revisar os artefatos,
congelar a avaliação final e gerar a matriz fonte/alvo. Não ajustar
hiperparâmetros retroativamente para selecionar o melhor braço.

**Atualização — 30/09/2026:** [piloto C3 de dedução](source-task-pilot-2026-09-30.md)
concluído, 64 passos/32 exemplos/seed 42; validação 6/120 → 19/120. Próxima
etapa: implementar a supervisão de resposta para C1/C2, congelar a seleção dos
mesmos problemas-base entre braços e sementes, executar repetições e estimar
incerteza agrupando as variantes por problema-base. A fase local não terminou.

**Atualização — 29/09/2026:** diagnósticos de [formato](format-diagnostic-2026-09-29.md)
e [profundidade](depth-diagnostic-2026-09-29.md) concluídos. O segundo teve 120
inferências novas e mostrou dificuldade crescente mesmo sem distratores.
Próximo marco: executor científico configurável e piloto de treinamento,
mantendo controles, avaliação nas tarefas-alvo e teste final reservado.
Os estados abaixo preservam a sequência histórica.

**Diagnóstico de ordem concluído:** 30 novas gerações sem treino; dedução mudou de 3/10 para 2/10, com cinco pares alterados; outras tarefas permaneceram em zero. [Resultado](order-diagnostic-2026-09-28.md). Próxima etapa: contraste de contrato de resposta com problemas fixos; depois profundidade com linguagem controlada. Não escolher a melhor ordem por item.

**Prioridade atual — 2026-09-28:** o baseline v2.1 terminou com 6/120 acertos e nenhuma tarefa atingiu 8/10 na condição limpa. Antes de treinamento de eficácia, definir diagnóstico que separe profundidade, ordem e contrato de resposta. Não modificar o protocolo retrospectivamente nem selecionar apenas exemplos acertados. [Resultado e limitações](baseline-v21-2026-09-28.md). Os registros abaixo preservam a sequência histórica.

## 1. Baseline local — concluído em 2026-09-27

[Resultado registrado](baseline-2026-09-27.md): 120 respostas geradas, mas 0 aceitas pelo contrato estrito, pois todas vieram em blocos Markdown. O próximo passo é separar avaliação de formato e de conteúdo antes de qualquer treinamento. Este resultado fica preservado.

**Atualização:** a separação foi implementada em `content-envelope-v2`, com 37/120 acertos de resposta e 102/120 saídas com identificadores válidos. [Diagnóstico completo](content-evaluation-2026-09-27.md). A prioridade passa a ser competência limpa e revisão do benchmark, antes de treinar.

**Comparação concluída:** repetidos ambos em BF16, 0,5B teve 38/120 acertos e 1,5B, 74/120. O modelo de 1,5B atingiu o limiar exploratório de competência limpa em dedução e acompanhamento, mas nenhum em aritmética. [Relatório completo](model-comparison-2026-09-27.md). Priorizar revisão dos dados e depois teste funcional com 1,5B; ainda falta medir memória de treinamento. Os [controles e repetições](training-and-cloud-plan.md) estão definidos separadamente.

Executar o Qwen2.5-0.5B-Instruct sem ajuste nos 120 exemplos de validação do piloto. A revisão é `7ae557604adf67be50417f59c2c2f167def9a775`; o modelo possui aproximadamente 0,49 bilhão de parâmetros e licença Apache 2.0, conforme sua [ficha oficial](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

O primeiro baseline usou FP16 na RTX 3060 de 6 GB, lote unitário, geração gulosa e limite de 128 tokens novos. Para a comparação atual entre tamanhos, usar explicitamente `--precision bfloat16` em ambos: a tentativa de 1,5B em FP16 apresentou [falha numérica](numerical-failure-2026-09-27.md). Registrar respostas brutas, erros de formato, tokens, duração de geração e pico de memória alocada/reservada pelo PyTorch. Esses picos não representam todo o uso de memória da GPU por outros programas. Downloads, carregamento, tokenização e gravação ficam fora da duração de geração.

**Conclusão:** execução real preservada, métricas por tarefa/condição, gráfico e limitações documentadas. O teste final permanece reservado. Esta etapa mede um modelo já ajustado para instruções, sem ajuste adicional neste projeto.

## 2. Revisar o benchmark

**Atualização em 2026-09-28:** gerada versão candidata v2 com 1.200 exemplos, fatos embaralhados, distratores sem prefixo exclusivo e tempos explícitos. Verificação automática e auditoria manual de três exemplos de treino concluídas. Faltam auditoria ampliada, ajuste de concordância, controle de comprimento e baseline da versão revisada. [Protocolo e limitações](benchmark-v2-and-smoke.md).

Inspecionar os erros de validação; diversificar enunciados, profundidade do raciocínio, posição e quantidade dos distratores. Evitar pistas triviais que denunciem quais frases ignorar. Separar templates e estruturas entre treino e avaliação, além de manter variantes do mesmo problema juntas. Revisar manualmente uma amostra estratificada e registrar defeitos encontrados.

**Conclusão:** versão nova e imutável dos dados, verificadores independentes, protocolo congelado e ausência de vazamento entre splits. Resultados do piloto atual não são misturados com os do benchmark novo.

## 3. Ajuste piloto

**Teste funcional concluído em 2026-09-28:** seis passos LoRA no modelo de 1,5B, com retomada exata em relação ao controle contínuo, pesos-base intactos e recarga verificada. Pico alocado de 3,41 GiB. [Resultado completo](training-smoke-2026-09-28.md). Ainda não é o experimento de eficácia ou uma medição com o comprimento máximo configurado.

Adicionar adaptadores LoRA em um modelo pequeno e medir memória antes de aumentar a escala. Se necessário, avaliar quantização em um ambiente compatível. Começar com poucos passos somente no treino; salvar adaptador, configuração, checkpoints, curvas de perda e versões. Alertas de falha via Telegram serão configurados antes de execuções longas, com credenciais exclusivamente por ambiente.

**Conclusão:** treino retomável e um teste funcional de recarga do adaptador. Uma queda de loss não prova melhora de raciocínio; a comparação usa as métricas externas.

## 4. Comparações controladas

Comparar modelo original, instrução sem ajuste, ajuste com dados limpos, ajuste com ruído/resposta e ajuste com ruído/resposta/evidências. Registrar orçamento de exemplos, tokens e passos para não atribuir à seleção de evidências uma melhora causada apenas por maior treinamento. Fixar as configurações usando validação.

**Conclusão:** comparações com orçamento explícito, ao menos três sementes de treinamento e intervalos de confiança por reamostragem de problemas-base, mantendo suas variantes juntas.

## 5. Transferência e tamanho

Treinar em uma tarefa-fonte e avaliar nas tarefas-alvo não usadas no ajuste. Repetir as direções previstas no protocolo. Depois comparar pelo menos dois tamanhos da mesma família, com controles de precisão, contexto, geração e competência inicial. Tamanho não é uma intervenção isolada: pré-treinamento e capacidade também podem variar.

**Conclusão:** matrizes fonte/alvo, acurácia limpa e com distratores, seleção de evidências, custo e incerteza. Não usar racionalizações textuais como prova de mecanismo interno; análise mecanística exige intervenções adicionais.

## 6. Avaliação final e artigo

Congelar modelos/configurações antes de abrir o teste final. Exportar tabelas e figuras a partir dos artefatos registrados, revisar limitações e então redigir o artigo em inglês. A contribuição e o título serão compatíveis com o que foi realmente demonstrado.

## Execução e retomada

Esta etapa roda sob demanda, sem serviço pago ou agendamento. Instalar a camada de inferência com `poetry install --with inference`, usando Python 3.12 em Windows/Linux e o índice CUDA oficial. O CI padrão não instala os pacotes pesados; testa o controle de execução com um backend simulado e arquivos temporários. O baseline local verifica o backend real.

O executor aceita `python -m foco_llm.inference` dentro do Poetry. O padrão é validação. A revisão deve ser um SHA completo, nunca `main`. Cada resposta é gravada atomicamente; ao repetir o comando, dados, configuração, código e runtime devem coincidir para reutilização. Para retomar em outro dia, indicar o mesmo `--output` da execução original. Não executar duas instâncias simultâneas para o mesmo destino.

JSON inválido, evidências repetidas ou identificadores inexistentes são preservados e contados como respostas ausentes na avaliação. Não há extração oportunista de respostas de textos explicativos. Falhas de infraestrutura interrompem com erro; checkpoints íntegros permanecem disponíveis. O limite de tempo de geração é cooperativo, verificado entre passos, e não encerra à força um driver travado.

Referências técnicas: [Transformers — geração](https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/text_generation), [PyTorch — binários CUDA](https://pytorch.org/get-started/previous-versions/).
