# Próximas etapas e critérios de conclusão

## 1. Baseline local — execução atual

Executar o Qwen2.5-0.5B-Instruct sem ajuste nos 120 exemplos de validação do piloto. A revisão é `7ae557604adf67be50417f59c2c2f167def9a775`; o modelo possui aproximadamente 0,49 bilhão de parâmetros e licença Apache 2.0, conforme sua [ficha oficial](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

Usar FP16 na RTX 3060 de 6 GB, lote unitário, geração gulosa e limite de 128 tokens novos. Registrar respostas brutas, erros de formato, tokens, duração de geração e pico de memória alocada/reservada pelo PyTorch. Esses picos não representam todo o uso de memória da GPU por outros programas. Downloads, carregamento, tokenização e gravação ficam fora da duração de geração.

**Conclusão:** execução real preservada, métricas por tarefa/condição, gráfico e limitações documentadas. O teste final permanece reservado. Esta etapa mede um modelo já ajustado para instruções, sem ajuste adicional neste projeto.

## 2. Revisar o benchmark

Inspecionar os erros de validação; diversificar enunciados, profundidade do raciocínio, posição e quantidade dos distratores. Evitar pistas triviais que denunciem quais frases ignorar. Separar templates e estruturas entre treino e avaliação, além de manter variantes do mesmo problema juntas. Revisar manualmente uma amostra estratificada e registrar defeitos encontrados.

**Conclusão:** versão nova e imutável dos dados, verificadores independentes, protocolo congelado e ausência de vazamento entre splits. Resultados do piloto atual não são misturados com os do benchmark novo.

## 3. Ajuste piloto

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
