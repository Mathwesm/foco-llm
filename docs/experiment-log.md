# Diário de experimentos

## 2026-09-26 — preparação do piloto

**Natureza:** desenvolvimento e validação da infraestrutura. Não é treinamento nem medição de desempenho de LLM.

Foram implementados geração determinística, verificadores de respostas e evidências, splits por problema, avaliação pareada e gráficos. Os testes usam respostas fabricadas somente para verificar o avaliador, em diretórios temporários. Esses valores não constituem resultados científicos.

Falhas encontradas nos testes iniciais:

- O validador aceitava uma lista incompleta de evidências. Foi incluída verificação independente por remoção de fatos; o teste de regressão falhou antes da correção.
- Uma execução sem qualquer resposta era rejeitada, impedindo registrar uma falha total do modelo. Foi corrigida para contar ausências como erros e deixar a taxa condicional indefinida quando necessário.
- O filtro de diretórios da varredura de segredos não reconhecia os separadores do Windows. Foi corrigido e coberto por teste de regressão, mantendo o código e as configurações do projeto na varredura.

Execução local de preparação: seed 42, 100 problemas por tarefa, 300 problemas-base e 1.200 exemplos. O manifesto registra 960 exemplos de treino, 120 de validação e 120 de teste. A partição é `2026-09-27/969659029e986d86` em UTC (a execução ocorreu na noite de 26/09 no horário de São Paulo). O gráfico de composição foi inspecionado visualmente. Nenhuma resposta de LLM foi produzida nessa execução.

## 2026-09-27 — primeiro baseline local

Execução `5fab79eb7f10fc4d`, código `c7c272c`, Qwen2.5-0.5B-Instruct em FP16 na RTX 3060 de 6 GB. Foram reaproveitados cinco checkpoints da sessão anterior e geradas as 115 respostas restantes. A tentativa inicial de retomada offline falhou com `OfflineModeIsEnabled` durante consulta de metadados pelo tokenizer; nenhuma resposta foi alterada. A retomada com rede concluiu normalmente.

Resultado: 120 respostas brutas, todas encerradas por EOS, sem limite de tokens ou tempo atingido. Todas começaram com bloco Markdown e foram rejeitadas pelo parser estrito. Acurácia estrita zero em todas as condições; não interpretar como incapacidade de raciocínio nem ausência de efeito dos distratores. O teste final não foi usado para inferência.

Geração acumulada: 175,88 segundos e 5.213 tokens de saída. Pico de memória alocada pelo PyTorch: 1.039.616.512 bytes; reservada: 1.080.033.280 bytes. Essas medidas excluem carregamento/download e não representam toda a memória utilizada na GPU. O intervalo de relógio inclui uma pausa solicitada entre sessões; não é usado como duração computacional.

As métricas foram recalculadas a partir das respostas brutas antes da exportação; o gráfico foi inspecionado visualmente. [Relatório, reprodução e artefatos](baseline-2026-09-27.md). Não houve ajuste de pesos nem mudança do parser para melhorar retroativamente esta execução.

## 2026-09-27 — separação de formato, resposta e evidências

As 120 respostas existentes foram reavaliadas sem inferência ou treinamento adicional. O protocolo `content-envelope-v2` mantém o relatório estrito e acrescenta aceitação de um único bloco Markdown completo, sem reparo de conteúdo. Verifica correspondência de prompts e recalcula o baseline estrito antes da publicação.

Foram lidos 120 objetos; 102 tinham identificadores de evidência válidos e 18 usavam referências inexistentes. Exigindo identificadores válidos, houve 22 acertos; avaliando a resposta independentemente desses identificadores, 37/120. Seleção exata de evidências: 35/120. Esses números são leituras diferentes das mesmas saídas, não ganhos do modelo.

A exportação intermediária v1 foi preservada; a v2 acrescenta a métrica de resposta independente. Foram acrescentados testes para cercas incompletas/múltiplas, texto externo, chaves duplicadas, schema incorreto, evidências inválidas, denominadores, integridade de checkpoints e preservação da leitura estrita. O gráfico foi inspecionado. [Protocolo, resultados e limitações](content-evaluation-2026-09-27.md).

## Como registrar cada treinamento futuro

Cada execução deverá registrar:

| Campo | Informação necessária |
|---|---|
| Objetivo | Hipótese e contraste testado |
| Código | Commit efetivamente executado |
| Modelo | Identificador, revisão, licença e precisão |
| Dados | Hash, versão do gerador, splits e tarefas usadas |
| Configuração | Seed, prompts, lote, contexto, otimizador, taxa de aprendizado e adaptadores |
| Execução | Início/fim UTC, hardware, duração, pico de memória e interrupções |
| Resultados | Métricas por tarefa/condição e caminhos dos artefatos |
| Comparação | Baseline equivalente e orçamento utilizado |
| Limitações | Falhas de formato, respostas ausentes, vieses e condições não avaliadas |

Manter curvas de treinamento, dispersão pareada, tabelas e intervalos de confiança ligados aos arquivos que os originaram. Não substituir resultados antigos nem preencher medidas que não foram obtidas. As curvas de treinamento só serão geradas quando houver histórico real de treinamento.
