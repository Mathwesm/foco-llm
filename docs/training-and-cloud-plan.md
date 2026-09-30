# Treinamento local, repetições e nuvem

## Perguntas separadas

1. O ajuste melhora a resposta e a resistência a distratores em relação ao mesmo checkpoint original?
2. Supervisionar a seleção de fatos acrescenta benefício à supervisão da resposta?
3. Esse benefício transfere para tarefas ausentes do ajuste?
4. O resultado varia entre tamanhos da mesma família?
5. A mesma configuração pode ser reproduzida na nuvem, e qual é a diferença de custo e desempenho operacional?

Executar na nuvem não melhora o modelo por si só. Trocar simultaneamente modelo, dados, precisão e hardware impede atribuir uma mudança a um único fator. Três modelos diferentes não substituem três repetições do mesmo treinamento.

## Ordem recomendada

| Fase | Execução | Critério de conclusão |
|---|---|---|
| Baselines | Checkpoints originais 0,5B e 1,5B no piloto, sem ajuste | Medir competência limpa, formato, evidências e recursos |
| Benchmark revisado | Variar linguagem, estruturas e dificuldade; separar templates | Congelar dados e protocolo antes dos controles de treinamento |
| Teste funcional local | Poucos passos com adaptadores em um modelo que caiba na GPU | Verificar máscara de perda, atualização dos pesos esperados, checkpoint e recarga |
| Experimento local | Controles de treinamento e três sementes por configuração escolhida | Resultados comparáveis, curvas e avaliação independente |
| Reprodução em nuvem | Repetir uma configuração local com mesmos artefatos e versões | Comparar métricas, duração, memória e custo; documentar diferenças numéricas |
| Ampliação | Segundo tamanho e terceiro apenas se os recursos permitirem | Comparação por tamanho sem misturar efeitos de plataforma |
| Transferência | Treinar em tarefa-fonte e avaliar em tarefas-alvo reservadas | Matriz fonte/alvo e comparação com controles |

O terceiro modelo ainda não está escolhido. A seleção depende de memória medida, competência inicial e orçamento. Não é necessário começar com um cluster distribuído; uma máquina com GPU pode atender o piloto. A plataforma de nuvem, tipo de GPU, orçamento máximo e credenciais serão definidos antes de contratar ou executar recursos pagos. BigQuery é útil para consultas analíticas, mas não é o ambiente proposto para ajustar a LLM em GPU.

## Matriz inicial de controles

| Condição | Dados de ajuste | Supervisão |
|---|---|---|
| C0 | Nenhum | Checkpoint original |
| C1 | Limpos | Resposta |
| C2 | Com distratores | Resposta |
| C3 | Mesmos exemplos de C2 | Resposta e identificadores dos fatos relevantes |

Primeiro validar um modelo e uma semente em pequena escala. Depois executar C1–C3 com três sementes distintas (por exemplo, 42, 43 e 44). Isso representa nove treinamentos por tamanho, além do baseline sem ajuste. Dois tamanhos representam dezoito treinamentos; três, vinte e sete. A escolha final da matriz depende do tempo medido no piloto, sem reduzir silenciosamente o estudo aprovado.

O contraste central C2 versus C3 deve manter checkpoint inicial, exemplos, ordem, passos, adaptadores e política de geração comparáveis. A supervisão adicional altera o objetivo e pode alterar o número de tokens com perda: registrar esse orçamento e fazer um controle correspondente se necessário. Não prometer igualdade simultânea de exemplos, passos e tokens supervisionados quando os alvos diferirem.

Antes de implementar, definir e testar a serialização e a máscara de perda: os tokens de prompt não recebem perda; evidências corretas não entram no prompt de inferência; resposta e evidências têm posições explicitamente documentadas; a leitura do campo de resposta não depende de evidências válidas. Não permitir que uma diferença de formato torne um controle artificialmente pior. O contrato de saída e as métricas ficam congelados para todos os braços.

## Qualidade de saída e “qualidade dos tokens”

- **Resposta:** acurácia exata com normalização previamente definida; rejeições e ausências permanecem no denominador.
- **Relevância:** precisão, revocação e correspondência exata dos IDs de evidência; não inferir mecanismo interno a partir desses IDs.
- **Distratores:** transições acerto→erro e erro→acerto em versões pareadas, além da condição limpa.
- **Eficiência:** tokens gerados, tempo e memória; uma resposta curta só é melhor quando atende à tarefa.
- **Aprendizado:** curvas de perda e, se implementada, probabilidade dos alvos em validação. Perda menor não substitui avaliação comportamental.
- **Incerteza:** variação entre sementes de treinamento e intervalos por reamostragem de problemas-base, mantendo todas as variantes juntas.

Não chamar todo token de bom ou ruim isoladamente: sua utilidade depende da tarefa e da resposta completa. Texto plausível, fluência e explicações longas não são evidência suficiente de correção ou de um processo interno de raciocínio.

## Portabilidade e registro

O código deverá receber configuração por arquivo/CLI, usar Poetry e caminhos relativos, salvar revisão exata do modelo, hashes de dados, commit, sementes, versões, hardware e parâmetros. Saídas ficam em diretórios novos por execução. Checkpoints devem permitir retomada; artefatos precisam sobreviver ao desligamento da máquina na nuvem.

Antes de execuções longas, configurar alerta de falha com credenciais por ambiente e limites explícitos de duração/custo. Não versionar credenciais ou pesos de acesso restrito. Comparar primeiro FP16 com FP16, ou a mesma quantização dos dois lados; mudanças necessárias de backend/precisão tornam a comparação aproximada e devem ser identificadas.

Um experimento local bem-sucedido não garante igualdade bit a bit em outra GPU. A conclusão sobre melhora ou piora deve considerar erros pareados, variabilidade e controles, não apenas uma diferença pequena na média.

## Estado atual

### Alternativa gratuita: Kaggle (29/09/2026)

Considerar Kaggle antes de contratar GPU paga, conforme sugestão do autor. A
[documentação oficial](https://www.kaggle.com/docs/efficient-gpu-usage) informa
cota semanal; confirmar saldo, GPU disponível e duração máxima na conta antes
de estimar o experimento. A disponibilidade pode ter fila. Não houve execução
ou autenticação no Kaggle nesta etapa.

Usar o mesmo repositório e Poetry, registrar o ambiente efetivamente fornecido,
salvar checkpoints e exportar resultados antes do encerramento da sessão.
Verificar compatibilidade de Python, CUDA e precisão; não presumir que BF16
local funciona na GPU fornecida. Se for necessário mudar precisão, registrar
como outra configuração e não atribuir a diferença somente ao hardware.

O [teste funcional de treinamento local](training-smoke-2026-09-28.md) foi implementado e executado: seis passos LoRA com retomada e recarga verificadas. A matriz científica C1/C2/C3, suas repetições e a comparação em nuvem ainda não foram executadas. O artigo e o teste final continuam reservados.
