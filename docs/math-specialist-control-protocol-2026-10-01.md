# Controle adicional: modelo matemático de 1,5B

Registrado após observar o diagnóstico dos modelos locais. Este controle é
**exploratório e pós-hoc**; não modifica a matriz predefinida de seis braços
nem constitui teste confirmatório independente.

Comparar Qwen2.5-1.5B-Instruct geral com
`Qwen/Qwen2.5-Math-1.5B-Instruct`, revisão
`aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`, nos mesmos 30 casos de
expressão direta com 1, 2 e 4 atualizações, semente de dados 20261002.
Usar BF16, geração gulosa, limite de 128 tokens e o mesmo pedido de
**somente um inteiro**. Registrar respostas brutas, taxa de resposta no
formato e acurácia exata; não reparar saídas discursivas.

O tamanho nominal é semelhante, mas pesos, corpus, pós-treino e possível
formato preferido de resposta diferem. Um ganho do modelo Math apoiaria a
hipótese de que treinamento matemático especializado ajuda mais do que o
pequeno ajuste LoRA local; **não isolaria causalmente** qual etapa desse
treinamento produziu a diferença. O modelo Math será baixado publicamente
uma vez e executado localmente, sem alterar dependências do projeto.

## Extensão pós-hoc, registrada antes da segunda execução

No primeiro controle, o modelo Math não respeitou o pedido de somente um
inteiro; gerou explicações e frequentemente atingiu o limite de 128 tokens.
Logo, zero acertos estritos nessa condição não mede sua competência
matemática. Será feita uma segunda execução sobre as **mesmas 30 contas**
usando a instrução de raciocínio passo a passo e resposta final em
`\boxed{}` indicada na ficha oficial do modelo, limite de 512 tokens.
Uma resposta só será aceita se houver **exatamente um inteiro em uma caixa
final**. Esta mudança de prompt/limite impede comparação causal direta com
o controle de inteiro simples; o primeiro resultado será preservado e
relatado como incompatibilidade de formato.
