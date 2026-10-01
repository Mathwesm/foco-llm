# Intervenção exploratória: decompor a conta em chamadas curtas

Registrada após o diagnóstico de profundidade. O modelo treinado com
subtotais acertou 10/10 expressões de uma atualização sob o contrato JSON,
mas não acertou nenhuma das dez expressões de quatro atualizações no pedido
de número simples. Esta intervenção verifica se a **orquestração externa**
de quatro chamadas curtas melhora a mesma conta sem alterar seus pesos.

Serão usadas as dez expressões de quatro atualizações do dataset
`arithmetic-diagnostic-v1` (semente 20261002). Um parser determinístico
separará o valor inicial e as quatro operações já explícitas; essa é uma
**condição oracular de seleção**, não um teste de leitura do texto com
distratores. A cada passo, o modelo receberá apenas `Calculate A +/- B.
Reply with only the integer.` O resultado produzido, certo ou errado,
entrará no próximo passo. Uma saída que não seja um inteiro faz o caso
falhar. Registrar as quatro saídas brutas, o primeiro erro, o acerto de
cada passo e o acerto final.

Comparar primeiro o Qwen 1,5B original, `mixed-15b` e `math-prefix-15b`
nas mesmas dez expressões, BF16, geração gulosa, limite de 128 tokens por
chamada. O cálculo independente permanece como gabarito. O custo é de até
quatro gerações por caso; um eventual ganho é do **sistema com
decomposição**, não demonstra que o LLM passou a fazer a conta inteira
sozinho. Nenhum resultado desta rodada altera o teste reservado anterior.
