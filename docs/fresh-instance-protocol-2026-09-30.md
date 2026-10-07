# Verificação local em novas instâncias

Protocolo registrado antes de gerar ou avaliar as novas instâncias.

- Gerador: benchmark v2.1, 100 problemas-base por tarefa, semente
  **20261001**. O split de validação contém 120 casos: 40 por tarefa,
  distribuídos igualmente entre contexto limpo e três distratores.
- Modelos fixados: Qwen2.5-1.5B-Instruct original e o adaptador `mixed-15b`
  da rodada dirigida. Mesma revisão base, prompt, decodificação gulosa BF16,
  máximo de 128 tokens e pontuação estrita para ambos.
- Métricas primárias: acertos estritos totais e por tarefa. Secundárias:
  acertos por condição, evidências exatas e respostas em JSON válido.
- Nenhum novo treino, escolha de checkpoint ou mudança de prompt será feita
  com base nesses 120 casos. O teste final antigo não será reaberto.

São **novas instâncias sintéticas**, não novos templates nem dados naturais.
Assim, essa verificação mede robustez a outra semente dentro do mesmo
gerador; não estabelece transferência para textos reais ou outros idiomas.
O adaptador deve ser verificado contra o hash do dataset usado no treino,
enquanto o dataset novo deve ter hash próprio no manifesto de inferência.
