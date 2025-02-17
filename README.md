# CategKeras

![image](https://github.com/user-attachments/assets/e3f8674b-b1cf-4b8d-9acc-5b1088a9edf7)


Ferramentas e Frameworks: 

| Pyhton | Pandas
LabelEncoder
**Tokenizer** 
Keras
**Dropout** 
deep learning LSTM
cx_freeze |
| --- | --- |

### **Descrição para o Stakeholder:**

O objetivo deste projeto foi desenvolver uma ferramenta para **categorizar defeitos** com base na descrição escrita pelo mecânico de campo.

Anteriormente, o processo era **100% manual**, realizado por um analista que recebia as informações do sistema e fazia a categorização manualmente em uma planilha do Excel.

Para otimizar a produtividade, foi desenvolvido um **executável** com uma **inteligência artificial treinada** a partir dos preenchimentos manuais realizados anteriormente. Agora, a categorização é feita de forma **automática**, e o sistema retorna um arquivo Excel já categorizado.

### **Descrição Técnica:**

### **Coleta de Dados**

- Tabela histórica preenchida por analista categorizando entrada →
    
    | Sistema | Conjunto | Item | Problema |
    
- Variável de entrada → Texto de Linguagem natural

### **Pré-processamento dos Dados**

- Para preparar os dados para o modelo, as categorias da variável **y** foram convertidas em valores numéricos utilizando o **LabelEncoder**. Isso é necessário para que o modelo possa processar as saídas categóricas.
- Os dados foram divididos em conjuntos de treino e teste, com 80% dos dados destinados ao treinamento e 20% para teste, utilizando a função **train_test_split**.

### **Tokenização e Padronização**

- O texto presente em **X** foi tokenizado, ou seja, convertido em sequências de números inteiros, onde cada número representa uma palavra específica no vocabulário. O **Tokenizer** foi configurado para considerar um máximo de 10.000 palavras no vocabulário.
- As sequências de texto foram padronizadas para terem o mesmo comprimento (**max_len = 100**), utilizando o método **pad_sequences**. Isso garante que todas as entradas tenham o mesmo tamanho, o que é necessário para o modelo de rede neural.

### **Construção do Modelo**

- Um modelo de rede neural sequencial foi construído utilizando a biblioteca **Keras**. O modelo consiste em:
    1. Uma camada de **Embedding** para converter as palavras tokenizadas em vetores densos de tamanho 128.
    2. Uma camada **LSTM** com 128 unidades, que é capaz de capturar dependências temporais no texto.
    3. Uma camada de **GlobalMaxPool1D** para reduzir a dimensionalidade das saídas da LSTM.
    4. Uma camada de **Dropout** com taxa de 0.3 para evitar overfitting.
    5. Uma camada densa (**Dense**) com 64 unidades e ativação **ReLU**.
    6. Uma camada densa final com ativação **softmax** para classificação multiclasse, onde o número de neurônios é igual ao número de classes únicas em **y**.

### **Compilação e Treinamento**

- O modelo foi compilado utilizando o otimizador **Adam** e a função de perda **sparse_categorical_crossentropy**, que é adequada para problemas de classificação com rótulos inteiros.
- O modelo foi treinado por 100 épocas, com um tamanho de lote (**batch_size**) de 32. Durante o treinamento, o desempenho do modelo foi monitorado tanto no conjunto de treino quanto no conjunto de teste.

### **Avaliação e Visualização**

- O desempenho do modelo foi avaliado com base na acurácia durante o treinamento e validação.
- A história do treinamento (**history**) foi armazenada, permitindo a visualização da evolução da acurácia e da perda ao longo das épocas.

### **Conclusão**

- O modelo desenvolvido é capaz de classificar textos em múltiplas categorias, utilizando uma abordagem de deep learning com LSTM.
- O pré-processamento dos dados, incluindo tokenização e padronização, foi crucial para garantir que o modelo pudesse processar os textos de forma eficiente.
- O uso de técnicas como **Dropout** e **GlobalMaxPool1D** ajudou a evitar overfitting e a melhorar a generalização do modelo.

### Entrega

- Foi feito um modelo para cada categoria e compilados em um Script_final.
- Executável para transformar uma Entrada.xlsx em Saída.xlsx

> Para mais detalhes sobre o código e os resultados, consulte o repositório do Git.
> 

### **Resultados e Ganhos:**

- O modelo final teve uma assertividade de 90% de assertividade em dados novos.
- Redução de 95% no tempo de preenchimento das categorias, agora substituído por uma simples conferência dos resultados gerados pelo modelo.
