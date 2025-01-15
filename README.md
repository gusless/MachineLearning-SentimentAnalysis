# Análise de sentimentos de um restaurante

## Introdução 

### Identificação 
* Augusto César da Silva Carvalho, 20230029867
* Rita de Cassia Melo Nascimento, 20210017617
### Informações Gerais
* Descrever o problema.  
#### Descrição da Base de Dados de Avaliações do Restaurante Camarões

#### Objetivo
A base de dados contém informações sobre as avaliações feitas por usuários acerca do restaurante **Camarões** na plataforma **TripAdvisor**, com o objetivo de analisar os **sentimentos** e as **tendências de satisfação**. A base será utilizada para o treinamento de **modelos de machine learning** para **previsão de avaliações**.

#### Fonte dos Dados
- **Plataforma**: TripAdvisor
- **Data de Coleta**: Dados coletados até o mês de **dezembro de 2024**.

#### Estrutura dos Dados
A base de dados é composta por um arquivo no formato **CSV**, com as seguintes colunas:

- **Título**: Título da avaliação feita pelo usuário.
- **Comentário**: Texto completo do comentário da avaliação.
- **Avaliação**: Nota atribuída em uma escala de 1 a 5.

#### Tamanho da Base de Dados
- **Número de Avaliações**: 16.387 avaliações.
- **Número de Colunas**: 3 (Título, Comentário e Avaliação).

#### Pré-processamento
Os dados foram tratados para ficarem legíveis para os modelos de machine learning. As seguintes transformações foram realizadas:

- **Retirada de Acentos, Emojis e Espaços em Branco**: Os elementos foram removidos para garantir consistência nos dados textuais.
- **Remoção de Stopwords**: Stopwords (palavras comuns e sem significado relevante, como "a", "de", "o", etc.) foram removidas dos comentários.

#### Métodos Utilizados
Para a análise de sentimentos e previsão das avaliações, foram utilizados os modelos **SIA** (Sentiment Intensity Analyzer) e **Gemini**.

#### Exemplo de Dados

| Título                      | Comentário                                                   | Estrelas |
|-----------------------------|--------------------------------------------------------------|----------|
| "Ótima comida e ambiente!"   | "O ambiente é perfeito para um jantar a dois. A comida estava deliciosa e o atendimento muito bom!" | 5        |
| "Decepção com o serviço"     | "A comida estava boa, mas o atendimento foi muito demorado. Não voltarei tão cedo." | 2        |
| "Recomendo a todos!"         | "Comida excelente, ambiente agradável e ótimo atendimento. Recomendo a todos!" | 4        |

#### Uso Potencial
A base de dados pode ser utilizada para:
- **Análise de sentimentos**: Classificação de avaliações em positivas, negativas ou neutras.
- **Treinamento de modelos preditivos**: Desenvolver modelos de machine learning para prever a nota de avaliação com base nos comentários.

#### Limitações
- **Representatividade**: A base contém apenas avaliações de um único restaurante, o que pode limitar a generalização dos resultados para outros estabelecimentos.
- **Possíveis viéses**: A maioria das avaliações podem ser de clientes altamente satisfeitos ou insatisfeitos, o que pode afetar a distribuição das avaliações.



## Metodologia   
* A técnica de _machine learning_ utilizada neste projeto foi o _Natural Language Processing_ (NLP). O Processamento de Linguagem Natural trata-se da área voltada para a compreensão e interação entre humanos e máquinas por meio da linguagem natural. Esta técnica permite que computadores processem, analisem e interpretem textos ou falas humanas, identificando padrões e extraindo informações relevantes.

  A aplicação específica foi a análise de sentimentos, uma abordagem que utiliza modelos de NLP para classificar emoções ou opiniões expressas em textos, como positivas, negativas ou neutras. Para este estudo, os dados analisados foram os comentários de avaliação do restaurante Camarões.
* Explicar as etapas do treinamento e teste. 
* Caso tenha selecionado atributos, explicar a motivação para a seleção de tais atributos.
## Códigos
* Mostrar trechos de códigos mais importantes e explicações.  
* Informar o link para acessar o código. 

## Experimentos
* Descrever em detalhes os tipos de testes executados. 
* Descrever os parâmentros avaliados. 
* Explicar os resultados. 
*


## Conclusão 
* O trabalho atendeu aos objetivos? 
*
*




---
