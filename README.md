# Análise de sentimentos de um restaurante utilizando NLP

## Introdução 

### Identificação 
* Augusto César da Silva Carvalho, 20230029867
* Rita de Cassia Melo Nascimento, 20210017617
  
### Informações Gerais 
#### Descrição da Base de Dados de Avaliações do Restaurante Camarões

#### Objetivo
A base de dados contém informações sobre as avaliações feitas por usuários acerca do restaurante *Camarões* na plataforma *TripAdvisor*, com o objetivo de analisar os sentimentos e as tendências de satisfação. A base será utilizada para o treinamento de modelos de machine learning para previsão de avaliações, contribuindo para a compreensão do comportamento dos consumidores e para a tomada de decisões estratégicas.

#### Fonte dos Dados
- **Plataforma**: TripAdvisor
- **Data de Coleta**: Dados coletados até o mês de dezembro de 2024.

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
- **Gemini API**
  
Para a análise de sentimentos e previsão das avaliações, foram utilizados os modelos **SIA** (Sentiment Intensity Analyzer) e **Gemini**.

A API do Gemini é uma interface de programação de aplicativos (API) que permite integrar as funcionalidades da plataforma Gemini, desenvolvida pela Google, em aplicativos Python. A API é projetada para realizar tarefas de processamento de linguagem natural (PLN), como análise de sentimentos, reconhecimento de entidades, tradução de texto, análise de emoções, entre outras.

A API do Gemini geralmente oferece acesso a diversos modelos treinados para lidar com textos em várias línguas e realizar tarefas complexas de PLN. É importante destacar que, enquanto o Gemini como um produto é amplamente acessado via interfaces gráficas, a API permite automação e integração em sistemas personalizados.

Usando a API para Análise de Sentimentos: A API do Gemini pode ser usada para análise de sentimentos e outras tarefas relacionadas. Você pode usar a API para analisar texto, identificar sentimentos, emoções, ou realizar outras tarefas de PLN. O processo de uso típico envolve enviar uma solicitação com texto para a API e obter a resposta de volta.

A API do Gemini pode exigir um plano de pagamento dependendo do volume de uso, já que muitas APIs de Google Cloud têm limites gratuitos, mas cobram por uso além desses limites.
Durante o desenvolvimento do projeto, para contornar a limitação de uso da API do Gemini, decidimos selecionar uma **quantidade pequena de comentários** para análise em cada execução, de forma a evitar exceder os limites gratuitos da API. As análises de sentimentos eram realizadas apenas sobre um subconjunto de comentários, e os resultados eram **salvos em um arquivo CSV**, onde as avaliações eram constantemente adicionadas, permitindo o acompanhamento contínuo sem ultrapassar os custos associados ao uso excessivo da API. Essa estratégia ajudou a minimizar os gastos, mas não resolveu completamente o problema da limitação de volume de requisições, pois ainda assim era necessário aguardar um tempo para fazer outra requisição. 

- **SentimentIntensityAnalyzer**

O **SentimentIntensityAnalyzer (SIA)** é uma ferramenta da biblioteca **NLTK** (Natural Language Toolkit) utilizada para análise de sentimentos. Ela avalia o texto e gera pontuações que indicam a polaridade e a intensidade emocional do conteúdo. O SIA utiliza um modelo baseado em léxico, que analisa a frequência de palavras associadas a sentimentos positivos, negativos ou neutros.

O SIA gera quatro pontuações principais:
1. **Positivo (pos)**: Quão positivo o texto é.
2. **Negativo (neg)**: Quão negativo o texto é.
3. **Neutro (neu)**: Quão neutro o texto é.
4. **Compound**: Uma pontuação agregada que reflete o sentimento geral do texto, variando de -1 (extremamente negativo) a +1 (extremamente positivo).

O **SIA** é simples de usar e é eficaz para analisar sentimentos em textos curtos, como resenhas de produtos ou posts em redes sociais. Contudo, pode ter limitações em textos mais complexos, com sarcasmo ou ambiguidade.

#### Exemplo de Dados

| Título                      | Comentário                                                   | Estrelas |
|-----------------------------|--------------------------------------------------------------|----------|
| "Ótima comida e ambiente!"   | "O ambiente é perfeito para um jantar a dois. A comida estava deliciosa e o atendimento muito bom!" | 5        |
| "Decepção com o serviço"     | "A comida estava boa, mas o atendimento foi muito demorado. Não voltarei tão cedo." | 2        |
| "Recomendo a todos!"         | "Comida excelente, ambiente agradável e ótimo atendimento. Recomendo a todos!" | 4        |

#### Limitações
- **Representatividade**: A base contém apenas avaliações de um único restaurante, o que pode limitar a generalização dos resultados para outros estabelecimentos.
- **Possíveis viéses**: A maioria das avaliações podem ser de clientes altamente satisfeitos ou insatisfeitos, o que pode afetar a distribuição das avaliações.



## Metodologia   
A técnica de *machine learning* utilizada neste projeto foi o Processamento de Linguagem Natural (Natural Language Processing - NLP). Essa área da inteligência artificial é voltada para a interação entre humanos e máquinas por meio da linguagem natural. O NLP permite que computadores processem, analisem e interpretem textos ou falas humanas, identificando padrões e extraindo informações relevantes.  

A aplicação específica foi a análise de sentimentos, uma abordagem que utiliza modelos de NLP para classificar emoções ou opiniões expressas em textos, como positivas, negativas ou neutras. Para este estudo, os dados analisados foram os comentários de avaliação do restaurante Camarões.

O primeiro passo foi o tratamento dos dados textuais, deixando os comentários normalizados para facilitar a interpretação pelo modelo **SIA (SentimentIntensityAnalyzer)**. Os processos de normalização incluíram:  
- **Remoção de acentos e caracteres especiais**: Garantiu uniformidade no texto.  
- **Padronização de caixa (lowercase)**: Todos os textos foram convertidos para letras minúsculas.  
- **Remoção de stopwords**: Palavras sem relevância semântica, como "de", "o", "a", foram excluídas.  
- **Eliminação de espaços em excesso e emojis**: Reduziu ruídos desnecessários no texto.

Após isso foi aplicado o método ```sia.polarity_scores()``` em cada elemento da lista contendo os comentarios:
```python
for i, row in data.iterrows():
    text = row['Coments_norm']
    text_t = traduzir(text)
    text_t_norm = text_normalizer(text_t)
    time.sleep(2)
    pol.append(sia.polarity_scores(text_t_norm))
    data.loc[i, 'Coments_norm'] = text_t_norm
```
Para que o método do **SIA** funcionasse, foi necessário traduzir os comentários, originalmente, em português (ma grande maioria dos casos), para inglês. A tradução foi realizada com a biblioteca **googletrans**, utilizando a seguinte função:  
```for``` anterior:
```python
def traduzir(text):
    translator = Translator()
    traducao = translator.translate(str(text), dest='en').text
    return traducao
```
A partir das pontuações geradas pelo SIA, foi realizada uma previsão da avaliação do usuário, convertendo os sentimentos em notas de 1 a 5 estrelas. Após análises e testes cuidadosos, chegou-se a uma lógica que considera as variáveis `compound`, `pos`, `neg` e `neu` para atribuir as estrelas:
```python
for score in pol:
    compound = score['compound']
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']

    if compound >= 0.7 and pos > neg and pos > neu/2:
        sia_stars.append(5)  # Muito positivo
    elif compound >= 0.3 and pos > neg:
        sia_stars.append(4)  # Positivo
    elif compound >= 0.05 and pos > neg - 0.1 and pos > neu/3:
        sia_stars.append(3) # Neutro tendendo ao positivo
    elif compound <= -0.7 and neg > pos:
        sia_stars.append(1)  # Muito negativo
    elif compound <= -0.3 and neg > pos :
        sia_stars.append(2)  # Negativo
    elif neg > pos + 0.2 and neg > neu/3 :
        sia_stars.append(2) # Neutro tendendo ao negativo
    else:
        sia_stars.append(3)  # Neutro
```
Em seguida, o resultado foi guardado [neste arquivo CSV](csv_folder/camaroes_sia_stars.csv), para ser analiado posteriormente.


## Códigos
### Treinamento 
### Tratamento do resultado do treinamento

```python

```


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
