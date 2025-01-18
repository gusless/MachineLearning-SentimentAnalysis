# Análise de Sentimentos de um Restaurante Utilizando NLP

## Introdução 

### Identificação 
* Augusto César da Silva Carvalho, 20230029867
* Rita de Cassia Melo Nascimento, 20210017617

---

### Objetivo
A base de dados contém informações sobre as avaliações feitas por usuários acerca do restaurante *Camarões* na plataforma *TripAdvisor*, com o objetivo de analisar os sentimentos e as tendências de satisfação. A base será utilizada para o treinamento de modelos de machine learning para previsão de avaliações, contribuindo para a compreensão do comportamento dos consumidores e para a tomada de decisões estratégicas.

#### Fonte dos Dados
- **Plataforma**: TripAdvisor
- **Data de Coleta**: Dados coletados até o mês de dezembro de 2024.

### Estrutura dos Dados
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

---

## Metodologia   
### SentimentIntensityAnalyzer
A técnica de *machine learning* utilizada neste projeto foi o Processamento de Linguagem Natural (Natural Language Processing - NLP). Essa área da inteligência artificial é voltada para a interação entre humanos e máquinas por meio da linguagem natural. O NLP permite que computadores processem, analisem e interpretem textos ou falas humanas, identificando padrões e extraindo informações relevantes.  

A aplicação específica foi a análise de sentimentos, uma abordagem que utiliza modelos de NLP para classificar emoções ou opiniões expressas em textos, como positivas, negativas ou neutras. Para este estudo, os dados analisados foram os comentários de avaliação do restaurante Camarões.

O primeiro passo foi o [tratamento dos dados textuais](#Etapas-de-Normalização-dos-Textos), deixando os comentários normalizados para facilitar a interpretação pelo modelo **SIA (SentimentIntensityAnalyzer)**.

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

Para que o método do **SIA** funcionasse, foi necessário traduzir os comentários, originalmente, em português (na grande maioria dos casos), para inglês. A tradução foi realizada com a biblioteca **googletrans**, utilizando a seguinte função:  
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

Os resultados finais, contendo as previsões de estrelas para cada comentário, foram salvos no arquivo [CSV](csv_folder/camaroes_sia_stars.csv), para ser analiado posteriormente. E o código completo pode ser encontrado [aqui](code_folder/camaroes_sentiment.py).

### Gemini API

A API do Gemini permite utilizar a inteligência artificial de forma similar à sua plataforma, possibilitando o uso de prompts diretamente no código. Além disso, a API pode ser empregada para processar variáveis de texto, automatizar prompts, entre outras funcionalidades.

No entanto, a API possui limitações de uso gratuito, que são restabelecidas após algumas horas, e exige um plano pago para volumes maiores. Para lidar com essa restrição, foi implementada uma estratégia que seleciona um pequeno intervalo de comentários (no máximo 50), definido pelo usuário, para análise em cada execução. Isso evita exceder os limites gratuitos da API.

O código abaixo solicita ao usuário a escolha de um intervalo válido para análise:

```python
print(f"Escolha um intervalo de 50 comentários, dentre {len(df)} comentários, para o Gemini analisar:")

a = 1
b = 0
max = 50
while int(a)>int(b) or int(a)<0 or int(b)>len(df) or int(b)-int(a)>max:
    a = input("Início: ")
    b = input("Fim: ")
    if not a.isnumeric() or not b.isnumeric():
        print("Intervalo inválido, digite novamente. Apenas números são permitidos.")
        a = 1
        b = 0
    elif int(a)>int(b) or int(a)<0 or int(b)>len(df):
        print("Intervalo inválido, digite novamente")
    elif int(b)-int(a)>max:
        print("Intervalo muito grande, digite novamente")
```
Após definir o intervalo, as análises de sentimentos foram realizadas utilizando um único prompt. O prompt pede ao Gemini que analise os comentários e adivinhe a nota atribuída pelos clientes (de 1 a 5 estrelas). Segue o código utilizado:

```python
try:
    prompt = (
        f"Aqui está um pequeno trecho do meu banco de dados:\n{dados.to_string()}\n"
        f"Agora tente adivinhar a nota dada pelos clientes dos comentários de 1 a 5 estrelas. "
        f"Envie sua avaliação sem explicações, apenas com os números formatados da seguinte forma: "
        f"índice:quantidade de estrelas"
    )
    response = chat.send_message(prompt)
    print(response.text)

    time.sleep(10)
except:
    print("Gemini fora do ar")
```

O uso de `try`/`except` foi necessário, considerando a possibilidade de instabilidade da API. Além disso, o `time.sleep(10)` foi incluído para evitar sobrecarga na API e respeitar os limites de requisições. Essa abordagem permitiu realizar análises contínuas e armazenar os resultados em um arquivo CSV, sem exceder os custos do serviço.

Os resultados das análises foram salvos no arquivo [CSV](csv_folder/camaroes_gemini_stars.csv), onde as avaliações eram [constantemente adicionadas](#Etapas-para-Atualização-das-Avaliações-do-Gemini), permitindo o acompanhamento contínuo sem ultrapassar os custos associados ao uso excessivo da API. Essa estratégia ajudou a minimizar os gastos, mas não resolveu completamente o problema da limitação de volume de requisições, pois ainda assim era necessário aguardar um tempo para fazer outra requisição. 

O código completo pode ser encontrado [aqui](code_folder/camaroes_gemini.ipynb).

---

## Outros códigos
### Etapas de Normalização dos Textos

Abaixo estão as funções utilizadas no processo de normalização dos textos, com breves explicações:

1. **converter_minusculo(text)**  
   - Converte todo o texto para letras minúsculas, garantindo uniformidade.
```python
def converter_minusculo(text):
    return text.lower()
```
2. **remove_espaco_branco(text)**  
   - Remove espaços em branco extras no início e no fim do texto.
```python
def remove_espaco_branco(text):
    return text.strip()
```
3. **remove_pontuacao(text)**  
   - Remove pontuações do texto, exceto apóstrofos, para reduzir ruídos.
```python
def remove_pontuacao(text):
    punct_str = string.punctuation
    punct_str = punct_str.replace("'", "")
    translator = str.maketrans("", "", punct_str)
    return text.translate(translator)
```
4. **remove_emoji(text)**  
   - Remove emojis usando um padrão Unicode, deixando o texto mais limpo.
```python
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", text)
```
5. **remove_http(text)**  
   - Remove URLs presentes no texto utilizando expressões regulares.
```python
def remove_http(text):
    http = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
    pattern = re.compile(http, re.IGNORECASE)
    return pattern.sub("", text)
```
6. **remove_stopwords(text)**  
   - Remove palavras comuns sem relevância semântica (stopwords), como "de", "o", "a", foram excluídas.  
```python
regexp = RegexpTokenizer(r"\b\w+\b")
linguas = ['portuguese', 'english', 'spanish', 'french']
stops = []
for lingua in linguas:
    stops += nltk.corpus.stopwords.words(lingua)
def remove_stopwords(text):
    return " ".join([word for word in regexp.tokenize(text) if word not in stops])
```
7. **text_normalizer(text)**  
   - Combina todas as funções acima em uma única sequência de normalização:  
```python
def text_normalizer(text):
    text = unidecode.unidecode(text)
    text = re.sub('\n', '', text)
    text = remove_http(text)
    text = remove_emoji(text)
    text = remove_pontuacao(text)
    text = remove_espaco_branco(text)
    text = converter_minusculo(text)
    text = remove_stopwords(text)
    return text
```
O código completo pode ser achado neste [arquivo](code_folder/camaroes_sentiment.py).

### Etapas para Atualização das Avaliações do Gemini

Este código atualiza um arquivo CSV com avaliações processadas pela API Gemini, evitando duplicações e garantindo organização. Abaixo está a explicação de cada bloco:

#### 1. Conversão Segura de Valores

```python
def safe_convert(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
```

- Função para tentar converter valores em `int` ou `float`. Caso a conversão falhe, retorna o valor original.

#### 2. Leitura do Arquivo CSV

```python
listarquivo = []
try:
    with open(caminho, "r") as arquivo:
        reader = csv.reader(arquivo)
        next(reader, None)
        listarquivo = [[safe_convert(x) for x in row] for row in reader]
except FileNotFoundError:
    listarquivo = []
```

- Lê o arquivo CSV especificado no caminho.
- Converte os valores usando a função `safe_convert`.
- Ignora o cabeçalho. Se o arquivo não existir, inicia com uma lista vazia.

#### 3. Tratamento e Filtragem de Valores no Arquivo

```python
listarquivo = [[int(x) if isinstance(x, str) and x.isdigit() else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in row] for row in listarquivo]
existing_comment_numbers = [int(row[0]) for row in listarquivo if isinstance(row[0], (int, float))]
gemilist = [[int(x) if isinstance(x, str) and x.isdigit() else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in row] for row in gemilist]
gemilist_filtered = [row for row in gemilist if int(row[0]) not in existing_comment_numbers]
if gemilist_filtered:
    with open(caminho, "a", newline='') as arquivo:
        writer = csv.writer(arquivo)
        writer.writerows(gemilist_filtered)
```
- Garante que os valores no arquivo sejam convertidos para números (`int` ou `float`), sempre que possível.
- Extrai os números dos comentários existentes no arquivo para evitar duplicações.
- Converte os novos dados (`gemilist`) e filtra os comentários que já existem no arquivo.
- Adiciona ao arquivo somente os dados filtrados que não estavam presentes anteriormente.

#### 4. Ordenação dos Dados e Regravação do Arquivo com Cabeçalho

```python
try:
    with open(caminho, "r") as arquivo:
        reader = csv.reader(arquivo)
        next(reader, None)
        dados = list(reader)
except FileNotFoundError:
    dados = []

dados.sort(key=lambda row: int(row[0]))

with open(caminho, "w", newline='') as arquivo:
    writer = csv.writer(arquivo)
    writer.writerow(["Número do Comentário", "Estrelas Gemini"])
    writer.writerows(dados)
```

- Recarrega o arquivo atualizado e ordena os dados pelo número do comentário em ordem crescente.
- Reescreve o arquivo, adicionando o cabeçalho e os dados ordenados.

---

## Resultados


* Descrever em detalhes os tipos de testes executados. 
* Descrever os parâmentros avaliados. 
* Explicar os resultados. 

---

## Conclusão 
* O trabalho atendeu aos objetivos? 

---
