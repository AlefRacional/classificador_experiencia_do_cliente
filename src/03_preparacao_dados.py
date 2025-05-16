# CLASSIFICADOR DE SENTIMENTOS EM PYTHON - ETAPA 3 - PREPARAÇÃO DOS DADOS (VETORIZAÇÃO)
# Objetivo: Transformar as frases em vetores numéricos para que o modelo de IA possa entender

# # Importar a biblioteca pandas para manipular tabelas de dados.
import pandas as pd

# Para transformar frases em números (vetores), para que o modelo de IA possa entender
from sklearn.feature_extraction.text import CountVectorizer

# Para dividir os dados em dois grupos: um para treinar a IA e outro para testar depois
from sklearn.model_selection import train_test_split

# é o modelo de IA usado para treinar e aprender com os dados (classificador Naive Bayes)
from sklearn.naive_bayes import MultinomialNB

# mostra um resumo da performance do modelo com métricas como precisão e f1-score
from sklearn.metrics import classification_report

# cria uma tabela que mostra onde o modelo acertou ou errou comparando resultado real e previsto
from sklearn.metrics import confusion_matrix

# cria um gráfico visual bonito a partir da matriz de confusão
from sklearn.metrics import ConfusionMatrixDisplay

# biblioteca usada para criar gráficos em Python como o da matriz de confusão
import matplotlib.pyplot as plt

# Carregar o mesmo dataset da etapa 1
dados = {
    'frase': [
        'a entrega foi muito rápida e o produto é ótimo',
        'péssimo atendimento não recomendo',
        'o produto veio certo mas demorou para chegar',
        'excelente qualidade vou comprar de novo',
        'não gostei do suporte foi confuso e demorado',
        'achei ok nada demais',
        'estou muito satisfeito com a compra',
        'veio com defeito e ninguém resolveu meu problema',
        'foi tudo certo mas esperava mais',
        'a loja foi super atenciosa e resolveu meu problema rapidamente'
    ],
    'sentimento': [
        'positivo',
        'negativo',
        'neutro',
        'positivo',
        'negativo',
        'neutro',
        'positivo',
        'negativo',
        'neutro',
        'positivo'
    ]
}

# Cria o DataFrame
df = pd.DataFrame(dados)

# Divide os dados em variáveis de entrada e saída
X = df['frase']            # Frases (entrada)
y = df['sentimento']       # Sentimentos (saída)

# Agora é hora de transformar as frases em vetores de palavras.
# Cada palavra vira uma coluna com contagem de vezes que aparece na frase

vectorizador = CountVectorizer()        # Cria o vetor de palavras (Bag of Words)
X_vetor = vectorizador.fit_transform(X) # Aplica a vetorização

# Dividir os dados entre treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X_vetor, y, test_size=0.2, random_state=42)

# Mostrar um exemplo de como as frases viraram números
print("🔢 Vetorização (visualização parcial):")
print(X_vetor.toarray()[:2])  # Mostra as duas primeiras frases vetorizadas

# Mostrar as palavras que viraram colunas no vetor
print("\n🗂️ Palavras no vocabulário:")
print(vectorizador.get_feature_names_out())