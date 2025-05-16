# CLASSIFICADOR DE SENTIMENTOS EM PYTHON - ETAPA 4 - TREINAMENTO DO MODELO DE IA
# Objetivo: Treinar a IA para classificar sentimentos a partir de texto

import pandas as pd  # Biblioteca para manipulação de dados em formato de tabelas

# Ferramentas do scikit-learn para processamento e análise de textos
from sklearn.feature_extraction.text import CountVectorizer  # Converte texto em uma matriz de contagem de palavras
from sklearn.model_selection import train_test_split  # Divide os dados em conjuntos de treino e teste
from sklearn.naive_bayes import MultinomialNB  # Algoritmo de aprendizado de máquina baseado em Naive Bayes
from sklearn.metrics import classification_report, confusion_matrix  # Métricas para avaliar a performance do modelo

import pandas as pd  
# Importa CountVectorizer do scikit-learn, que transforma textos em uma matriz de contagem de palavras
from sklearn.feature_extraction.text import CountVectorizer  
# Ferramenta do scikit-learn para dividir os dados em conjuntos de treino e teste
from sklearn.model_selection import train_test_split  
# Importação do modelo Naive Bayes multinomial, usado para classificação de textos
from sklearn.naive_bayes import MultinomialNB  # Modelo escolhido
# Importa funções para avaliar o desempenho do modelo usando métricas como matriz de confusão e relatório de classificação
from sklearn.metrics import classification_report, confusion_matrix  
# Dataset simulado sempre o mesmo das etapas anteriores
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

# Separar os dados X = entrada, y = saída
X = df['frase']
y = df['sentimento']

# Vetorização (transforma frases em números)
vectorizador = CountVectorizer()
X_vetor = vectorizador.fit_transform(X)

# Dividir em treino e teste 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(X_vetor, y, test_size=0.2, random_state=42)

# Cria o modelo de Naive Bayes
modelo = MultinomialNB()

# Treina o modelo com os dados de treino
modelo.fit(X_train, y_train)

# Faz previsões com os dados de teste
y_pred = modelo.predict(X_test)

# Avalia o desempenho do modelo
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão para ver acertos e erros
print("\n Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Teste com frases novas
frases_novas = [
    "adorei o atendimento, foi muito bom",
    "não recomendo esse produto",
    "ok, nada demais"
]

# Vetorizar essas novas frases com o mesmo vetor usado antes
frases_vetor = vectorizador.transform(frases_novas)

# Previsão da IA
previsoes = modelo.predict(frases_vetor)

# Mostrar resultado
print("\n Teste com frases novas:")
for frase, sentimento in zip(frases_novas, previsoes):
    print(f'"{frase}" → Sentimento previsto: {sentimento}')
