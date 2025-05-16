# CLASSIFICADOR DE SENTIMENTOS EM PYTHON - ETAPA 5 - AVALIAÇÃO DO MODELO COM GRÁFICOS
# Objetivo: Avaliar o desempenho da IA com métricas e visualização

# BIBLIOTECAS IMPORTADAS

# Importa a biblioteca Pandas, usada para carregar e manipular conjuntos de dados em DataFrames
import pandas as pd  

# Ferramenta do scikit-learn para transformar textos em uma matriz de contagem de palavras
from sklearn.feature_extraction.text import CountVectorizer  

# Divide os dados em conjuntos de treino e teste para avaliar a performance do modelo
from sklearn.model_selection import train_test_split  

# Algoritmo de classificação baseado em Naive Bayes, eficiente para análise de textos
from sklearn.naive_bayes import MultinomialNB  

# Métricas para avaliar o desempenho do modelo, incluindo relatório de classificação e matriz de confusão
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  

# Biblioteca para criação de gráficos, útil para visualizar a matriz de confusão
import matplotlib.pyplot as plt  

# DADOS para ultilização

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

# TREINAMENTO DO MODELO

# Cria o DataFrame
df = pd.DataFrame(dados)

# Separação dos dados
X = df['frase']
y = df['sentimento']

# Vetorização das frases
vectorizador = CountVectorizer()
X_vetor = vectorizador.fit_transform(X)

# Divisão treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_vetor, y, test_size=0.2, random_state=42)

# Treina o modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Faz previsões com os dados de teste
y_pred = modelo.predict(X_test)

# AVALIAÇÃO COM RELATÓRIO

print("📈 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred))

# MATRIZ DE CONFUSÃO

# Cria a matriz de confusão
cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

# Exibe visualmente
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)

# Configura o gráfico
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Classificação de Sentimentos")
plt.xlabel("Previsto pelo modelo")
plt.ylabel("Sentimento real")
plt.tight_layout()

# Mostra o gráfico
plt.show()
