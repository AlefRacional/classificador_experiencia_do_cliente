# CLASSIFICADOR DE SENTIMENTOS EM PYTHON - ETAPA 2 - ANÁLISE EXPLORATÓRIA DE DADOS
# Objetivo: Visualizar a distribuição dos sentimentos no dataset em graficos.

# Importar a biblioteca pandas para ler o dataset
import pandas as pd

# Importar bibliotecas para criar gráficos
import matplotlib.pyplot as plt
import seaborn as sns

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

# Criar o DataFrame 
df = pd.DataFrame(dados)

# Contar quantas frases existem para cada tipo de sentimento
contagem_sentimentos = df['sentimento'].value_counts()

# Vamos exibir o resultado no terminal
print("📊 Contagem de frases por tipo de sentimento:")
print(contagem_sentimentos)

# Criar um gráfico de barras para mostrar a distribuição
plt.figure(figsize=(6, 4))
sns.countplot(x='sentimento', data=df, palette='Set2')

# Adicionar título e rótulos
plt.title('Distribuição de Sentimentos no Dataset Simulado')
plt.xlabel('Sentimento')
plt.ylabel('Quantidade de Frases')

# Exibir o gráfico
plt.tight_layout()
plt.show()

# Exibir algumas frases de cada tipo de sentimento
print("\n🗨️ Exemplos de frases por sentimento:")
for sentimento in df['sentimento'].unique():
    print(f"\n{sentimento.upper()}:")
    exemplos = df[df['sentimento'] == sentimento]['frase'].head(2)
    for frase in exemplos:
        print(f"- {frase}")
