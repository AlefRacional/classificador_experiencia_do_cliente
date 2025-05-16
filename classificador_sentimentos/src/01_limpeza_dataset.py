
# CLASSIFICADOR DE SENTIMENTOS EM PYTHON - ETAPA 1 - Limpeza dos dados
# Descrição: Simulação de um dataset + limpeza básica


# Importar a biblioteca pandas para manipular tabelas de dados.
import pandas as pd

# Usar o dicionário com duas colunas 'frase' e 'sentimento'.
# A coluna 'frase' contém os textos de avaliações de clientes.
# A coluna 'sentimento' indica se a frase é positiva, negativa ou neutra.

dados = {
    'frase': [
        'A entrega foi muito rápida e o produto é ótimo!',
        'Péssimo atendimento, não recomendo.',
        'O produto veio certo, mas demorou para chegar.',
        'Excelente qualidade, vou comprar de novo!',
        'Não gostei do suporte, foi confuso e demorado.',
        'Achei ok. Nada demais.',
        'Estou muito satisfeito com a compra!',
        'Veio com defeito e ninguém resolveu meu problema.',
        'Foi tudo certo, mas esperava mais.',
        'A loja foi super atenciosa e resolveu meu problema rapidamente.'
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

# Criar um DataFrame a partir dos dados que temos.
df = pd.DataFrame(dados)

# Exibir as primeiras linhas da tabela para garantir que tudo está correto.
print("🟢 Dataset original:")
print(df)

# Etapa 2: Limpeza básica dos textos (pré-processamento)

# Transformar os textos em letras minúsculas
df['frase'] = df['frase'].str.lower()

# Remover pontuações com a biblioteca 're' (expressões regulares)
import re

# Remover qualquer caractere que não seja letra ou número
def limpar_texto(texto):
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação XD!
    return texto

# Aplicar a função de limpeza em cada frase
df['frase'] = df['frase'].apply(limpar_texto)

# Eliminamos espaços extras
df['frase'] = df['frase'].str.strip()

# Exibimos novamente o dataset após limpeza
print("\n🧼 Dataset após limpeza:")
print(df)

# Verifique se há dados (NaN) ou duplicados
print("\n🔍 Verificando dados ausentes ou duplicados:")
print("Dados nulos por coluna:")
print(df.isnull().sum())
print("Entradas duplicadas:")
print(df.duplicated().sum())
