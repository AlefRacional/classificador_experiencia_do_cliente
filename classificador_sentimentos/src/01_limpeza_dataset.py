# ============================================
# CLASSIFICADOR DE SENTIMENTOS EM PYTHON - ETAPA 1
# Autor: Allef Hiago Gomes da Silva
# Descrição: Simulação de um dataset + limpeza básica
# ============================================

# Importamos a biblioteca pandas, muito usada para manipular tabelas de dados.
import pandas as pd

# Etapa 1: Criar um dataset simulado com frases e seus respectivos sentimentos.
# Para isso, usamos um dicionário com duas colunas: 'frase' e 'sentimento'.
# A coluna 'frase' contém textos fictícios como se fossem avaliações de clientes.
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

# Criamos um DataFrame (estrutura de tabela) a partir dos dados simulados.
df = pd.DataFrame(dados)

# Exibimos as primeiras linhas da tabela para garantir que tudo está correto.
print("🟢 Dataset original:")
print(df)

# Etapa 2: Limpeza básica dos textos (pré-processamento)

# 1. Vamos transformar todos os textos em letras minúsculas
df['frase'] = df['frase'].str.lower()

# 2. Podemos remover pontuações com a biblioteca 're' (expressões regulares)
import re

# Função que remove qualquer caractere que não seja letra ou número
def limpar_texto(texto):
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    return texto

# Aplicamos a função de limpeza em cada frase
df['frase'] = df['frase'].apply(limpar_texto)

# 3. Eliminamos espaços extras (se houver)
df['frase'] = df['frase'].str.strip()

# Exibimos novamente o dataset após limpeza
print("\n🧼 Dataset após limpeza:")
print(df)

# 4. Verificamos se há dados faltando (NaN) ou duplicados
print("\n🔍 Verificando dados ausentes ou duplicados:")
print("Dados nulos por coluna:")
print(df.isnull().sum())
print("Entradas duplicadas:")
print(df.duplicated().sum())

# Pronto! Temos um mini-dataset limpo e estruturado para treinar um modelo de IA.