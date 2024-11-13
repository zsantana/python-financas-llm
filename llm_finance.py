import pandas as pd
import os

df = pd.DataFrame()

for file in os.listdir("extratos"):
    if file.endswith(".csv"):
        file_path = os.path.join("extratos", file)
        df_temp = pd.read_csv(file_path, encoding='utf-8', header=None, names=["Data", "Valor", "descricao", "idx"])
        df_temp["Valor"] = pd.to_numeric(df_temp["Valor"], errors='coerce')
        df_temp["Data"] = pd.to_datetime(df_temp["Data"], errors='coerce')
        df = pd.concat([df, df_temp])

# if "ID" in df.columns:
#     df = df.set_index("ID")
df = df.dropna(subset=["Data"], how='any')
print(df)


# ===============
# LLM

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
template = """
Você é um analista de dados, trabalhando em um projeto de limpeza de dados.
Seu trabalho é escolher uma categoria adequada para cada lançamento financeiro
que vou te enviar.

Todos são transações financeiras de uma pessoa física.

Escolha uma dentre as seguintes categorias:
- Alimentação
- Receitas
- Saúde
- Mercado
- Saúde
- Educação
- Compras
- Transporte
- Investimento
- Transferências para terceiros
- Telefone
- Moradia

Escola a categoria deste item:
{text}

Responda apenas com a categoria.
"""

# Local LLM
prompt = PromptTemplate.from_template(template=template)


# Groq
chat = ChatGroq(model="llama-3.1-70b-versatile")
chain = prompt | chat | StrOutputParser()

categorias = chain.batch(list(df["descricao"].values))
df["Categoria"] = categorias

df = df.drop(df[df["descricao"] == "Aguardo a descrição do item"].index)  # ignora a linha com descricao "Aguardo a descrição do item"

# df = df[df["Data"] >= datetime(2024, 3, 1).date()]
df.to_csv("finances.csv")