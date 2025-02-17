import warnings
import pandas as pd
import os
import pandas as pd  # Certifique-se de importar o pandas
from aux_functions import *

# Caminho da pasta que contém os arquivos
caminho_pasta = r"C:\Users\leona\OneDrive\Ambiente de Trabalho\IA_Keras_Tay\classificador"

# Filtrar apenas os arquivos Excel
arquivos_excel = [arquivo for arquivo in os.listdir(
    caminho_pasta) if arquivo.endswith(('.xlsx', '.xls'))]

# Criar caminhos completos para os arquivos
caminhos_completos = [os.path.join(caminho_pasta, arquivo)
                      for arquivo in arquivos_excel]

# Salvar o primeiro arquivo na variável (se existir)
primeiro_arquivo = caminhos_completos[0] if caminhos_completos else None

if primeiro_arquivo:
    print("Primeiro arquivo Excel encontrado:", primeiro_arquivo)

    # Ler o arquivo Excel
    df_original = pd.read_excel(primeiro_arquivo)

    # Filtrar linhas com 'DESCRIÇÃO DA AVARIA' não nulas e não vazias
    df_trated_1 = df_original[df_original['X_Text_input'].notna() & (
        df_original['X_Text_input'] != '')]

    # Filtrar linhas onde 'OCORRÊNCIA' está nulo ou vazio
    # df_trated_2 = df_trated_1[df_trated_1['OCORRÊNCIA'].isna() | (df_trated_1['OCORRÊNCIA'] == '')]
    # df_trated_2 = df_trated_2[(df_trated_2['RETENÇÃO OBRIGATÓRIA'] == 'SIM (REV)')]

    # Exibir as primeiras 5 linhas do DataFrame resultante
    print(df_trated_1.head(5))
else:
    print("Nenhum arquivo Excel encontrado na pasta especificada.")


# df_trated_3  = df_trated_2[['DESCRIÇÃO DA AVARIA']].reset_index()
# df_trated_3['X_Text_input']  = df_trated_3['DESCRIÇÃO DA AVARIA']
df_original = df_trated_1


warnings.filterwarnings("ignore")


def identify(text_input):
    print("------")
    y_sistema_var = y_sistema(text_input)
    y_conjunto_var = y_conjunto(text_input)
    y_item_var = y_item(text_input)
    y_problema_var = y_problema(text_input)
    y_ocorrencia_var = y_ocorrencia(
        y_sistema_var, y_conjunto_var, y_item_var, y_problema_var)
    print(text_input)
    print("Sistema: ", y_sistema_var)
    print("Conjunto: ", y_conjunto_var)
    print("Item: ", y_item_var)
    print("Problema: ", y_problema_var)
    print("------")

    # Criando o DataFrame
    return {
        "resultado_IA_Sistema": y_sistema_var,
        "resultado_IA_Conjunto": y_conjunto_var,
        "resultado_IA_Item": y_item_var,
        "resultado_IA_Problema": y_problema_var,
        "resultado_IA_Ocorrência": y_ocorrencia_var
    }

# Função auxiliar para lidar com erros


def safe_identify(text_input):

    try:
        return identify(text_input)
    except Exception as e:
        # Retorna valores genéricos em caso de erro
        print("------")
        print(e)
        print("------")
        return {
            "resultado_IA_Sistema": "Error",
            "resultado_IA_Conjunto": "Error",
            "resultado_IA_Item": "Error",
            "resultado_IA_Problema": "Error",
            "resultado_IA_Ocorrência": "Error"
        }


resultados = df_original["X_Text_input"].apply(safe_identify).apply(pd.Series)
df_result = pd.concat([df_original, resultados], axis=1)
print("Salvar Saida")
df_result.to_excel('classificador\Saída_final.xlsx', index=False)
print("end")
