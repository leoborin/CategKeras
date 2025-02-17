import pandas as pd
from aux_functions import *
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

import warnings
warnings.filterwarnings("ignore")
print("init")

max_words = 10000  # Número máximo de palavras no vocabulário
max_len = 100      # Tamanho máximo de sequência

# Carregar modelos, tokenizers e label encoders uma vez
modelo_sistema = load_model("modelo_classificacao_y_Sistema.h5")
with open("tokenizer_Sistema.pkl", "rb") as f:
    tokenizer_sistema = pickle.load(f)
with open("label_encoder_Sistema.pkl", "rb") as f:
    label_encoder_sistema = pickle.load(f)

modelo_conjunto = load_model("modelo_classificacao_y_CONJUNTO.h5")
with open("tokenizer_CONJUNTO.pkl", "rb") as f:
    tokenizer_conjunto = pickle.load(f)
with open("label_encoder_CONJUNTO.pkl", "rb") as f:
    label_encoder_conjunto = pickle.load(f)

modelo_item = load_model("modelo_classificacao_y_Item.h5")
with open("tokenizer_Item.pkl", "rb") as f:
    tokenizer_item = pickle.load(f)
with open("label_encoder_Item.pkl", "rb") as f:
    label_encoder_item = pickle.load(f)

modelo_problema = load_model("modelo_classificacao_y_Problema.h5")
with open("tokenizer_PROBLEMA.pkl", "rb") as f:
    tokenizer_problema = pickle.load(f)
with open("label_encoder_PROBLEMA.pkl", "rb") as f:
    label_encoder_problema = pickle.load(f)

modelo_ocorrencia = load_model("modelo_multientrada_y_OCORRÊNCIA.h5")
with open("tokenizers_OCORRÊNCIA.pkl", "rb") as f:
    tokenizers_ocorrencia = pickle.load(f)
with open("label_encoder_y_OCORRÊNCIA.pkl", "rb") as f:
    label_encoder_ocorrencia = pickle.load(f)


def y_sistema(text_input):
    # Fazer previsão com o modelo carregado
    novo_texto = [text_input]
    novo_texto_seq = tokenizer_sistema.texts_to_sequences(novo_texto)
    novo_texto_padded = pad_sequences(
        novo_texto_seq, maxlen=max_len, padding='post', truncating='post')

    pred_proba = modelo_sistema.predict(novo_texto_padded)
    pred_classe = np.argmax(pred_proba, axis=1)
    pred_label = label_encoder_sistema.inverse_transform(pred_classe)

    return pred_label[0]


def y_conjunto(text_input):
    # Fazer previsão com o modelo carregado
    novo_texto = [text_input]
    novo_texto_seq = tokenizer_conjunto.texts_to_sequences(novo_texto)
    novo_texto_padded = pad_sequences(
        novo_texto_seq, maxlen=max_len, padding='post', truncating='post')

    pred_proba = modelo_conjunto.predict(novo_texto_padded)
    pred_classe = np.argmax(pred_proba, axis=1)
    pred_label = label_encoder_conjunto.inverse_transform(pred_classe)

    return pred_label[0]


def y_item(text_input):
    # Fazer previsão com o modelo carregado
    novo_texto = [text_input]
    novo_texto_seq = tokenizer_item.texts_to_sequences(novo_texto)
    novo_texto_padded = pad_sequences(
        novo_texto_seq, maxlen=max_len, padding='post', truncating='post')

    pred_proba = modelo_item.predict(novo_texto_padded)
    pred_classe = np.argmax(pred_proba, axis=1)
    pred_label = label_encoder_item.inverse_transform(pred_classe)

    return pred_label[0]


def y_problema(text_input):
    # Fazer previsão com o modelo carregado
    novo_texto = [text_input]
    novo_texto_seq = tokenizer_problema.texts_to_sequences(novo_texto)
    novo_texto_padded = pad_sequences(
        novo_texto_seq, maxlen=max_len, padding='post', truncating='post')

    pred_proba = modelo_problema.predict(novo_texto_padded)
    pred_classe = np.argmax(pred_proba, axis=1)
    pred_label = label_encoder_problema.inverse_transform(pred_classe)

    return pred_label[0]


def y_ocorrencia(y_sistema, y_conjunto, y_item, y_problema):
    # 3. Definir uma nova entrada para teste
    novo_exemplo = {
        "Y_SISTEMA_output": [y_sistema],
        "Y_CONJUNTO_output": [y_conjunto],
        "Y_ITEM_output": [y_item],
        "Y_PROBLEMA_output": [y_problema]
    }

    # 4. Preprocessar as novas entradas
    novo_exemplo_padded = [
        pad_sequences(tokenizers_ocorrencia[col].texts_to_sequences(
            novo_exemplo[col]), maxlen=max_len, padding='post')
        for col in novo_exemplo.keys()
    ]

    # 5. Fazer previsões
    pred_proba = modelo_ocorrencia.predict(novo_exemplo_padded)
    pred_class = np.argmax(pred_proba, axis=1)
    pred_label = label_encoder_ocorrencia.inverse_transform(pred_class)

    return pred_label[0]
