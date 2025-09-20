"""
Treinamento incremental do modelo da Lei de Snell.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

MODELO_ARQUIVO = 'modelo_snell.keras'
DADOS_X_ARQUIVO = 'dados_X.npy'
DADOS_Y_ARQUIVO = 'dados_y.npy'
LOG_ARQUIVO = 'log_treinamento.csv'

def calcular_angulo_refracao(theta1, n1, n2):
    argumento_arcsin = (n1 / n2) * np.sin(theta1)
    argumento_arcsin = np.clip(argumento_arcsin, -1.0, 1.0)
    return np.arcsin(argumento_arcsin)

def registrar_log(num_amostras, history):
    """Registra os resultados da sessão de treinamento no arquivo CSV."""
    # Cabeçalho do arquivo
    header = ['timestamp', 'total_amostras', 'erro_treinamento_final', 'erro_validacao_final']
    
    # Dados da sessão atual
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Pega o erro da última época de treinamento
    erro_treino = history.history['loss'][-1]
    erro_validacao = history.history['val_loss'][-1]
    
    nova_linha = [timestamp, num_amostras, erro_treino, erro_validacao]
    
    # Escreve no arquivo
    file_exists = os.path.isfile(LOG_ARQUIVO)
    with open(LOG_ARQUIVO, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)  # Escreve o cabeçalho se o arquivo for novo
        writer.writerow(nova_linha)
    
    print(f"Resultados da sessão registrados em '{LOG_ARQUIVO}'.")

# Parâmetros
num_novas_amostras = 2000
n1_meio = 1.0

# Geração de novos dados
novos_angulos_incidencia = np.random.uniform(0, np.pi/2, num_novas_amostras)
novos_indices_n2 = np.random.uniform(1.3, 2.0, num_novas_amostras)
novos_angulos_refracao = calcular_angulo_refracao(novos_angulos_incidencia, n1_meio, novos_indices_n2)
X_novo = np.vstack([novos_angulos_incidencia, novos_indices_n2]).T
y_novo = novos_angulos_refracao

# Carrega e combina dados antigos
if os.path.exists(DADOS_X_ARQUIVO):
    print("Dados anteriores encontrados. Carregando e combinando...")
    X_antigo = np.load(DADOS_X_ARQUIVO)
    y_antigo = np.load(DADOS_Y_ARQUIVO)
    X = np.vstack([X_antigo, X_novo])
    y = np.concatenate([y_antigo, y_novo])
else:
    print("Nenhum dado anterior encontrado.")
    X = X_novo
    y = y_novo

# Salva o dataset combinado
np.save(DADOS_X_ARQUIVO, X)
np.save(DADOS_Y_ARQUIVO, y)
print(f"Dataset atualizado. Tamanho total: {len(X)} amostras.")

# Carrega ou cria o modelo
if os.path.exists(MODELO_ARQUIVO):
    print(f"Carregando modelo de '{MODELO_ARQUIVO}'...")
    modelo = load_model(MODELO_ARQUIVO)
else:
    print("Criando um novo modelo.")
    modelo = Sequential()
    modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.summary()

# Treinamento
print("\nIniciando treinamento...")
history = modelo.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=1)
print("Treinamento concluído!")

# Salva o modelo aprimorado
modelo.save(MODELO_ARQUIVO)
print(f"Modelo aprimorado salvo em '{MODELO_ARQUIVO}'.")

# --- 3. Registro dos Resultados ---
registrar_log(len(X), history)