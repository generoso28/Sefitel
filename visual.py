"""
Simulador Visual Interativo da Lei de Snell com IA
"""

# Importação das bibliotecas necessárias
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
MODELO_ARQUIVO = 'modelo_snell.keras'
def calcular_angulo_real(theta1_rad, n1, n2):
    argumento_arcsin = (n1 / n2) * np.sin(theta1_rad)
    if abs(argumento_arcsin) > 1:
        return None  # Indica reflexão interna total
    return np.arcsin(argumento_arcsin)

def desenhar_simulacao(n1, n2, theta1_rad, theta2_pred_rad, theta2_real_rad):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Desenha a interface entre os meios
    ax.axhline(0, color='black', linewidth=2)
    # Desenha a linha normal (perpendicular à interface)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    # 1. Desenha o raio incidente
    # O raio termina em (0,0). Calculamos o ponto de partida.
    x_incidente = -np.sin(theta1_rad)
    y_incidente = np.cos(theta1_rad)
    ax.plot([x_incidente, 0], [y_incidente, 0], 'g-', label='Raio Incidente', linewidth=2)

    # 2. Desenha o raio refratado REAL
    if theta2_real_rad is not None:
        x_real = np.sin(theta2_real_rad)
        y_real = -np.cos(theta2_real_rad)
        ax.plot([0, x_real], [0, y_real], 'r--', label='Refração Real (Lei de Snell)', linewidth=2)
    else:
        # Caso de reflexão interna total
        x_refletido = -np.sin(theta1_rad)
        y_refletido = -np.cos(theta1_rad)
        ax.plot([0, x_refletido], [0, y_refletido], 'r--', label='Reflexão Interna Total (Real)', linewidth=2)


    # 3. Desenha o raio refratado PREVISTO PELA IA
    x_pred = np.sin(theta2_pred_rad)
    y_pred = -np.cos(theta2_pred_rad)
    ax.plot([0, x_pred], [0, y_pred], 'b-', label='Refração Prevista pela IA', linewidth=2)

    # Adiciona textos e legendas
    ax.text(-1.4, 1.0, f'Meio 1 (n1 = {n1})', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(-1.4, -1.0, f'Meio 2 (n2 = {n2})', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    titulo = (f'Simulação da Refração\n'
              f'Ângulo de Incidência: {np.degrees(theta1_rad):.2f}°')
    ax.set_title(titulo)
    ax.legend(loc='lower left')
    ax.axis('off') # Remove os eixos para um visual mais limpo
    plt.show()

def main():
    # Carrega o modelo treinado
    if not os.path.exists(MODELO_ARQUIVO):
        print(f"Erro: Arquivo do modelo '{MODELO_ARQUIVO}' não encontrado.")
        print("Por favor, execute o script de treinamento primeiro para gerar o modelo.")
        return
    
    print(f"Carregando modelo de IA de '{MODELO_ARQUIVO}'...")
    modelo = load_model(MODELO_ARQUIVO)
    print("Modelo carregado com sucesso!")

    while True:
        try:
            print("\n--- Nova Simulação ---")
            n1_meio = 1.0 # Fixo como ar para simplificar
            n2_input = float(input("Digite o índice de refração do segundo meio (ex: 1.5 para vidro): "))
            theta1_input_graus = float(input("Digite o ângulo de incidência em graus (0 a 90): "))

            if not (0 <= theta1_input_graus <= 90):
                print("Por favor, insira um ângulo entre 0 e 90 graus.")
                continue
            theta1_rad = np.radians(theta1_input_graus) # Converte para radianos

            # Prepara os dados para o modelo de IA
            input_ia = np.array([[theta1_rad, n2_input]])
            
            theta2_pred_rad = modelo.predict(input_ia)[0][0]  # # Faz a previsão com a IA, obtendo o ângulo em escalar
            # Calcula o valor real
            theta2_real_rad = calcular_angulo_real(theta1_rad, n1_meio, n2_input)

            print("\n--- Resultados ---")
            print(f"Ângulo de Refração Real (calculado): {np.degrees(theta2_real_rad):.2f}°" if theta2_real_rad is not None else "Reflexão Interna Total (Real)")
            print(f"Ângulo de Refração Previsto (IA):   {np.degrees(theta2_pred_rad):.2f}°")
            
            desenhar_simulacao(n1_meio, n2_input, theta1_rad, theta2_pred_rad, theta2_real_rad) # Gera a imagem da simulação

        except ValueError:
            print("Entrada inválida. Por favor, digite apenas números.")
        continuar = input("\nDeseja fazer outra simulação? (s/n): ").lower()
        if continuar!= 's':
            break

if __name__ == "__main__":
    main()