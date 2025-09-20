"""
Lê o arquivo de log gerado pelo 'treinamento_com_log.py'
e apresenta um relatório sobre a efetividade dos treinamentos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_ARQUIVO = 'log_treinamento.csv'

def gerar_relatorio():
    """Lê o log e gera os relatórios textual e visual."""
    
    # Verifica se o arquivo de log existe
    if not os.path.exists(LOG_ARQUIVO):
        print(f"Erro: Arquivo de log '{LOG_ARQUIVO}' não encontrado.")
        print("Execute o script 'treinamento_com_log.py' pelo menos uma vez para gerá-lo.")
        return

    # Carrega os dados usando pandas
    df = pd.read_csv(LOG_ARQUIVO)

    # --- 1. Relatório Textual ---
    print("="*80)
    print(" " * 20 + "RELATÓRIO DE EFETIVIDADE DOS TREINAMENTOS")
    print("="*80)
    print("\nAnálise de cada sessão de treinamento (um a um):\n")
    
    # Adiciona uma coluna 'sessao' para facilitar a leitura
    df.index.name = 'Sessão'
    df.index = df.index + 1
    
    print(df.to_string())
    
    print("\n--- Análise ---")
    print("A 'taxa de assertividade' em um problema como este é o inverso do erro.")
    print("Valores MENORES para 'erro_treinamento_final' e 'erro_validacao_final' indicam um modelo MAIS PRECISO.")
    print("Observe como o 'erro_validacao_final' tende a diminuir à medida que o 'total_amostras' aumenta.")
    
    # --- 2. Relatório Visual ---
    if len(df) < 2:
        print("\nÉ necessário ter pelo menos duas sessões de treinamento para gerar um gráfico de evolução.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(df.index, df['erro_treinamento_final'], marker='o', linestyle='--', label='Erro de Treinamento')
    ax.plot(df.index, df['erro_validacao_final'], marker='s', linestyle='-', label='Erro de Validação (Mais Importante)')
    
    # Melhora a visualização
    ax.set_title('Evolução da Efetividade do Modelo a Cada Treinamento', fontsize=16)
    ax.set_xlabel('Sessão de Treinamento', fontsize=12)
    ax.set_ylabel('Erro Quadrático Médio (Quanto menor, melhor)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(df.index) # Garante que todos os números de sessão apareçam no eixo X
    
    # Adiciona um texto explicativo no gráfico
    plt.figtext(0.5, 0.01, 
                "Este gráfico mostra que o modelo está aprendendo. A tendência de queda na linha 'Erro de Validação' \nindica que a precisão do modelo aumenta a cada novo treinamento com mais dados.", 
                ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    print("\nExibindo gráfico de evolução... Feche a janela do gráfico para finalizar.")
    plt.show()


if __name__ == "__main__":
    gerar_relatorio()