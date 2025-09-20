# Projeto Sefitel: Simulador da Lei de Snell com Inteligência Artificial

Este projeto demonstra como uma **rede neural** pode ser treinada para aprender e prever o comportamento de um fenômeno físico: a **Lei de Snell**, que descreve a refração da luz.

O sistema é composto por três scripts principais:
- Um para **treinar o modelo de forma incremental**.
- Um para **gerar relatórios de eficácia**.
- Um para **visualizar e interagir com a IA em tempo real**.

---

## 📂 Arquivos do Projeto

### Scripts Python

#### `treinamento_relatorio.py`
- **Função:** É o coração do projeto. Treina um modelo de rede neural para aprender a Lei de Snell.  
- **Como funciona:**  
  - A cada execução, gera 2000 novos exemplos de ângulos de incidência e índices de refração.  
  - Calcula o ângulo de refração correto e usa esses dados para treinar o modelo.  
  - O treinamento é incremental: carrega o modelo e os dados de sessões anteriores e adiciona os novos dados.  
  - Ao final, salva o modelo atualizado e registra um log com o desempenho.  

---

#### `relatorio.py`
- **Função:** Analisa o histórico de treinamento e gera um relatório sobre a evolução e eficácia do modelo.  
- **Como funciona:**  
  - Lê o arquivo `log_treinamento.csv`.  
  - Apresenta um relatório textual com as métricas de erro de cada sessão.  
  - Gera um gráfico que mostra visualmente como o erro de validação diminui a cada novo treinamento.  

---

#### `visual.py`
- **Função:** Oferece uma simulação visual e interativa para testar o modelo treinado.  
- **Como funciona:**  
  - Carrega o modelo salvo (`modelo_snell.keras`).  
  - O usuário informa:  
    - O índice de refração do segundo meio.  
    - O ângulo de incidência da luz.  
  - O script desenha três raios:  
    1. O raio incidente.  
    2. O raio refratado real (pela Lei de Snell).  
    3. O raio refratado previsto pela IA.  
  - Assim é possível comparar a previsão da IA com o resultado físico real.  

---

## 📁 Arquivos Gerados
- `modelo_snell.keras`: modelo de rede neural treinado (atualizado a cada execução de `treinamento_relatorio.py`).  
- `dados_X.npy` e `dados_y.npy`: armazenam os dados de treino (entradas e saídas esperadas).  
- `log_treinamento.csv`: log em formato CSV com timestamp, número de amostras e métricas de erro de cada sessão.  

---

## ⚙️ Pré-requisitos
Instale as dependências com:

```bash
pip install tensorflow numpy matplotlib pandas
```
## 🚀 Como Usar
### Passo 1: Treinar o Modelo
Execute o script de treinamento várias vezes.
Cada execução tornará o modelo mais preciso.

```bash
python treinamento_relatorio.py
```

Na primeira execução: cria o modelo e os arquivos de dados.
Nas execuções seguintes: carrega os arquivos existentes, adiciona novos dados e continua o treinamento.

### Passo 2: Analisar a Eficácia do Treinamento

Após algumas sessões de treino (pelo menos duas), rode o relatório:

```bash
python relatorio.py
```

Exibe uma tabela com os resultados de cada sessão.

Mostra um gráfico da queda do erro ao longo do tempo.

### Passo 3: Interagir com a Simulação Visual

Com o modelo já treinado, execute o simulador:

```bash
python visual.py
```

O programa solicitará:

- Índice de refração do segundo meio (ex: 1.5 para vidro).

- Ângulo de incidência em graus (ex: 45).

- Uma janela do Matplotlib será aberta, mostrando:

- O raio incidente.

- O raio refratado real.

- O raio previsto pela IA.

Você pode realizar várias simulações até encerrar o programa.
