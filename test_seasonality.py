# Configuração do backend do matplotlib
import matplotlib
matplotlib.use('Agg')

# Filtrar warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Importações padrão
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import List, Tuple
import matplotlib.pyplot as plt
import math
import ruptures as rpt

def rolling_seasonal_strength(series: pd.Series, period: int = 12, window_size: int = 60, step: int = 12) -> List[Tuple[int, int, float]]:
    """
    Calcula a força sazonal em janelas móveis ao longo da série temporal.
    
    Parâmetros:
    - series: Série temporal para análise
    - period: Período sazonal (default: 12)
    - window_size: Tamanho da janela móvel (default: 60)
    - step: Passo entre janelas (default: 12)
    
    Retorna:
    - Lista de tuplas contendo (início, fim, força_sazonal) para cada janela
    """
    results = []
    for start in range(0, len(series) - window_size, step):
        window = series[start:start + window_size]
        stl = STL(window, period=period, robust=True).fit()
        seasonal_strength = np.var(stl.seasonal) / np.var(window)
        results.append((start, start + window_size, seasonal_strength))
    return results

def detect_breaks(strengths: List[float], pen: float = 10) -> List[int]:
    """
    Detecta pontos de ruptura na série de força sazonal usando Bai-Perron (via ruptures).
    
    Parâmetros:
    - strengths: Lista de valores de força sazonal
    - pen: Penalidade para o algoritmo (maior = menos quebras)
    
    Retorna:
    - Lista de índices onde ocorrem as quebras estruturais
    """
    # Converte para array numpy
    signal = np.array(strengths)
    
    # Aplica o algoritmo Pelt (implementação eficiente do Bai-Perron)
    # Usando modelo normal que é bom para detectar mudanças na média e variância
    algo = rpt.Pelt(model="normal").fit(signal)
    breakpoints = algo.predict(pen=pen)
    
    # Remove o último ponto (fim da série)
    return breakpoints[:-1]

def plot_all_seasonal_strengths(data: pd.DataFrame, filename: str = 'seasonal_strength.png', pen: float = 10, verbose: bool = True):
    """
    Plota a evolução da força sazonal ao longo do tempo para todas as séries em um grid n x n.
    
    Parâmetros:
    - data: DataFrame com as séries temporais
    - filename: Nome do arquivo PNG para salvar o plot
    """
    n_series = len(data.columns)
    n_rows = math.ceil(math.sqrt(n_series))
    n_cols = math.ceil(n_series / n_rows)
    
    # Cria figura principal para os gráficos
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows + 2))
    
    # Cria um grid para os subplots, deixando espaço para a legenda
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[4]*n_rows + [1])
    
    # Dicionário para mapear índices aos nomes completos das séries
    series_names = {}
    
    # Plota os gráficos
    for idx, col in enumerate(data.columns, 1):
        series = data[col].dropna()
        results = rolling_seasonal_strength(series)
        series_names[idx] = col
        
        time_ranges = [f"{series.index[start].year}" 
                      for start, _, _ in results]
        strengths = [strength for _, _, strength in results]
        
        ax = fig.add_subplot(gs[((idx-1)//n_cols), ((idx-1)%n_cols)])
        # Plota a série de força sazonal
        ax.plot(range(len(strengths)), strengths, marker='o', label='Força Sazonal')
        
        # Detecta e plota os pontos de ruptura
        breakpoints = detect_breaks(strengths, pen=pen)
        
        for bp in breakpoints:
            ax.axvline(bp, color="red", linestyle="--")
        
        ax.set_title(f'Série {idx}')
        ax.set_xticks([0, len(strengths)-1])
        ax.set_xticklabels([time_ranges[0], time_ranges[-1]], rotation=45)
        ax.set_ylabel("Força Sazonal")
        ax.grid(True)
        
        if len(breakpoints) > 0:
            ax.set_title(f'Série {idx} ({len(breakpoints)} rupturas)')
    
    # Adiciona a legenda em texto na parte inferior
    legend_text = "Legenda:\n" + "\n".join([f"Série {idx}: {name}" for idx, name in series_names.items()])
    fig.text(0.1, 0.02, legend_text, fontsize=8, va='top', ha='left', wrap=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_stl_decomposition(series: pd.Series, filename: str = 'stl_decomposition.png'):
    """
    Plota a decomposição STL de uma série temporal.
    
    Parâmetros:
    - series: Série temporal para análise
    - filename: Nome do arquivo PNG para salvar o plot
    """
    # Aplica STL
    stl = STL(series, period=12, robust=True).fit()
    
    # Cria figura com subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plota série original
    series.plot(ax=ax1)
    ax1.set_title('Série Original')
    ax1.grid(True)
    
    # Plota tendência
    stl.trend.plot(ax=ax2)
    ax2.set_title('Tendência')
    ax2.grid(True)
    
    # Plota componente sazonal
    stl.seasonal.plot(ax=ax3)
    ax3.set_title('Componente Sazonal')
    ax3.grid(True)
    
    # Plota resíduos
    stl.resid.plot(ax=ax4)
    ax4.set_title('Resíduos')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Carrega dados
    data = pd.read_excel('sidra_data.xlsx', index_col='Date', parse_dates=True)
    
    # Executa a análise de força sazonal
    plot_all_seasonal_strengths(data, pen=5.0, verbose=True)
    
    # Plota decomposição STL da primeira série
    first_series = data.iloc[:, 0].dropna()
    # Imprime o nome da série e o arquivo de saída
    print(f"Primeira série: {first_series.name}")
    plot_stl_decomposition(first_series)
