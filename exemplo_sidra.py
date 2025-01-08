import pandas as pd
import sidrapy as sidra
from typing import Dict, Optional
from openpyxl import Workbook

def sidra_series_adapt(df_sidra: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Processa e formata dados brutos do SIDRA
    
    Args:
        df_sidra: DataFrame com dados brutos do SIDRA
        column_name: Nome da coluna para os valores processados
        
    Returns:
        DataFrame formatado com índice temporal
    """
    # Remove linhas de cabeçalho
    df_sidra = df_sidra.iloc[2:].reset_index(drop=True)
    
    # Renomeia colunas
    df_sidra.rename(columns={
        'D2C': 'Mês (Código)',
        'V': 'Valor'
    }, inplace=True)
    
    # Seleciona colunas relevantes
    df_sidra = df_sidra[['Mês (Código)', 'Valor']]
    
    # Converte para datetime
    df_sidra['Date'] = pd.to_datetime(df_sidra['Mês (Código)'], format='%Y%m')
    df_sidra.drop(columns=['Mês (Código)'], inplace=True)
    
    # Define índice temporal
    df_sidra.set_index('Date', inplace=True)
    df_sidra.index = df_sidra.index.to_period('M')
    
    # Converte valores para numérico
    df_sidra['Valor'] = pd.to_numeric(df_sidra['Valor'], errors='coerce')
    df_sidra.rename(columns={'Valor': column_name}, inplace=True)
    
    return df_sidra

def get_sidra_series(column_name: str, table_code: str, variable: str, 
                    classifications: Optional[Dict] = None) -> pd.DataFrame:
    """
    Extrai dados de uma tabela específica do SIDRA
    
    Args:
        column_name: Nome da coluna no DataFrame final
        table_code: Código da tabela SIDRA
        variable: Código da variável
        classifications: Dicionário com classificações (opcional)
        
    Returns:
        DataFrame com os dados processados
    """
    if classifications:
        df_sidra_brute = sidra.get_table(
            table_code=table_code,
            territorial_level="1",
            ibge_territorial_code="all",
            period="all",
            variable=variable,
            classifications=classifications
        )
    else:
        df_sidra_brute = sidra.get_table(
            table_code=table_code,
            territorial_level="1",
            ibge_territorial_code="all",
            period="all",
            variable=variable
        )
    
    return sidra_series_adapt(df_sidra_brute, column_name)

def get_sidra_data(sidra_list: list) -> pd.DataFrame:
    """
    Extrai e consolida múltiplas séries do SIDRA
    
    Args:
        sidra_list: Lista de dicionários com configurações das séries
        
    Returns:
        DataFrame consolidado com todas as séries
    """
    from functools import reduce
    
    df_list = []
    
    for item in sidra_list:
        df_item = get_sidra_series(
            column_name=item['column_name'],
            table_code=item['table_code'],
            variable=item['variable'],
            classifications=item['classifications']
        )
        df_list.append(df_item)
    
    # Combina todas as séries
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'),
        df_list
    )
    
    # Filtra dados a partir de 2003
    df_merged = df_merged[df_merged.index >= '2003-01-01']
    
    return df_merged

if __name__ == "__main__":
    # Exemplo de uso
    sidra_list = [
        {
            'column_name': 'IBGE: Indicador de Produção - Indústria de Transformação',
            'table_code': '8888',
            'variable': '12606',
            'classifications': {'544': '129316'}
        },
        {
            'column_name': 'IBGE: Indicador de Produção - Indústria Extrativa',
            'table_code': '8888',
            'variable': '12606',
            'classifications': {'544': '129315'}
        },
        {
            'column_name': 'IBGE: Indicador de Produção - Bens de Capital',
            'table_code': '8887',
            'variable': '12606',
            'classifications': {'543': '129278'}
        },
        {
            'column_name': 'IBGE: Indicador de Produção - Bens Intermediários',
            'table_code': '8887',
            'variable': '12606',
            'classifications': {'543': '129283'}
        },
        {
            'column_name': 'IBGE: Indicador de Produção - Bens de Consumo Duráveis',
            'table_code': '8887',
            'variable': '12606',
            'classifications': {'543': '129301'}
        },
        {
            'column_name': 'IBGE: Indicador de Produção - Bens de Consumo Semi-Duráveis',
            'table_code': '8887',
            'variable': '12606',
            'classifications': {'543': '129306'}
        },
        {
            'column_name': 'IBGE: Índice de Receita Nominal de Vendas no Comércio Varejista',
            'table_code': '8880',
            'variable': '7169',
            'classifications': {'11046': '56733'}
        },
        {
            'column_name': 'IBGE: Índice Nacional de Preços ao Consumidor Amplo (IPCA)',
            'table_code': '1737',
            'variable': '2266',
            'classifications': None
        },
        {
            'column_name': 'IBGE: Índice de Receita Nominal de Vendas de Material de Construção',
            'table_code': '8757',
            'variable': '7169',
            'classifications': {'11046': '56731'}
        }
    ]
    
    # Extrai e salva os dados
    data = get_sidra_data(sidra_list)
    # Salva em Excel
    with pd.ExcelWriter('sidra_data.xlsx') as writer:
        data.to_excel(writer, sheet_name='Dados')
        print("Dados extraídos e salvos em sidra_data.xlsx")
