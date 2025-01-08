import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, shapiro


def calc_monthly_returns(df, ret_type='simple'):
    """Calcula retornos mensais de um DataFrame com dados financeiros"""
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    # Agrupa por mês/ano e pega o último valor ajustado
    grouped = df.groupby(['year', 'month']).agg({
        'Date': 'last',
        'Close': 'last'
    }).reset_index()

    # Calcula retornos
    if ret_type == 'log':
        grouped['monthly_returns'] = np.log(grouped['Close'] /
                                            grouped['Close'].shift(1))
    else:
        grouped['monthly_returns'] = (grouped['Close'] /
                                      grouped['Close'].shift(1)) - 1

    return grouped.dropna(subset=['monthly_returns'])


def compare_with_risk_normality(ticker_fund,
                                ticker_index,
                                start_date='2014-01-01',
                                end_date='2023-12-31',
                                ret_type='simple'):
    """Compara retornos mensais de um fundo e um índice"""
    # Obtém dados do Yahoo Finance
    df_fund = yf.download(ticker_fund, start=start_date, end=end_date)
    df_index = yf.download(ticker_index, start=start_date, end=end_date)

    # Remove MultiIndex e mantém apenas o nível inferior das colunas
    if isinstance(df_fund.columns, pd.MultiIndex):
        df_fund.columns = df_fund.columns.droplevel(1)
    if isinstance(df_index.columns, pd.MultiIndex):
        df_index.columns = df_index.columns.droplevel(1)

    # Calcula retornos mensais
    ret_fund = calc_monthly_returns(df_fund, ret_type)
    ret_index = calc_monthly_returns(df_index, ret_type)

    # Merge dos retornos para teste t pareado
    merged = pd.merge(ret_fund,
                      ret_index,
                      on=['year', 'month'],
                      suffixes=('_fund', '_index'))

    # Calcula as diferenças dos retornos
    merged['differences'] = merged['monthly_returns_fund'] - merged[
        'monthly_returns_index']

    # Teste de normalidade (Shapiro-Wilk) nas diferenças
    stat_shapiro, p_shapiro = shapiro(merged['differences'])

    # Teste t pareado
    t_result = ttest_rel(merged['monthly_returns_fund'],
                         merged['monthly_returns_index'])
    mean_fund = merged['monthly_returns_fund'].mean()
    mean_index = merged['monthly_returns_index'].mean()

    print(
        "========== COMPARAÇÃO: {} (Fundo) vs. {} (Índice) ==========".format(
            ticker_fund, ticker_index))
    print(f"Período: {start_date} até {end_date}")
    print(f"Frequência: Retornos Mensais | Retorno: {ret_type}")
    print(
        "---------------------------------------------------------------------------------"
    )
    print("RETORNOS MÉDIOS MENSAL:")
    print(f"  Fundo: {mean_fund:.4f}")
    print(f"  Índice: {mean_index:.4f}")
    print(
        "---------------------------------------------------------------------------------"
    )
    print("TESTE DE NORMALIDADE NAS DIFERENÇAS (Shapiro-Wilk):")
    print(f"  Estatística Shapiro-Wilk: {stat_shapiro:.4f}")
    print(f"  Valor-p: {p_shapiro:.4f}")
    if p_shapiro > 0.05:
        print("  As diferenças seguem uma distribuição normal.")
    else:
        print(
            "  As diferenças não seguem uma distribuição normal. Considere um teste não paramétrico."
        )
    print(
        "---------------------------------------------------------------------------------"
    )
    print("TESTE T (duas amostras) NOS RETORNOS:")
    print(f"  Estatística t: {t_result.statistic:.4f}")
    print(f"  Valor-p (bilateral): {t_result.pvalue:.4f}")

    # Interpretação do teste t
    alpha = 0.05
    if t_result.pvalue < alpha:
        print(
            f"  Rejeitamos a hipótese nula (H0) ao nível de significância de {alpha:.2f}."
        )
        print(
            f"  Existe uma diferença estatisticamente significativa entre os retornos do fundo ({ticker_fund}) e do índice ({ticker_index})."
        )
    else:
        print(
            f"  Não rejeitamos a hipótese nula (H0) ao nível de significância de {alpha:.2f}."
        )
        print(
            f"  Não há evidência suficiente para afirmar que os retornos do fundo ({ticker_fund}) e do índice ({ticker_index}) são diferentes."
        )
    print(
        "---------------------------------------------------------------------------------\n"
    )

    return {
        'mean_fund': mean_fund,
        'mean_index': mean_index,
        't_statistic': t_result.statistic,
        'p_value': t_result.pvalue,
        'shapiro_stat': stat_shapiro,
        'shapiro_p': p_shapiro
    }


if __name__ == "__main__":
    resultado = compare_with_risk_normality("ARKK",
                                            "^GSPC",
                                            start_date="2014-01-01",
                                            end_date="2024-12-31",
                                            ret_type="simple")
