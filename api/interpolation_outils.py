import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit
from nelson_siegel_svensson.calibrate import calibrate_ns_ols


def convert_to_months(years):
    if 'month' in years:
        return int(years.split()[0])
    elif 'year' in years:
        return int(years.split()[0]) * 12
    else:
        return None


def convert_to_int(rate):
    return float(rate.strip('%'))


def scrapping(country):
    url = f'https://www.worldgovernmentbonds.com/country/{country}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'w3-table money pd22 -f14'})

    table_data = []

    for row in table.find_all('tr'):
        cells = row.find_all('td')

        if len(cells) >= 2:
            maturity = convert_to_months(cells[1].text.strip())
            rate = convert_to_int(cells[2].text.strip())
            row_data = [maturity, rate]
            table_data.append(row_data)

    df = pd.DataFrame(table_data, columns=['maturité(mois)', 'taux'])
    return df


def linear_interpolation(df):
    linear_interp = interp1d(
        df['maturité(mois)'], df['taux'], kind='linear', fill_value='extrapolate')
    return linear_interp


def cubic_interpolation(df):
    df = df.sort_values(by='maturité(mois)')
    cubic_interp = interp1d(
        df['maturité(mois)'], df['taux'], kind='cubic', fill_value='extrapolate')
    return cubic_interp


def spline_cubic_interpolation(df):
    df = df.sort_values(by='maturité(mois)')
    spline_cubic_interp = CubicSpline(
        df['maturité(mois)'], df['taux'], extrapolate=True)
    return spline_cubic_interp


def exp_interpolation(df):
    df = df[df['maturité(mois)'] > 12]
    df = df.sort_values(by='maturité(mois)')

    exp_interp = interp1d(df['maturité(mois)'], df['taux'],
                          kind='quadratic', fill_value='extrapolate')
    return exp_interp


def calibrate_curve_to_data(df):
    t = np.array(df['maturité(mois)'])
    y = np.array(df['taux'])
    curve, status = calibrate_ns_ols(t, y, tau0=1.0)

    # if not status.success:
    #     print("Calibration did not converge successfully.")

    return curve


def exponential_function(x, a, b, c):
    return a * np.exp(b * x) + c


def exp_spline_interpolation(df):

    df_filtered = df[df['maturité(mois)'] > 12]
    df_filtered = df_filtered.sort_values(by='maturité(mois)')

    exp_spline_interp = UnivariateSpline(
        df_filtered['maturité(mois)'], df_filtered['taux'], k=3, s=0)
    return exp_spline_interp


def exp_curve_fit_interpolation(df):

    df_filtered = df[df['maturité(mois)'] > 12]
    df_filtered = df_filtered.sort_values(by='maturité(mois)')

    initial_guess = [1.0, 0.0, 1.0]

    try:
        popt, pcov = curve_fit(
            exponential_function, df_filtered['maturité(mois)'], df_filtered['taux'], p0=initial_guess)

        def exp_interp(x): return exponential_function(x, *popt)
        return exp_interp
    except Exception as e:
        # print(f"Erreur lors de l'ajustement exponentiel : {e}")
        return None
