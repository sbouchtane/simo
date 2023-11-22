from bs4 import BeautifulSoup

import numpy as np
import requests
import numpy.random as rn
import pandas as pd
import holidays


def calculer_date_observation(date_sortie, jours_ouvrables):

    date_sortie = pd.to_datetime(date_sortie, format="%m/%d/%Y")

    jours_feries = holidays.UnitedStates(years=date_sortie.year)

    offset = pd.offsets.CustomBusinessDay(
        weekmask='Mon Tue Wed Thu Fri', holidays=list(jours_feries))

    date_premiere_observation = date_sortie
    for _ in range(jours_ouvrables):
        date_premiere_observation -= offset
        while date_premiere_observation.weekday() in {5, 6} or date_premiere_observation in jours_feries:
            date_premiere_observation -= offset

    jours_total_entre_dates = (
        date_sortie - date_premiere_observation).days + 1

    return jours_total_entre_dates


def scrape_table(target_date, num_rows=30, MAP=5):
    try:
        bkm = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value="
        target_date1 = target_date[6:]
        url = bkm + target_date1
        page = requests.get(url)

        if page.status_code == 200:
            content = page.content
            soup = BeautifulSoup(content, 'html.parser')

            table = soup.find(
                'table', class_='usa-table views-table views-view-table cols-23')

            if table:

                target_row = None
                rows = table.find_all('tr')

                for row in rows:
                    cells = row.find_all('td')
                    if cells and cells[0].text.strip() == target_date:
                        target_row = row
                        break

                if target_row:

                    cells = target_row.find_all('td')
                    first_char_target_row = cells[1].text.strip()[1]
                    last_13_cells = [float(cell.text.strip())
                                     for cell in cells[-13:]]

                    data = {'1': [first_char_target_row] + last_13_cells}
                    for i in range(1, num_rows):
                        prev_row = rows[rows.index(target_row) - i]
                        prev_cells = prev_row.find_all('td')
                        first_char_prev_row = prev_cells[0].text.strip()[0]
                        last_13_prev_cells = [
                            float(cell.text.strip()) for cell in prev_cells[-13:]]
                        data[str(i+1)] = [first_char_prev_row] + \
                            last_13_prev_cells

                    df = pd.DataFrame(data)
                    last_row = df.iloc[-MAP]
                    result_list = last_row.values.flatten().tolist()
                    result_list.reverse()

                    return result_list

                else:
                    raise ValueError(
                        f"La ligne pour la date {target_date} n'a pas été trouvée dans le tableau.")
            else:
                raise ValueError("Le tableau n'a pas été trouvé sur la page.")
        else:
            raise requests.exceptions.RequestException(
                f"La requête vers {url} a échoué avec le code de statut {page.status_code}")
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")


def vasicek_simulation(r0, alpha, beta, sigma, T, N):
    dt = T / N
    M = 5000
    dz = rn.randn(M, N)
    r = r0 * np.ones((M, N+1))
    for i in range(0, N):
        r[:, i+1] = r[:, i] + alpha * \
            (beta - r[:, i]) * dt + sigma * dz[:, i] * np.sqrt(dt)
    return r


# def vasicek(r0, alpha, beta, sigma, T, N, r):
#     dt = T/N
#     r = np.zeros(N+1)
#     r[0] = r0
#     for i in range(1, N+1):
#         e = r[i - 1]
#         r[i] = r[i-1] + alpha*(beta-r[i-1])*dt + sigma*np.sqrt(dt)*e
#     return r


def neg_log_likelihood(params, r, T):
    alpha, beta, sigma = params
    N = len(r) - 1
    r_sim = vasicek_simulation(r[0], alpha, beta, sigma, T, N)
    # sigma2 = np.var(r[1:] - r_sim[1:])
    log_likelihood = -np.sum(
        np.log(np.exp(-(r[1:] - r_sim[:, 1:]) ** 2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)))
    return log_likelihood
