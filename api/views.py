from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy.random as rn
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import status
from .vasicek_outils import scrape_table, calculer_date_observation, vasicek_simulation, neg_log_likelihood
from .interpolation_outils import linear_interpolation, cubic_interpolation, spline_cubic_interpolation, exp_interpolation, exp_spline_interpolation, calibrate_curve_to_data, scrapping


@api_view(['POST'])
def vasicek_view(request):
    try:
        target_date = request.data.get("target_date")
        NR = int(request.data.get("NR"))
        MAP = int(request.data.get("MAP"))
        k = int(request.data.get("k"))

        r = scrape_table(target_date, NR, MAP)

        T = calculer_date_observation(target_date, NR)

        params_init = [0.5751942478830763,
                       5.348663798797681, 0.02639738900617135]
        result = minimize(neg_log_likelihood, params_init,
                          args=(r, T,), method='Nelder-Mead')
        alpha_est, beta_est, sigma_est = result.x

        r_simulated = vasicek_simulation(
            r[0], alpha_est, beta_est, sigma_est, 1, 10000)

        # Create and save the plot as an image
        plt.figure(figsize=(17, 10))
        for runner in range(min(10, r_simulated.shape[0])):
            plt.plot(np.arange(0, k, k / r_simulated.shape[1]), r_simulated[runner],
                     label=f'Trajectory {runner + 1}')
        plt.title('vasicek')
        plt.xlabel('Time MOIS')
        plt.ylabel('Interest Rate')

        # Save the plot as base64-encoded image in the response
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
        plt.close()

        context = {
            "success": True,
            "taux": r,
            "alpha_est": alpha_est,
            "beta_est": beta_est,
            "sigma_est": sigma_est,
            "image_base64": image_base64,
        }
        return Response(context)

    except Exception as e:
        return Response({"error": str(e)}, status=400)


@api_view(['POST'])
def interpolation_view(request):

    country = request.data.get("country")
    selections = request.data.get('selections')

    df = scrapping(country)

    maturite_interp = np.linspace(
        df['maturité(mois)'].min(), df['maturité(mois)'].max(), 1000)

    # linear_interp = linear_interpolation(df)
    # cubic_interp = cubic_interpolation(df)
    # spline_cubic_interp = spline_cubic_interpolation(df)
    # exp_interp = exp_interpolation(df)
    # exp_spline_interp = exp_spline_interpolation(df)
    # calibrate_curve = calibrate_curve_to_data(df)

    plt.plot(df['maturité(mois)'], df['taux'], 'o', label='Données brutes')

    # # Plot linear_interpolation
    # plt.plot(maturite_interp, linear_interp(
    #     maturite_interp), label='Interpolation linéaire')

    # # Plot cubic_interpolation
    # plt.plot(maturite_interp, cubic_interp(
    #     maturite_interp), label='Interpolation cubique')

    # # Plot spline_cubic_interpolation
    # plt.plot(maturite_interp, spline_cubic_interp(
    #     maturite_interp), label='Interpolation spline cubique')

    # # Plot exp_interpolation
    # plt.plot(maturite_interp, exp_interp(maturite_interp),
    #          label='Interpolation exponentielle')

    # # Plot exp_spline_interpolation
    # plt.plot(maturite_interp, exp_spline_interp(maturite_interp),
    #          label='Interpolation spline exponentielle')

    # # Plot calibrate_curve_to_data
    # plt.plot(maturite_interp, calibrate_curve(maturite_interp), label='NSS')

    interpolation_functions = {
        'Interpolation linéaire': linear_interpolation,
        'Interpolation cubique': cubic_interpolation,
        'Interpolation spline cubique': spline_cubic_interpolation,
        'Interpolation exponentielle': exp_interpolation,
        'Interpolation spline exponentielle': exp_spline_interpolation,
        'NSS': calibrate_curve_to_data
    }

    # Plot the selected interpolation functions
    for selection in selections:
        if selection in interpolation_functions:
            interpolation_function = interpolation_functions[selection]
            interpolation_result = interpolation_function(df)
            plt.plot(maturite_interp, interpolation_result(
                maturite_interp), label=selection)

    plt.xlabel('Maturité(mois)')
    plt.ylabel('Taux')
    plt.legend()
    plt.title(f'Interpolations des taux pour {country}')

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
    plt.close()

    context = {
        "success": True,
        "image_base64": image_base64,
    }
    return Response(context)
