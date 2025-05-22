"""
Basic example of using ScatterBrain.

This example demonstrates the basic functionality of the library.
"""

# Conceptual example - API is under development
import scatterbrain as sb
import scatterbrain.analysis as sba
import scatterbrain.visualization as sbv

# Load data
try:
    curve = sb.io.load_ascii_1d("path/to/your/data.dat", q_col=0, i_col=1, err_col=2)
    print(curve)

    # Perform Guinier analysis
    guinier_result = sba.guinier_fit(curve, q_range=(0.01, 0.05)) # Example q-range
    if guinier_result:
        print(f"Rg: {guinier_result['Rg']:.2f} +/- {guinier_result['Rg_err']:.2f}")
        print(f"I(0): {guinier_result['I0']:.2e} +/- {guinier_result['I0_err']:.2e}")

        # Plot Guinier fit
        sbv.plot_guinier(curve, guinier_result=guinier_result)
        sbv.plt.show() # Assuming sbv exposes matplotlib.pyplot as plt

except FileNotFoundError:
    print("Error: Data file not found.")
except Exception as e:
    print(f"An error occurred: {e}")