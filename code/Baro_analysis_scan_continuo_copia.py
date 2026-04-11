# Standard scientific Python stack + astrophysical tools for timing analysis
import os 
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from astropy.table import Table
from astropy.io import fits
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord

from pathlib import Path

from scipy.optimize import minimize




# Input FITS file containing photon event data (Fermi-LAT observations)
file_fits = Path("/content/drive/MyDrive/GammaRayPulsarTiming/Gamma-ray-timing-analysis-of-pulsar-J2021-3651/data/Goodtime_baro.fits") 




# Extract photon event data from FITS file
# Times are given in Mission Elapsed Time (MET) in seconds, already barycenter-corrected (TDB) and referenced to MJDREF
def getdata(file_fits):

    with fits.open(file_fits) as baro_info:
        baro_info.info()
        baro_table = Table(baro_info[1].data)

    # TIME column contains photon arrival times # Photon arrival times (MET, seconds)
    times = baro_table['TIME']

    print(f"tempo massimo: {times.max()}, tempo minimo {times.min()}")

    return times,baro_table



    
# Compute Z_n^2 Rayleigh statistic to test periodicity in photon phases
# Larger values indicate stronger deviation from uniform phase distribution
def rayleigh_Zn(phases, n_harm=8):
    
    N = len(phases)
    Z = 0.0
    
    # Harmonic k contribution (Fourier components)
    for k in range(1, n_harm+1):
        
        C = np.sum(np.cos(2*np.pi*k*phases))
        S = np.sum(np.sin(2*np.pi*k*phases))
        
        # Sum of cosine and sine terms for Rayleigh test
        Z += C**2 + S**2
        
    # Normalize Z_n^2 statistic
    Z *= 2 / N
    
    return Z





# Iterative grid search over (f0, f0_dot=spin-down) parameter space
# Goal: maximize Rayleigh Z_n^2 to detect pulsar periodicity
# Grid is progressively refined around the best solution
def iterative_scan(times, repeat, n_f=1000, n_fdot=50, range_f=500, range_fdot=25):
    
    # Total observation time span
    T = times.max()-times.min()
    
    # Reference time (midpoint of observation) to improve numerical stability of the phase model
    t0 = (times.max() + times.min())/2
    
    # Resolution scales for f0 and fdot parameters as expected from timing analysis theory
    df0 = 1/T
    dfdot = 2/T**2
    
    # Initial guess for pulsar parameters
    f0_exp = 9.63935 # frequency
    fdot_exp = -8.8892 * 10**(-12) # spin-down
    
    
    for c in range(repeat):
        print(f"--- Iterazione {c+1} ---")
        
        # Shrinking search window for iterative refinement
        range_f_current = range_f * (0.3**c)
        range_fdot_current = range_fdot * (0.3**c)

        
        # centering f,fdot grid around current estimate
        f_grid = np.linspace(f0_exp - range_f_current * df0,
                             f0_exp + range_f_current * df0,
                             n_f)
        fdot_grid = np.linspace(fdot_exp - range_fdot_current * dfdot,
                                fdot_exp + range_fdot_current * dfdot,
                                n_fdot)

    
        # Initialize Z_n^2 map over parameter space
        Z_grid = np.zeros((len(fdot_grid), len(f_grid)))
        
        
        
        for i, fdot in enumerate(fdot_grid):
            for j, f0 in enumerate(f_grid):
                
                # Compute rotational phase including spin-down correction
                phases = ( f0*(times - t0) + 0.5*fdot*(times - t0)**2 ) % 1
                # Evaluate Rayleigh statistic on grid point
                Z_grid[i,j] = rayleigh_Zn(phases, n_harm=8)
    
    
    
    
        # Find f,fdot maximizing Z_n^2 
        max_idx = np.unravel_index(np.argmax(Z_grid), Z_grid.shape)
        best_fdot = fdot_grid[max_idx[0]]
        best_f0 = f_grid[max_idx[1]]
        best_Zn = Z_grid[max_idx]
        
        # Update best estimate for next iteration
        f0_exp = best_f0
        fdot_exp = best_fdot
        
        print(f"\nIterazione {c+1}:")
        print(f"f0 = {best_f0:.8f} Hz")
        print(f"fdot = {best_fdot:.3e} Hz/s")
        print(f"Z8^2 = {best_Zn:.3f}")
       
        # Visualization of Z_n^2 map
        # Save scan result before display
        plt.figure(figsize=(12,6))
        plt.imshow(Z_grid, origin='lower',
                extent=[f_grid.min(), f_grid.max(), fdot_grid.min(), fdot_grid.max()],
                aspect='auto', cmap='viridis')
        plt.colorbar(label='Rayleigh Z8')
        plt.xlabel('Frequency f0 [Hz]')
        plt.ylabel('f_dot [Hz/s]')
        plt.title(f'Scan 2D (f, f_dot) - Iterazione {c+1}')
        plt.scatter(best_f0, best_fdot, color='red', marker='x', s=80, label='Massimo corrente')
        plt.legend()
        plt.savefig(f'scan_iter_{c+1}.png')  # salva prima di show
        plt.show()
        

    return best_f0, best_fdot




# Compute photon phases using best-fit pulsar parameters and build phosogram to check phase coherence of signal
def phasogram(times, f0, fdot):
    
    # Reference time (midpoint of observation)
    t0 = (times.max() + times.min()) / 2

    # Phase-folded representation of photon arrival times
    phases_best = (f0*(times - t0) + 0.5*fdot*(times - t0)**2) % 1

    plt.figure(figsize=(8,6))
    plt.scatter(phases_best, times, s=5, alpha=0.6)

    plt.xlabel('Fase rotazionale')
    plt.ylabel('Tempi di arrivo')
    plt.title('Fase rotazionale vs tempi di arrivo')

    plt.grid(True)
    plt.show()
    
    return phases_best
    
    

# Construct pulse profiles in different energy bands and use them to study energy dependence of pulsar emission
def histoplot(baro_table, f0, fdot):
    
    # Reference time (midpoint of observation)
    t0 = (baro_table['TIME'].max() + baro_table['TIME'].min()) / 2



    # Define low-energy (LE) and high-energy (HE) event selection
    LE_mask = (baro_table['ENERGY'] > 100) & (baro_table['ENERGY'] < 1000)
    HE_mask = baro_table['ENERGY'] > 1000
    
    times_HE = baro_table['TIME'][HE_mask]
    times_LE = baro_table['TIME'][LE_mask]
    
    
    
    # Compute rotational phases for each energy band
    phases_HE = (f0*(times_HE - t0) + 0.5*fdot*(times_HE - t0)**2) % 1
    phases_LE = (f0*(times_LE - t0) + 0.5*fdot*(times_LE - t0)**2) % 1
    
    
    
    # Number of phase bins for histogram
    N_BINS = 48
    
    
    # Pulse profile visualization for different energy ranges
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    
    for ax, phases, label, color in zip(
        [ax1, ax2],
        [phases_LE, phases_HE],
        ['100 MeV < E < 1 GeV', 'E > 1 GeV'],
        ['steelblue', 'tomato']
    ):
        ax.hist(phases, bins=N_BINS, range=(0, 1),
                color=color, alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.set_ylabel("Weighted counts", fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.text(0.02, 0.97, label,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=11, color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
    ax1.set_ylim(0, 1000)
    ax2.set_ylim(0,300)
    ax2.set_xlabel('Pulse phase', fontsize=12)
    plt.tight_layout()
    plt.show()


# Negative Rayleigh statistic for minimization algorithms
# (since optimization routines minimize by default)
def neg_Zn(params, times):
    f0, fdot = params
    t0 = (times.max() + times.min()) / 2
    
    # Compute phases for given parameter set
    phases = (f0*(times - t0) + 0.5*fdot*(times - t0)**2) % 1
    
    # Evaluate periodicity strength
    Z = rayleigh_Zn(phases, n_harm=8)
    
    return -Z






# Continuous optimization refinement of pulsar parameters 
# Use L-BFGS-B starting from grid-search result
def continuous_scan(times,best_f0,best_fdot):

    
    
    # Initial guess from grid search
    x0 = [best_f0, best_fdot]
    
    
    # Local parameter bounds around best solution
    bounds = [
        (best_f0 - 1e-6, best_f0 + 1e-6),
        (best_fdot - 1e-13, best_fdot + 1e-13)
    ]
    
    # Numerical optimization of Z_n^2
    result = minimize(
        neg_Zn,
        x0,
        args=(times,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    # Extract optimized parameters
    best_f0_opt = result.x[0]
    best_fdot_opt = result.x[1]
    best_Zn_opt = -result.fun
    
    
    print("\n=== RISULTATO OTTIMIZZAZIONE CONTINUA ===")
    print(f"f0_opt = {best_f0_opt:.10f} Hz")
    print(f"fdot_opt = {best_fdot_opt:.5e} Hz/s")
    print(f"Z8^2_opt = {best_Zn_opt:.3f}")

    return best_f0_opt, best_fdot_opt, best_Zn_opt





# Suppress compatibility warning for numpy deprecated types (np.bool)
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, 'bool'):
        np.bool = bool








if __name__ == "__main__":

    # =========================================================
    # FULL PULSAR TIMING ANALYSIS PIPELINE
    # =========================================================

    # 1. Load photon arrival times and event table from FITS file
    #    (data already barycenter-corrected and in MET seconds)
    times, baro_table = getdata(file_fits)

    # ---------------------------------------------------------
    # 2. Coarse-to-fine grid search in (f0, f_dot)
    #    Goal: maximize Rayleigh Z_n^2 statistic
    #    -> identifies candidate pulsar periodicity
    # ---------------------------------------------------------
    repeat = 2
    best_f0, best_fdot = iterative_scan(times, repeat)

    # ---------------------------------------------------------
    # 3. Phase folding of photon arrival times
    #    -> visual check of periodic signal coherence
    # ---------------------------------------------------------
    phases_best = phasogram(times, best_f0, best_fdot)

    # ---------------------------------------------------------
    # 4. Energy-dependent pulse profile analysis
    #    -> investigate how pulsar emission depends on energy
    # ---------------------------------------------------------
    histoplot(baro_table, best_f0, best_fdot)

    # ---------------------------------------------------------
    # 5. Continuous parameter refinement
    #    -> local optimization around grid-search solution
    # ---------------------------------------------------------
    best_f0_opt, best_fdot_opt, best_Zn_opt = continuous_scan(
        times,
        best_f0,
        best_fdot
    )

    # =========================================================
    # 6. Derivation of physical pulsar parameters
    # =========================================================

    F = best_f0_opt
    FDOT = best_fdot_opt

    # Spin period and its derivative
    P = 1 / F
    PDOT = -FDOT / F**2

    # Magnetic field estimate (magnetic dipole approximation)
    B = 3.2e19 * np.sqrt(P * PDOT)
    print(f"B = {B:.4e} G")

    # Characteristic age of the pulsar
    tau = P / (2 * PDOT)
    print(f"Characteristic age = {tau/(3600*24*365.25):.2e} years")

    # Spin-down luminosity (rotational energy loss rate)
    I = 1e45  # moment of inertia (g cm^2, canonical neutron star)
    Omega = 2 * np.pi / P
    Omega_dot = -2 * np.pi * PDOT / P**2

    dE = I * Omega * Omega_dot
    print(f"Spin-down luminosity dE/dt = {dE:.2e} erg/s")
    