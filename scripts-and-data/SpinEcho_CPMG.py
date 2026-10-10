# Example script: spin-echo NMR simulation using a CPMG pulse train
import time

from axionbloch.dependency import np, unit, PI
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation

RCF_Freq = 1 * unit.MHz
signalFreqRot = 1 * unit.Hz
T1 = 1e6 * unit.s

# short Tdelta
Tdelta = 1.0 * unit.s
T2 = 10.0 * unit.s

# CH3CH2OH
sample = Sample(
    name="Ethanol",
    gamma=gamma_p,
    massDensity=0.78945 * unit.g / unit.cm**3,
    molarMass=46.069 * unit.g / unit.mol,
    numOfSpinsPerMolecule=6 * unit.one,
    T2=T2,
    T1=T1,
    vol=1 * unit.cm**3,
    mu=mu_p,
    temp=300 * unit.K,
    verbose=False,
)

FWHM = (1 / (np.pi * Tdelta) / RCF_Freq) * unit.one

# set detection magnet
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq - signalFreqRot) / (sample.gamma / (2 * PI)),
    FWHM=FWHM,
    nFWHM=20.0,
)
magnet.setHomogeneity(numPt=500)
print(f"numPt for magnet homogeneity = {magnet.numPt}")

excField = MagField(name="CPMG pulse train")
init_phase = 0 * unit.rad

simu = Simulation(
    name="Spin-echo CPMG simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    rate=10000 * unit.Hz,
    RCF_freq=RCF_Freq,
    duration=20 * unit.s,
    verbose=False,
)

t90 = 1/50 * unit.s
tau = 4 * Tdelta
numEcho = 2
simu.excField.setCPMGPulseTrain(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    gamma=simu.sample.gamma,
    t90=t90,
    tau=tau,
    numEcho=numEcho,
    nu_rot=signalFreqRot,
    init_phase=init_phase,
    verbose=True,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"{simu.generateTrajectories.__name__} time consumption = {toc - tic:.3g} s")

save_data = True
if save_data:
    # Save the raw field and magnetization before keepMeanStd() releases them.
    timeStamp_s = simu.timeStep.to_value(unit.s) * np.arange(simu.trjry.shape[1])
    output_path = "scripts-and-data/SpinEcho_CPMG_simu.npz"
    np.savez_compressed(
        output_path,
        timeStamp_s=timeStamp_s,
        B_vec=simu.excField.B_vec.to_value(unit.T),
        trjry=simu.trjry,
        T2_s=simu.sample.T2.to_value(unit.s),
        Tdelta_s=simu.Tdelta.to_value(unit.s),
        signalFreqRot_Hz=signalFreqRot.to_value(unit.Hz),
        init_phase_rad=init_phase.to_value(unit.rad),
        tau_s=tau.to_value(unit.s),
        num_echoes=numEcho,
        t90_s=t90.to_value(unit.s),
        detuning_rad_s=(
            sample.gamma * (magnet.B_spread - magnet.B0)
        ).to_value(unit.rad / unit.s),
        homogeneity_weights=magnet.ratios,
    )
    print(f"Saved plotting data to {output_path}")

simu.keepMeanStd()
simu.displayTrjries(verbose=True)
