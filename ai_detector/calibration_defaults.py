"""Paramètres de calibration par défaut (repli si ``model/calibration.json`` est absent).

Ces valeurs sont **conservatrices** : elles encodent uniquement le sens
physique attendu de chaque indice (poids positif = évidence « synthétique »,
poids négatif = évidence « capteur réel ») avec des amplitudes modérées, et des
statistiques de normalisation approximatives pour des photographies.

Elles sont destinées à être remplacées par ``training/calibrate_fusion.py``,
qui ajuste moyennes, écarts-types, poids et biais sur un jeu de validation
étiqueté et écrit ``model/calibration.json``.
"""

DEFAULT_CALIBRATION: dict = {
    "schema_version": 1,
    "version": "defaults-0.1.0",
    "source": "defaults",
    "frequency": {
        "features": [
            "slope",
            "hf_residual",
            "hf_residual_abs",
            "hf_energy_ratio",
            "peak_max_z",
            "peak_density",
            "grid_peak_z",
            "anisotropy",
            "profile_roughness",
        ],
        "mean": [-2.8, -0.8, 1.0, 0.05, 3.9, 0.0, 2.2, 0.4, 0.14],
        "std": [0.45, 0.9, 0.7, 0.035, 1.0, 0.001, 0.5, 0.8, 0.05],
        "weights": [0.0, 0.0, 0.1, 0.1, 0.4, 0.2, 0.9, 0.15, 0.3],
        "bias": -1.2,
    },
    "noise": {
        "features": [
            "flat_noise_std",
            "shot_noise_corr",
            "flat_kurtosis",
            "residual_ac_peak",
            "residual_spec_peak_z",
            "residual_grid_peak_z",
            "cfa_strength",
            "jpeg_grid_strength",
            "channel_residual_corr",
            "flat_block_fraction",
        ],
        "mean": [0.005, 0.0, 3.0, 0.02, 3.6, 2.2, -0.5, 0.10, 0.85, 0.35],
        "std": [0.003, 0.45, 1.0, 0.03, 0.7, 0.45, 1.0, 0.20, 0.10, 0.15],
        "weights": [-0.2, -0.4, 0.2, 0.3, 0.4, 0.9, -0.2, 0.0, 0.0, 0.0],
        "bias": -1.0,
    },
    "fusion": {
        "threshold": 0.5,
        # Avec un CNN entraîné : le réseau porte l'essentiel de l'évidence, les
        # indices physiques servent de garde-fous et d'explication.
        "full": {"weights": {"fft": 0.5, "noise": 0.5, "cnn": 1.5}, "bias": 0.0},
        # Sans CNN (modèle absent ou non entraîné) : indices physiques seuls.
        "handcrafted": {"weights": {"fft": 0.8, "noise": 0.8}, "bias": 0.0},
    },
    "metadata": {
        "note": (
            "Valeurs par défaut : sens physique des indices + statistiques de normalisation "
            "observées sur un petit jeu de contrôle (55 photos FFHQ vs 115 images de diffusion latente, "
            "Community Forensics-Small). Non représentatif : lancer training/calibrate_fusion.py "
            "sur un jeu de validation large (GenImage / Community Forensics / AIGenImages2026)."
        ),
    },
}
