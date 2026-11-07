#!/bin/bash
#SBATCH --job-name=sgh
#SBATCH --account=Sis26_mazzola
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=log_spec_gap_hyper_%j.out
#SBATCH --error=log_spec_gap_hyper_%j.err

# Carica i moduli necessari del cluster (es. python se richiesto dal tuo sistema)
# module load python

# Attiva l'ambiente virtuale Python richiesto
source "$HOME/monaqa_venv/bin/activate"

# Muoviti nella cartella del repository (se non lanci lo script direttamente da lì)
# cd /percorso/della/tua/copia/pubmonaqa

# Lancia il file Python
python3 launch_spectral_gap.py

