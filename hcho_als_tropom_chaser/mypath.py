# %%
import os
from glob import glob

chaser_base_dir = "/mnt/dg3/ngoc/CHASER_output/"
var_name = "colhchot"

hcho_chaser_inhi_paths = glob(
    os.path.join(chaser_base_dir, f"*UKpft*no_nudg/**/{var_name}"), recursive=True
)

hcho_chaser_noInhi_paths = glob(
    os.path.join(chaser_base_dir, f"*VISIT*no_nudg/**/{var_name}"), recursive=True
)

hcho_tropomi_path = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/original/var/hcho_vcd/hcho_AERmon_BIRA-TROPOMI-L3_historical_gn_20180601-20240701.nc"

# %%
