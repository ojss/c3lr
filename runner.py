import fire
import traceback
from runners.cactus import cactus
from runners.protoclr_ae import protoclr_ae
from runners.cdfsl import cdfsl_train

if __name__ == "__main__":
    try:
        fire.Fire(
            {"cactus": cactus, "protoclr_ae": protoclr_ae, "cdfsl_train": cdfsl_train}
        )
    except Exception as e:
        traceback.print_exc()
