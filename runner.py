import fire
import traceback
from runners.cactus import cactus
from runners.protoclr_ae import protoclr_ae
if __name__ == "__main__":
    try:
        fire.Fire({"cactus": cactus, "protoclr_ae": protoclr_ae})
    except Exception as e:
        traceback.print_exc()


