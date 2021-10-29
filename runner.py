import fire
import traceback
from runners.cactus import cactus
from runners.protoclr_ae import protoclr_ae
from pympler import muppy, summary

if __name__ == "__main__":
    try:
        fire.Fire({"cactus": cactus, "protoclr_ae": protoclr_ae})
    except Exception as e:
        traceback.print_exc()
    finally:
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)
        pass


