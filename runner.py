import fire
from runners.cactus import cactus
from runners.protoclr_ae import protoclr_ae


if __name__ == "__main__":
    fire.Fire({"cactus": cactus, "protoclr_ae": protoclr_ae})
