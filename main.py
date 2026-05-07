from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
import argparse


def print_args(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print("************")


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    # Device setting
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ''
    cfg.DEVICE.GPU_ID = ''

    cfg.METHOD = ''
    cfg.SEED = -1

    cfg.OUTPUT = CN()
    cfg.OUTPUT.ROOT = 'experiments'

    # For dataset config
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ''
    cfg.DATASET.ROOT = ''
    cfg.DATASET.GPT_PATH = ''
    cfg.DATASET.NUM_CLASSES = -1
    cfg.DATASET.NUM_INIT_CLS = -1
    cfg.DATASET.NUM_INC_CLS = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT = -1
    cfg.DATASET.BETA = -1.0
    cfg.DATASET.ENSEMBLE_ALPHA = -1.0
    cfg.DATASET.DESCRIPTION_NOISE_RATIO = 0.0
    cfg.DATASET.DESCRIPTION_NOISE_SEED = -1
    cfg.DATASET.LOW_RES_SIZE = -1

    # For data
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    # For model
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ''

    # For methods
    cfg.TRAINER = CN()
    cfg.TRAINER.BiMC = CN()
    cfg.TRAINER.BiMC.PREC = ''
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0
    cfg.TRAINER.BiMC.GAMMA_BASE = -1.0
    cfg.TRAINER.BiMC.GAMMA_INC = -1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = False

    cfg.TRAINER.BiMCAdaptive = CN()
    cfg.TRAINER.BiMCAdaptive.ENABLE_ADAPTIVE_BETA = False
    cfg.TRAINER.BiMCAdaptive.ENABLE_ADAPTIVE_TEXT = False
    cfg.TRAINER.BiMCAdaptive.ENABLE_ADAPTIVE_VISION = False
    cfg.TRAINER.BiMCAdaptive.USE_PROMPT_ENSEMBLE = False
    cfg.TRAINER.BiMCAdaptive.PROMPT_TEMPLATES = ['a photo of a {}.']
    cfg.TRAINER.BiMCAdaptive.USE_AGREEMENT_GATE = True
    cfg.TRAINER.BiMCAdaptive.AGREEMENT_POWER = 1.0
    cfg.TRAINER.BiMCAdaptive.MIN_AGREEMENT = 0.05
    cfg.TRAINER.BiMCAdaptive.BETA_MODE = 'classwise'
    cfg.TRAINER.BiMCAdaptive.PRIOR_MODE = 'fixed'
    cfg.TRAINER.BiMCAdaptive.UNIVERSAL_BETA = 0.5
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_OBJECTIVE = 'nll'
    cfg.TRAINER.BiMCAdaptive.EPS = 1e-6
    cfg.TRAINER.BiMCAdaptive.MAX_KAPPA = 1000.0
    cfg.TRAINER.BiMCAdaptive.SINGLETON_KAPPA = 1.0
    cfg.TRAINER.BiMCAdaptive.RELIABILITY_COUNT_POWER = 0.5
    cfg.TRAINER.BiMCAdaptive.RELIABILITY_LOGIT_SCALE = 0.5
    cfg.TRAINER.BiMCAdaptive.BETA_SHRINKAGE_NU = 5.0
    cfg.TRAINER.BiMCAdaptive.TEXT_SHRINKAGE_NU = 8.0
    cfg.TRAINER.BiMCAdaptive.VISION_SHRINKAGE_NU = 5.0
    cfg.TRAINER.BiMCAdaptive.MIN_WEIGHT = 0.05
    cfg.TRAINER.BiMCAdaptive.MAX_WEIGHT = 0.95
    cfg.TRAINER.BiMCAdaptive.MIN_VISION_WEIGHT = 0.0
    cfg.TRAINER.BiMCAdaptive.MAX_VISION_WEIGHT = 0.35
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_GRID_SIZE = 31
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_REG_LAMBDA = 0.05
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_MAX_SUPPORT_PER_CLASS = 32
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_USE_CURRENT_TASK_ONLY = True
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_INCLUDE_ENSEMBLE = True
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_MARGIN_WEIGHT = 0.25
    cfg.TRAINER.BiMCAdaptive.SESSION_RISK_MARGIN_TARGET = 0.02
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_GRID_SIZE = 31
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_REG_LAMBDA = 0.02
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_CALIB_FRACTION = 0.2
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_MAX_CALIB_PER_CLASS = 32
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_MIN_PROTO_PER_CLASS = 8
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_PSEUDO_EPISODES = 8
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_PSEUDO_NOVEL_COUNT = -1
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_PSEUDO_SHOT = -1
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_PSEUDO_CALIB_PER_CLASS = 16
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_PSEUDO_MIN_BASE_PROTO = 8
    cfg.TRAINER.BiMCAdaptive.AUTO_PRIOR_PSEUDO_USE_ENSEMBLE = False

    cfg.TRAINER.BiMCOracle = CN()
    cfg.TRAINER.BiMCOracle.ENABLE = False
    cfg.TRAINER.BiMCOracle.MODE = 'none'
    cfg.TRAINER.BiMCOracle.SEARCH_METRIC = 'acc'
    cfg.TRAINER.BiMCOracle.BETA_GRID_SIZE = 19
    cfg.TRAINER.BiMCOracle.DEFAULT_DELTA = 0.0
    cfg.TRAINER.BiMCOracle.DEFAULT_TAU = 1.0
    cfg.TRAINER.BiMCOracle.DELTA_MIN = -1.5
    cfg.TRAINER.BiMCOracle.DELTA_MAX = 1.5
    cfg.TRAINER.BiMCOracle.DELTA_GRID_SIZE = 25
    cfg.TRAINER.BiMCOracle.TAU_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]



def setup_cfg(dataset_cfg_file, method_cfg_file):
    cfg = CN()
    extend_cfg(cfg)

    # 1. From the dataset config file
    cfg.merge_from_file(dataset_cfg_file)

    # 2. From the method config file
    cfg.merge_from_file(method_cfg_file)

    cfg.freeze()
    return cfg


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline")

    parser.add_argument('--data_cfg', type=str, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, help="Path to the training configuration file")

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    from engine.engine import Runner
    engine = Runner(cfg)
    engine.run()


if __name__ == '__main__':
    main()
