"""This script trains a FasterRCNN object detector."""
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_setup, default_argument_parser, launch
from detectron2.config import get_cfg

import examples.object_detection_2d.detectron_dataset as detectron_dataset

MODEL_PATH = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def define_cfg(output_dir):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_dir
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    cfg.DATASETS.TRAIN = (detectron_dataset.DATASET_TRAIN,)
    cfg.DATASETS.TEST = (detectron_dataset.DATASET_VAL,)
    cfg.DATALOADER.NUM_WORKERS = 16
    # Set correct number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(detectron_dataset.OBJECT_CATEGORIES)
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_PATH)

    # Optimizer settings (NOTE: this was set for a 2-gpu run and might need changing)
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 24000
    cfg.SOLVER.STEPS = (18000,)

    # Augmentation settings
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 832, 864, 896, 928, 960, 992, 1024)
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.INPUT.MAX_SIZE_TEST = 2048

    # Run evaluation once in a while
    cfg.TEST.EVAL_PERIOD = 10000
    return cfg


def parse_args():
    parser = default_argument_parser()
    parser.add_argument("--dataset-dir", required=True, help='root path of the dataset')
    parser.add_argument("--output-dir", required=True, help="base output path")
    parser.add_argument("--run-name", required=True, help="will be used for output sub-path")
    parser.add_argument("--split", default=0, type=int)
    return parser.parse_args()


def main(args):
    print("Registering datasets")
    detectron_dataset.register_detectron(args.dataset_dir, split=args.split)

    # Setting up trainer
    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    cfg = define_cfg(output_dir=output_dir)
    cfg.freeze()
    default_setup(cfg, args)
    trainer = DefaultTrainer(cfg)

    if args.eval_only:
        print("Training evaluation")
        trainer.resume_or_load(resume=True)
        trainer.test(cfg, model=trainer.checkpointer.model)
    else:
        print("Training starting")
        trainer.resume_or_load(resume=args.resume)
        trainer.train()


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
