import logging
from pathlib import Path

import torch as t
import yaml

import process
import quan
import util
from model import create_model, KDModel


def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / "config.yaml")

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, script_dir / "logging.conf")
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == "cpu" or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error(
                    "GPU device ID {0} requested, but only {1} devices available".format(
                        dev_id, available_gpu
                    )
                )
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info(
        "Dataset `%s` size:" % args.dataloader.dataset
        + "\n          Training Set = %d (%d)"
        % (len(train_loader.sampler), len(train_loader))
        + "\n        Validation Set = %d (%d)"
        % (len(val_loader.sampler), len(val_loader))
        + "\n              Test Set = %d (%d)"
        % (len(test_loader.sampler), len(test_loader))
    )

    # Create the model
    model = create_model(args)

    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    tbmonitor.writer.add_graph(
        model, input_to_model=train_loader.dataset[0][0].unsqueeze(0)
    )
    logger.info("Inserted quantizers into the original model")

    if args.kd.enable:
        teacher_model = create_model(args.kd)
        student_model = model
        model = KDModel(teacher_model, student_model, args.kd.scheme)

    start_epoch = 0
    if args.resume.path and (not args.kd.enable or args.kd.resume_training):
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, lean=args.resume.lean
        )
    elif args.kd.enable and not args.kd.resume_training:
        if args.resume.path:
            student_model, _, _ = util.load_checkpoint(
                student_model, args.resume_path, lean=True
            )
        if args.kd.teacher_checkpoint:
            teacher_model, _, _ = util.load_checkpoint(
                teacher_model, args.kd.teacher_checkpoint, lean=True
            )
        model = KDModel(teacher_model, model, args.kd.scheme)

    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)
        dataparallel = True
    else:
        dataparallel = False

    model.to(args.device.type)

    # Define loss function (criterion) and optimizer
    if args.kd.enable:
        criterion = util.MishraDistiller(
            args.kd.alpha, args.kd.beta, args.kd.gamma, args.kd.temperature
        ).to(args.device.type)
    else:
        criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    optimizer = t.optim.SGD(
        model.parameters(),
        lr=args.optimizer.learning_rate,
        momentum=args.optimizer.momentum,
        weight_decay=args.optimizer.weight_decay,
    )
    lr_scheduler = util.lr_scheduler(
        optimizer,
        batch_size=train_loader.batch_size,
        num_samples=len(train_loader.sampler),
        **args.lr_scheduler,
    )
    logger.info(("Optimizer: %s" % optimizer).replace("\n", "\n" + " " * 11))
    logger.info("LR scheduler: %s\n" % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)

    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info(">>>>>>>> Epoch -1 (pre-trained model evaluation)")
            top1, top5, _ = process.validate(
                val_loader, model, criterion, start_epoch - 1, monitors, args
            )
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info(">>>>>>>> Epoch %3d" % epoch)
            t_top1, t_top5, t_loss = process.train(
                train_loader,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                epoch,
                monitors,
                args,
            )
            v_top1, v_top5, v_loss = process.validate(
                val_loader, model, criterion, epoch, monitors, args
            )

            tbmonitor.writer.add_scalars(
                "Train_vs_Validation/Loss", {"train": t_loss, "val": v_loss}, epoch
            )
            tbmonitor.writer.add_scalars(
                "Train_vs_Validation/Top1", {"train": t_top1, "val": v_top1}, epoch
            )
            tbmonitor.writer.add_scalars(
                "Train_vs_Validation/Top5", {"train": t_top5, "val": v_top5}, epoch
            )

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(
                epoch,
                args.arch,
                model,
                {"top1": v_top1, "top5": v_top5},
                is_best,
                args.name,
                log_dir,
                dataparallel,
            )

        logger.info(">>>>>>>> Epoch -1 (final model evaluation)")
        process.validate(test_loader, model, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info("Program completed successfully ... exiting ...")
    logger.info(
        "If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net"
    )


if __name__ == "__main__":
    main()
