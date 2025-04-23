import math
import sys
from typing import Iterable

import torch
import util.lr_sched as lr_sched
import util.misc as misc
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    # import ipdb; ipdb.set_trace()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.MGT:
            # import ipdb; ipdb.set_trace()
            bs = examples.shape[0]
            
            with open('/home/qsyang2/datasets/llm-sft/KC/category.txt', 'r') as f:
                categories = [line.strip() for line in f.readlines()]
            text_prompts = [f"What does {categories[label]} look like?" for label in labels]
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
            predefined_text_prompt = clip_model.get_text_features(**inputs)
            image_embeddings = clip_model.get_image_features(examples)
            
            if examples.shape[-1] % 16 == 0 and examples.shape[-2] % 16 == 0:
                num_patches = (examples.shape[-1]//16) * (examples.shape[-2]//16)
            image_embeddings = image_embeddings.view(batch_size, -1, num_patches)
            image_embeddings = image_embeddings.permute(0, 2, 1)
            
            if args.CKI:
                # import ipdb; ipdb.set_trace()
                cosine_similarities = F.cosine_similarity(image_embeddings, predefined_text_prompt.unsqueeze(1), dim=-1)
                weighted_image_embeddings = image_embeddings * cosine_similarities.unsqueeze(-1)
                visual_prompt = weighted_image_embeddings.permute(0, 2, 1).contiguous().view(batch_size, -1, image_height, image_width)
            else:
                visual_prompt = image_embeddings.contiguous().view(batch_size, -1, image_height, image_width)

        # import ipdb; ipdb.set_trace()
        c_loss = model(examples, labels, visual_prompt=None, predefined_text_prompt=1, image_embeddings=image_embeddings)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            if args.MGT:
                bs = examples.shape[0]

                with open('/home/qsyang2/datasets/llm-sft/KC/category.txt', 'r') as f:
                    categories = [line.strip() for line in f.readlines()]
                
                text_prompts = [f"What does {categories[label]} look like?" for label in labels]
                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
                predefined_text_prompt = clip_model.get_text_features(**inputs)
                image_embeddings = clip_model.get_image_features(examples)
                
                if examples.shape[-1] % 16 == 0 and examples.shape[-2] % 16 == 0:
                    num_patches = (examples.shape[-1]//16) * (examples.shape[-2]//16)
                image_embeddings = image_embeddings.view(batch_size, -1, num_patches)
                image_embeddings = image_embeddings.permute(0, 2, 1)
                
                if args.CKI:
                    # import ipdb; ipdb.set_trace()
                    cosine_similarities = F.cosine_similarity(image_embeddings, predefined_text_prompt.unsqueeze(1), dim=-1)
                    weighted_image_embeddings = image_embeddings * cosine_similarities.unsqueeze(-1)
                    visual_prompt = weighted_image_embeddings.permute(0, 2, 1).contiguous().view(batch_size, -1, image_height, image_width)
                else:
                    visual_prompt = image_embeddings.contiguous().view(batch_size, -1, image_height, image_width)
            # import ipdb; ipdb.set_trace()
            c_loss, output = model(examples, labels, visual_prompt=None, predefined_text_prompt=1, image_embeddings=image_embeddings, val=True)
            # c_loss, output = model(examples, labels, val=True)
        
        loss = c_loss
        loss_value = loss.item()

        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
