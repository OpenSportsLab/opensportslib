import os
import gc
import json
import torch
import logging
from tqdm import tqdm
#import wandb


def trainer(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, best_model_path, epoch_start, model_name,
    max_epochs=100, device="cuda", project_name="classification", run_name=None, config_dict=None):
    """
    Trainer for Multi-view Action Classification.
    Logs metrics and checkpoints to Weights & Biases.
    """

    # Initialize W&B
    # wandb.init(
    #     project=project_name,
    #     name=run_name if run_name else model_name,
    #     config=config_dict,
    #     reinit=True,
    #     settings=wandb.Settings(start_method="fork"),
    # )

    logging.info("=====> Training Started")
    os.makedirs(best_model_path, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(epoch_start, max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        logging.info(f"Epoch {epoch + 1}/{max_epochs}")

        ##########################
        # ---- TRAIN ----
        ##########################
        train_loss, train_acc = train_epoch(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch + 1,
            train=True,
            set_name="train",
            device=device,
        )

        ##########################
        # ---- VALIDATION ----
        ##########################
        val_loss, val_acc = train_epoch(
            dataloader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch + 1,
            train=False,
            set_name="valid",
            device=device,
        )

        ##########################
        # ---- TEST ----
        ##########################
        test_loss, test_acc = train_epoch(
            dataloader=test_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch + 1,
            train=False,
            set_name="test",
            device=device,
        )

        ##########################
        # ---- LOGGING ----
        ##########################
        scheduler.step()

        # wandb.log(
        #     {
        #         "epoch": epoch + 1,
        #         "train/loss": train_loss,
        #         "train/acc": train_acc,
        #         "val/loss": val_loss,
        #         "val/acc": val_acc,
        #         "test/loss": test_loss,
        #         "test/acc": test_acc,
        #         "lr": scheduler.get_last_lr()[0],
        #     }
        # )

        print(f"[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"[Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"[Test]  Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

        ##########################
        # ---- SAVE CHECKPOINT ----
        ##########################
        checkpoint_name = None

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_name = f"{model_name}_best.pth.tar"
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                os.path.join(best_model_path, checkpoint_name),
            )
            #wandb.run.summary["best_val_acc"] = best_val_acc
            print(f"New best model saved with val acc = {best_val_acc:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_name = f"{model_name}_epoch_{epoch+1}.pth.tar"
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                os.path.join(best_model_path, checkpoint_name),
            )

        #if checkpoint_name:
        #    wandb.save(os.path.join(best_model_path, checkpoint_name))

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    logging.info("=====> Training Finished")
    #wandb.finish()


def train_epoch(dataloader, model, criterion, optimizer, epoch, train=True, set_name="train", device="cuda"):
    """
    Runs one epoch (train or eval) and logs predictions.
    """

    model.train() if train else model.eval()
    loss_total = 0.0
    correct = 0
    total = 0

    predictions = {"Set": set_name, "Actions": {}}
    pbar = tqdm(dataloader, desc=f"{'Training' if train else 'Evaluating'} ({set_name})", leave=False)

    for batch_idx, (mvclips, labels, clip_ids) in enumerate(pbar):
        mvclips = mvclips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        #print(labels)
        with torch.set_grad_enabled(train):
            #print(mvclips.shape)
            outputs, _ = model(mvclips)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_total += loss.item() * mvclips.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        acc_batch = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc_batch:.2f}%"})

        # Store predictions for later JSON output
        for i, clip_id in enumerate(clip_ids):
            predictions["Actions"][clip_id] = {"Predicted Class": int(predicted[i].item())}

    # Save JSON predictions
    os.makedirs("predictions", exist_ok=True)
    pred_path = os.path.join("predictions", f"pred_{set_name}_epoch_{epoch}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=4)

    gc.collect()
    torch.cuda.empty_cache()

    avg_loss = loss_total / total
    acc = correct / total
    return avg_loss, acc


def save_checkpoint(epoch, model, optimizer, scheduler, path):
    """
    Save model checkpoint.
    """
    state = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, path)
