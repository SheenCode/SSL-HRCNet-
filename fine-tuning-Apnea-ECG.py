import os
import torch
import torch.nn as nn


def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_num = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 3:
                data, target, _ = batch_data
            else:
                data, target = batch_data

            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            loss = criterion(pred, target)

            total_loss += loss.item() * data.size(0)
            total_num += data.size(0)

            prediction = torch.argmax(pred, dim=1)
            total_correct += (prediction == target).sum().item()

    avg_loss = total_loss / total_num
    acc = 100.0 * total_correct / total_num
    return avg_loss, acc


def train_finetune(
    model,
    train_dataloader,
    val_dataloader,
    device,
    save_dir="trainstage2",
    pretrained_path=None,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-6,
):
    """
    Fine-tuning stage for apnea classification.
    """
    model = model.to(device)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from: {pretrained_path}")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_num = 0
        total_correct = 0

        for batch_idx, batch_data in enumerate(train_dataloader):
            if len(batch_data) == 3:
                data, target, _ = batch_data
            else:
                data, target = batch_data

            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_num += batch_size

            prediction = torch.argmax(pred, dim=1)
            total_correct += (prediction == target).sum().item()

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_dataloader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        train_loss = total_loss / total_num
        train_acc = 100.0 * total_correct / total_num

        val_loss, val_acc = evaluate(model, val_dataloader, device)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.3f}% | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.3f}%"
        )

        with open(os.path.join(save_dir, "stage2_log.txt"), "a") as f:
            f.write(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.6f}, "
                f"train_acc={train_acc:.3f}, "
                f"val_loss={val_loss:.6f}, "
                f"val_acc={val_acc:.3f}\n"
            )

        # save every epoch
        ckpt_path = os.path.join(save_dir, f"fine_tuned_epoch{epoch}_ACC{val_acc:.3f}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model updated: {best_path} (ACC={best_acc:.3f}%)")
