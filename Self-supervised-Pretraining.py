import os
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, out_1, out_2):
        """
        Args:
            out_1: [B, D]
            out_2: [B, D]

        Returns:
            scalar loss
        """
        batch_size = out_1.size(0)
        temperature = self.temperature

        # [2B, D]
        out = torch.cat([out_1, out_2], dim=0)

        # similarity matrix: [2B, 2B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

        # remove self-similarity
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)  # [2B, 2B-1]

        # positive similarity
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)  # [B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2B]

        loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))
        return loss.mean()


def pre_train(
    model,
    data_loader,
    device,
    save_dir="trainstage1",
    epochs=50,
    lr=1e-3,
    weight_decay=1e-6,
    save_every=2,
):
    """
    Pre-train the SimCLR Stage-1 model.

    Args:
        model: SimCLRStage1 model
        data_loader: PyTorch DataLoader
        device: torch.device
        save_dir: directory to save checkpoints and loss log
        epochs: number of training epochs
        lr: learning rate
        weight_decay: optimizer weight decay
        save_every: save checkpoint every N epochs
    """
    model = model.to(device)
    criterion = ContrastiveLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch_data in enumerate(data_loader):
            # Compatible with dataset returning (data, label) or (data, label, domain)
            if len(batch_data) == 3:
                data, label, _ = batch_data
            else:
                data, label = batch_data

            data = data.to(device)
            label = label.to(device)

            # model output
            pre_L, pre_R, L, R = model(data)

            loss = criterion(L, R)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(data_loader)}] Loss: {loss.item():.6f}")

        epoch_loss = total_loss / total_samples
        print(f"Epoch [{epoch}/{epochs}] Average Loss: {epoch_loss:.6f}")

        with open(os.path.join(save_dir, "stage1_loss.txt"), "a") as f:
            f.write(f"{epoch_loss:.6f}\n")

        if epoch % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"model_stage1_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
