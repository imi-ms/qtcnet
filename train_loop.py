from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from import_functions import *


# Training step
def train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch, clip_grad=True, debug=False):
    model.train()
    train_loss = 0

    for batch_idx, (ecg_data, qtc_interval) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                    desc=f"Epoch {epoch + 1} - Training"):
        # Move data to device
        ecg_data, qtc_interval = ecg_data.to(device), qtc_interval.to(device)

        # Optional NaN/Inf checks
        if debug:
            if torch.isnan(ecg_data).any() or torch.isinf(ecg_data).any():
                print(f"NaN/Inf found in ecg_data at batch {batch_idx}")
            if torch.isnan(qtc_interval).any() or torch.isinf(qtc_interval).any():
                print(f"NaN/Inf found in rr_interval at batch {batch_idx}")

        # Forward pass
        outputs = model(ecg_data)
        loss = criterion(outputs, qtc_interval.view(-1, 1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

        # Log loss for each step
        writer.add_scalar("Loss/step_train", loss.item(), epoch * len(train_loader) + batch_idx)

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


# Validation step
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    true_values, predicted_values = [], []

    with torch.no_grad():
        for ecg_data, rr_interval in test_loader:
            ecg_data, rr_interval = ecg_data.to(device), rr_interval.to(device)

            # Forward pass
            outputs = model(ecg_data)
            loss = criterion(outputs, rr_interval.view(-1, 1))

            # Accumulate loss
            test_loss += loss.item()

            # Collect true and predicted values for plotting
            true_values.extend(rr_interval.cpu().numpy())
            predicted_values.extend(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss, np.array(true_values), np.array(predicted_values)


# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, num_epochs, device,
                **kwargs):
    best_test_result = None
    debug = kwargs.get("debug", False)
    early_stopper_patience = kwargs.get("early_stopper_patience", 5)
    early_stopper_min_delta = kwargs.get("min_delta", 0.01)
    early_stopper = EarlyStopper(patience=early_stopper_patience, min_delta=early_stopper_min_delta)
    for epoch in range(num_epochs):
        # Train one epoch
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch, debug=debug)

        # Evaluate on test data
        avg_test_loss, true_values, predicted_values = evaluate(model, test_loader, criterion, device)

        # Log metrics
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, epoch)
        if best_test_result is None:
            best_test_result = avg_test_loss
        if avg_test_loss <= best_test_result:
            best_test_result = avg_test_loss
            save_path = os.path.join(writer.log_dir, "best_model.pth.tar")
            save_checkpoint(model, optimizer, scheduler, epoch, avg_test_loss, save_path)

        # Print metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(writer.log_dir, f"checkpoint_epoch{epoch + 1}.pth.tar")
            save_checkpoint(model, optimizer, scheduler, epoch, avg_test_loss, checkpoint_path)

        # Plot and log true vs predicted values
        fig = plot_real_vs_predicted(true_values, predicted_values, epoch)
        writer.add_figure("True vs Predicted", fig, global_step=epoch)
        plt.close(fig)  # Close the figure to free memory

        # Evaluate on additional datasets provided in `kwargs`
        for name, dataloader in kwargs.items():
            if isinstance(dataloader, torch.utils.data.DataLoader):
                mae, mse, _ = check_accuracy(model, dataloader, device)

                # Log additional dataset metrics
                writer.add_scalar(f"{name}_MAE", mae, epoch)
                writer.add_scalar(f"{name}_MSE", mse, epoch)

                print(f"Epoch [{epoch + 1}/{num_epochs}] - {name} MAE: {mae:.4f}, MSE: {mse:.4f}")

        # Scheduler step
        scheduler.step(avg_test_loss)

        # Flush the writer
        writer.flush()

        if early_stopper.early_stop(avg_test_loss):
            print(f"Stop early at epoch {epoch + 1}")
            checkpoint_path = os.path.join(writer.log_dir, f"checkpoint_epoch{epoch + 1}.pth.tar")
            save_checkpoint(model, optimizer, scheduler, epoch, avg_test_loss, checkpoint_path)
            break

    print("Training Complete!")
    return avg_test_loss


# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
