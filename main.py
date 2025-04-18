import torch
import random
import numpy as np
import tqdm
import train
import evaluate
import preprocess_data
import model as m

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

n_epochs = 100
clip = 1.0
teacher_forcing_ratio = 0.5
data_percentage = 1.0

best_valid_loss = float("inf")
objects, image_vocab, voxel_vocab, train_data_loader, valid_data_loader, test_data_loader = preprocess_data.preprocess_data(data_percentage)
model = m.init_model(len(image_vocab), len(voxel_vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training model with device: ", device)
optimizer = m.optimizer(model)
criterion = m.criterion(1)
evaluate_fn = evaluate.evaluate_fn

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train.train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
        torch.save(optimizer.state_dict(), "tut1-optimizer.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")