import sys
import timeit


# Train, to be updated
def train(device, model, epochs, optimizer, loss_function, train_loader):
    """ Train the model with the given parameters

    :param device: Where the model and data should be loaded (gpu or cpu).
    :param model: The PyTorch model that we wish to train.
    :param epochs: The total number of epochs to train for.
    :param optimizer: The optimizer to use during training.
    :param loss_function: The loss function used for training.
    :param train_loader: The PyTorch DataLoader that should be used during training.
    """
    loss = None
    timer = timeit.default_timer
    start_all = timer()
    for epoch in range(1, epochs+1):
        start_epoch = timer()
        total_loss = 0
        for times, data in enumerate(train_loader, 1):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            outputs = model(inputs.view(inputs.shape[0], -1))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Show progress
            if times % 5 == 0 or times == len(train_loader):
                pct_ind = int((1 + times) * 50 / len(train_loader))
                sys.stdout.write('\r')
                sys.stdout.write(f'{epoch}/{epochs} | '
                                 f'[{"=" * pct_ind}{"-" * (50 - pct_ind)}] {pct_ind * 2:>3}% | '
                                 f'{times}/{len(train_loader)}')
        print(f'\nDone {epoch:>5} / {epochs:>4} | loss = {total_loss / len(train_loader):.3f} | '
              f'used {timer() - start_epoch:.4}s')
    print(f'Finished training; used {timer() - start_all}s')
    return
