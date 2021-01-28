import sys
import timeit


# Train
def train(device, model, epochs, optimizer, loss_function, train_loader):
    loss = None
    timer = timeit.default_timer
    start_all = timer()
    for epoch in range(1, epochs+1):
        start_epoch = timer()
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

            # Show progress
            if times % 5 == 0 or times == len(train_loader):
                pct_ind = int((1 + times) * 50 / len(train_loader))
                sys.stdout.write('\r')
                sys.stdout.write(f'{epoch}/{epochs} | '
                                 f'[{"=" * pct_ind}{"-" * (50 - pct_ind)}] {pct_ind * 2:>3}% | '
                                 f'{times}/{len(train_loader)}')
        print(f'\nDone {epoch:>5} / {epochs:>4} | loss = {loss.item():.8} | used {timer() - start_epoch:.4}s')
    print(f'Finished training; used {timer() - start_all}s')
    return model
