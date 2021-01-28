import torch


# to be modified
def predict(device, model, test_loader):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs.view(inputs.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accurecy:', correct / total)
    return

