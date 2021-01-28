from predict import *
from train import *
from basic_nn import *

if __name__ == '__main__':

    # GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device state:', device)

    # Settings
    epochs = 100
    batch_size = 64
    lr = 0.002
    loss_function = nn.NLLLoss()
    model = BasicNeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    # Data
    # Your own dataset
    train_loader = None  # Sth you need to prepare
    test_loader = None

    # Train
    model = train(device, model, epochs, optimizer, loss_function, train_loader)

    # Test
    predict(device, model, test_loader)

