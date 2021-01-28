from predict import *
from train import *
from basic_nn import *
from dynamic_nn import *

if __name__ == '__main__':

    # Define target network
    target_nn = BasicNeuralNetwork

    # GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device state:', device)

    # Settings
    epochs = 100
    lr = 0.002
    loss_function = nn.NLLLoss()
    model = BasicNeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Data from your own dataset
    train_loader = None  # something you need to prepare
    test_loader = None

    # Train
    model = train(device, model, epochs, optimizer, loss_function, train_loader)

    # Test
    predictions = predict(model, test_loader)

