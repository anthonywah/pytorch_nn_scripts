from predict import *
from train import *
from basic_nn import *
from dynamic_nn import *
import pandas as pd
import torch.utils.data as data


def _get_data_loader(input_df, batch_size=1):
    # labels are first column
    cols = input_df.columns
    y = torch.from_numpy(input_df[cols[0]].values).float().squeeze()
    x = torch.from_numpy(input_df.drop(cols[0], axis=1).values).float()

    # create dataset
    ds = torch.utils.data.TensorDataset(x, y)

    # create DataLoader
    out_dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    print('Got data loader from input_df')
    return out_dl


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
    model = target_nn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Data from your own dataset
    # Sample:
    train_loader = _get_data_loader(pd.DataFrame([[1, 2], [1, 2], [1, 2]], columns=['y', 'x1', 'x2']))
    test_loader = None  # similar steps

    # Train
    model = train(device, model, epochs, optimizer, loss_function, train_loader)

    # Test
    predictions = predict(model, test_loader)

