import torch


# to be modified
def predict(model, input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Put model into evaluation mode
    model.eval()

    # Convert input_data to numpy array then to Tensor
    data = torch.from_numpy(input_data)
    data = data.to(device)

    # Predict
    out = model(data)
    result = out.cpu().detach().numpy()

    print('Prediction finished')
    return result

