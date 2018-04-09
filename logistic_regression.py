import torch
from torch.autograd import Variable, Function
import numpy as np

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 2)

    def forward(self, inputs):
        out = self.linear(inputs)
        return out

# class BinaryLoss(torch.nn.Module):
#     def __init__(self):
#         super(BinaryLoss, self).__init__()

#     def forward(self, input, target):
#         loss = - torch.mean(target * torch.log(torch.clamp(input, 1e-10, 1.)) + (1 - target) * torch.log(torch.clamp(1 - input, 1e-10, 1.))) 
#         return loss

def get_eclipse_data(path="data/eclipse-data.npz"):

    """get_eclipse_data
    Load the data into 4 numpy arrays: train_x, train_y, test_x, test_y and return them
    :param path: path to the eclipse dataset file
    """
    f = np.load(path)
    train_x = f['train_x']
    train_y = f['train_y']
    test_x = f['test_x']
    test_y = f['test_y']
    return (train_x, train_y, test_x, test_y)

if __name__ == "__main__":
    torch.manual_seed(2018)

    train_x, train_y, test_x, test_y = get_eclipse_data()
    num_train, input_dim = train_x.shape
    num_test = test_x.shape[0]

    input_tensor = torch.from_numpy(train_x).float()
    target_tensor = torch.from_numpy(train_y).long().view(-1)
    test_data = Variable(torch.from_numpy(test_x).float())

    logistic_model = LogisticRegression(input_dim)

    learning_rate = 0.01
    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    # criterion = torch.nn.NLLLoss()
    # criterion = BinaryLoss()

    num_epoch = 10000
    print_every = 1

    for epoch in range(num_epoch):
        input_var = Variable(input_tensor)
        target_var = Variable(target_tensor)
        optimizer.zero_grad()
        output_var = logistic_model(input_var)

        loss = criterion(output_var, target_var)

        loss.backward()
        optimizer.step()

        prediction = logistic_model(test_data)
        pred_y = prediction.data.numpy().argmax(axis=1)

        if epoch % print_every == 0:
            print("Epoch %d, cost = %f, acc = %.2f%%"
                % (epoch + 1, loss.data[0], 100. * np.mean(pred_y == test_y.reshape(-1))))
