# trains neural network based on data
def train(net, x, y, optimizer, criterion, iterations):

    for i in range(iterations):

        # computes the hypothesis and cost
        outputs = net(x.float())
        cost = criterion(outputs, y)

        # edits the parameters
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print((i + 1), '- Cost: {:.4f}'.format(cost.item()))
