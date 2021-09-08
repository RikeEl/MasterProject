# for reproducibility (do not change)
torch.manual_seed(0)

# parameters
batch_size = 32
num_epochs = 10

# models
featnet1 = torch.load('/content/featnet1.pt').cuda()
classifier1 = Classifier1().cuda()
print('Parameter count:', mdl_e1_utils.parameter_count(featnet1) + mdl_e1_utils.parameter_count(classifier1))

# optimizer
optimizer = torch.optim.Adam(params=list(featnet1.parameters()) + list(classifier1.parameters()), lr=0.001)  # TODO

# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)  # TODO

# criterion
class_weights = 1 / torch.sqrt(
    class_distribution)  # 1-torch.sum(F.one_hot(torch.Tensor([data['target'] for data in dataset_train]).to(torch.int64), 4), 0) / num_recordings_train #TODO
print(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())  # TODO

# input and target
input_train = torch.stack([entry['input'] for entry in dataset_train]).cuda()
target_train = torch.cat([entry['target'] for entry in dataset_train]).cuda()
input_valid = torch.stack([entry['input'] for entry in dataset_valid]).cuda()
target_valid = torch.cat([entry['target'] for entry in dataset_valid]).cuda()

# statistics
losses_train = []
f1s_train = []
losses_valid = []
f1s_valid = []

# for num_epochs
for epoch in range(num_epochs):

    # train mode
    featnet1.train()
    classifier1.train()

    # update learning rate
    lr_scheduler.step()

    # random mini-batches
    batch_train = torch.randperm(num_recordings_train)
    batch_train = batch_train[(batch_train.numel() % batch_size):]
    batch_train = batch_train.view(-1, batch_size)

    # statistics
    running_loss = 0.0
    conf_mat = torch.zeros(num_classes, num_classes).cuda()

    # for each mini-batch
    for i in range(batch_train.size(0)):
        # mini-batch
        input = input_train[batch_train[i], :].unsqueeze(1)
        target = target_train[batch_train[i]]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = featnet1(input)
        output1 = classifier1(output)
        # print("Input-Shape: {}\nTarget-Shape: {}\nOutput-Shape: {}\nOutput1-Shape: {}\n Max of Target: {}".format(input.shape, target.shape, output.shape, output1.shape, torch.max(target)))
        loss = criterion(output1, target)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        # conf_mat += mdl_e1_utils.confusion_mat(output, target)

    running_loss /= batch_train.size(0)
    f1 = mdl_e1_utils.f1_score(conf_mat)

    losses_train.append(running_loss)
    f1s_train.append(f1)

    # output
    print('Epoch {} (train) -- loss: {:.4f} f1: {:.4f}'.format(epoch, running_loss, f1))

    # validate
    with torch.no_grad():

        # eval mode
        featnet1.eval()
        classifier1.eval()

        # random mini-batches
        batch_valid = torch.randperm(num_recordings_valid)
        batch_valid = batch_valid[(batch_valid.numel() % batch_size):]
        batch_valid = batch_valid.view(-1, batch_size)

        # statistics
        running_loss = 0.0
        conf_mat = torch.zeros(num_classes, num_classes).cuda()

        # for each mini-batch
        for i in range(batch_valid.size(0)):
            # mini-batch
            input = input_valid[batch_valid[i], :].unsqueeze(1)
            target = target_valid[batch_valid[i]]

            # forward
            output = classifier1(featnet1(input))
            loss = criterion(output, target)

            # statistics
            running_loss += loss.item()
            conf_mat += mdl_e1_utils.confusion_mat(output, target)

        running_loss /= batch_valid.size(0)
        f1 = mdl_e1_utils.f1_score(conf_mat)

        losses_valid.append(running_loss)
        f1s_valid.append(f1)

        # output
        print('Epoch {} (valid) -- loss: {:.4f} f1: {:.4f}'.format(epoch, running_loss, f1))

# best F1 score
print('Best F1 score (valid):', '{:.2f}'.format(round(max(f1s_valid), 2)))