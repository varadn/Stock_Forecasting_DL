import torch

# def train_one_epoch(epoch, loss_function, optimizer, model, train_loader, device):
    

#     #for printing val loss
# def validate_one_epoch(loss_function, model, test_loader, device):


def train_model(num_epochs, model, loss_function, optimizer, train_loader,test_loader, device):
    for epoch in range(num_epochs):

        ### TRAIN ###
        model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0
        
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            #forward prop
            output = model(x_batch)
            #loss
            # print(output.shape)
            # print(y_batch.shape)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            #backward prop
            
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.7f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

        ### VALIDATE ###
        model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        print('Val Loss: {0:.7f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()
    
    return model