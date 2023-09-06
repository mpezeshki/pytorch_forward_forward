import torch


def FedAvg(Userlist, Global_model):
    """
    :param w: the list of user
    :return: the userlist after aggregated and the global mode
    """


    l_user = len(Userlist)    # the number of user

    client_weights = [1/l_user for i in range(l_user)]
    with torch.no_grad():
        for key in Global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                Global_model.state_dict()[key].data.copy_(
                    Userlist[0].state_dict()[key])
            else:
                temp = torch.zeros_like(
                    Global_model.state_dict()[key], dtype=torch.float32)

                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * \
                        Userlist[client_idx].state_dict()[
                        key]

                Global_model.state_dict()[
                    key].data.copy_(temp)

                for client_idx in range(len(client_weights)):
                    Userlist[client_idx].state_dict()[key].data.copy_(
                        Global_model.state_dict()[key])
    return Userlist, Global_model