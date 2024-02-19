def fed_avg(parameters, freq_this_round):
    new_param = {}
    for key in parameters[0].keys():
        new_param[key] = sum([param[key].data * freq_this_round[idx] for idx, param in enumerate(parameters)])
    return new_param