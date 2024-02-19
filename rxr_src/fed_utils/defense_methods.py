#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
#from statistics import StatisticsError, mode
import pandas as pd
import numpy as np
#from statistics import median
import torch


def no_detect(gradients):
    return []


# this function does not return candidates
def median(parameters):
    
    new_params = {}
    for name in parameters[0].keys():
        if len(parameters[0][name].shape) > 0:
            # quantile or mean on gradient is the same as new parameters where constant is added to all elements.
            new_params[name] = torch.quantile(torch.stack([param[name].data for param in parameters]), dim=0, q=0.5)
        else:
            # handle 0 dimensional parameter
            new_params[name] = parameters[0][name]           
 
    # ensure param shape is preserved
    assert parameters[0][name].shape == new_params[name].shape

    return new_params


# this function does not return candidates
def tr_mean(parameters, n_attackers):
    assert n_attackers > 0

    new_params = {}
    for name in parameters[0].keys():
        if len(parameters[0][name].shape) > 0:
            potential_params = torch.sort(torch.stack([param[name].data for param in parameters]), 0)[0]
            # quantile or mean on gradient is the same as new parameters where constant is added to all elements.
            new_params[name] = torch.mean(potential_params[n_attackers:-n_attackers], 0)
        else:
            # handle 0 dimensional parameter
            new_params[name] = parameters[0][name]
        # ensure param shape is preserved
        assert parameters[0][name].shape == new_params[name].shape

    return new_params

def cal_cos_layerwise(parameters):
    distances = []
    for param in parameters:
        distance = []
        for param_ in parameters:
            dist = 0
            for name in parameters[0].keys():
                norm_1 = torch.nn.functional.normalize(param[name], p = 2, dim = -1)
                norm_2 = torch.nn.functional.normalize(param_[name], p = 2, dim = -1)
                cos_sim = torch.mul(norm_1, norm_2).sum(dim = -1).sum()
                dist += cos_sim
            distance.append(dist)
        distance = torch.Tensor(distance).float()
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    return distances

def cal_cos(gradients):
    distances = []
    grads = torch.from_numpy(gradients)
    for grad in grads:
        distance = []
        for grad_ in grads:
            norm_1 = torch.nn.functional.normalize(grad, p = 2, dim = -1)
            norm_2 = torch.nn.functional.normalize(grad_, p = 2, dim = -1)
            cos_sim = torch.mul(norm_1, norm_2).sum()
            distance.append(cos_sim)
        distance = torch.Tensor(distance).float()
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    return distances



# def multi_krum(grads, n_attackers, multi_k=False):
#     #grads = flatten_grads(gradients)
#     candidates = []
#     candidate_indices = []
#     remaining_updates = torch.from_numpy(grads)
#     all_indices = np.arange(len(grads))
#     distances = []
#     for i, update in enumerate(remaining_updates):
#         distance = []
#         for j, update_ in enumerate(remaining_updates):
#             distance.append(torch.norm((update - update_)) ** 2)
#         distance = torch.Tensor(distance).float()
#         distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

#     while len(distances) > 2 * n_attackers + 2:
#         torch.cuda.empty_cache()
#         distances = torch.sort(distances, dim=1)[0]
#         scores = torch.sum(distances[:, :len(distances) - 1 - n_attackers], dim=1)
#         indices = torch.argsort(scores)[:len(distances) - 1 - n_attackers]

#         candidate_indices.append(all_indices[indices[0].cpu().numpy()])
#         all_indices = np.delete(all_indices, indices[0].cpu().numpy())
#         #candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
#         #remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
#         distances = torch.cat((distances[:indices[0]], distances[indices[0] + 1:]), dim = 0)
#         distances = torch.cat((distances[:, :indices[0]], distances[:, indices[0] + 1:]), dim = 1)
#         if not multi_k:
#             break

#     # aggregate = torch.mean(candidates, dim=0)

#     # return aggregate, np.array(candidate_indices)
#     return np.array(candidate_indices)

def multi_krum_cos(dists, n_attackers, multi_k=False):
    #grads = flatten_grads(gradients)
    candidates = []
    candidate_indices = []
    all_indices = np.arange(len(dists))
    distances = dists.clone()
    
    while len(distances) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        sorted_distances = torch.sort(distances, dim=1, descending = True)[0]
        scores = torch.sum(sorted_distances[:, :len(sorted_distances) - 1 - n_attackers], dim=1)
        indices = torch.argsort(scores, descending = True)[:len(sorted_distances) - 1 - n_attackers]
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        #candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        #remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        distances = torch.cat((distances[:indices[0]], distances[indices[0] + 1:]), dim = 0)
        distances = torch.cat((distances[:, :indices[0]], distances[:, indices[0] + 1:]), dim = 1)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    return np.array(candidate_indices)

def bulyan_align(dists, n_attackers, bot_lim):
    #grads = flatten_grads(gradients)
    candidates = []
    candidate_indices = []
    all_indices = np.arange(len(dists))
    distances = dists.clone()
    
    while len(distances) > bot_lim:
        torch.cuda.empty_cache()
        sorted_distances = torch.sort(distances, dim=1, descending = True)[0]
        scores = torch.sum(sorted_distances[:, :len(sorted_distances) - 1 - n_attackers], dim=1)
        indices = torch.argsort(scores, descending = True)[:len(sorted_distances) - 1 - n_attackers]
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        #candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        #remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        distances = torch.cat((distances[:indices[0]], distances[indices[0] + 1:]), dim = 0)
        distances = torch.cat((distances[:, :indices[0]], distances[:, indices[0] + 1:]), dim = 1)

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    return np.array(candidate_indices)

def multi_krum(grads, n_attackers, multi_k=False):

    #grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))
    return_dist = []
    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        if len(return_dist) == 0:
            return_dist = distances
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 1 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 1 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    return np.array(candidate_indices)

def bulyan(grads, n_attackers):

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))
    distances = []
    for i, update in enumerate(remaining_updates):
        distance = []
        for j, update_ in enumerate(remaining_updates):
            distance.append(torch.norm((update - update_)) ** 2)
        distance = torch.Tensor(distance).float()
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    while len(distances) > 2 * n_attackers:
        torch.cuda.empty_cache()
        sorted_distances = torch.sort(distances, dim=1, descending = True)[0]
        scores = torch.sum(sorted_distances[:, :len(sorted_distances) - 1 - n_attackers], dim=1)
        indices = torch.argsort(scores, descending = True)[:len(sorted_distances) - 1 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        distances = torch.cat((distances[:indices[0]], distances[indices[0] + 1:]), dim = 0)
        distances = torch.cat((distances[:, :indices[0]], distances[:, indices[0] + 1:]), dim = 1)

    return candidate_indices

def bulyan_cos(dists, n_attackers):
    #grads = flatten_grads(gradients)
    candidates = []
    candidate_indices = []
    all_indices = np.arange(len(dists))
    distances = dists.clone()

    while len(distances) > 2 * n_attackers:
        torch.cuda.empty_cache()
        sorted_distances = torch.sort(distances, dim=1, descending = True)[0]
        scores = torch.sum(sorted_distances[:, :len(sorted_distances) - 1 - n_attackers], dim=1)
        indices = torch.argsort(scores, descending = True)[:len(sorted_distances) - 1 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        distances = torch.cat((distances[:indices[0]], distances[indices[0] + 1:]), dim = 0)
        distances = torch.cat((distances[:, :indices[0]], distances[:, indices[0] + 1:]), dim = 1)

    return np.array(candidate_indices)

def mandera_detect(gradients):
    # gradients is a dataframe, poi_index is a lite-type object
    if type(gradients) == pd.DataFrame:
        ranks = gradients.rank(axis=0, method='average')
        vars = ranks.var(axis=1).pow(1./2)
        mus = ranks.mean(axis=1)
        feats = pd.concat([mus, vars], axis=1)
        assert feats.shape == (gradients.shape[0], 2)
    elif type(gradients) == list:
        flat_grad = flatten_grads(gradients)
        ranks = pd.DataFrame(flat_grad).rank(axis=0, method='average')
        vars = ranks.var(axis=1).pow(1./2)
        mus = ranks.mean(axis=1)
        feats = pd.concat([mus, vars], axis=1)
        assert feats.shape == (ranks.shape[0], 2)
    else:
        print("Support not implemented for generic matrixes, please use a pandas dataframe, or a list to be cast into a dataframe")
        assert type(gradients) in [pd.DataFrame, list]

    # scaler = StandardScaler()
    # feats = scaler.fit_transform(feats.values)

    model = KMeans(n_clusters=2)
    group = model.fit_predict(feats.values)
    group = np.array(group)

    diff_g0 = len(vars[group == 0]) - vars[group == 0].nunique()
    diff_g1 = len(vars[group == 1]) - vars[group == 1].nunique()

    # diff_g0 = len(vars[group == 0]) - gradients[group == 0].nunique(axis=1)
    # diff_g1 = len(vars[group == 1]) - gradients[group == 1].nunique(axis=1)

    # diff_g0 = len(vars[group == 0]) - gradients[0][group == 1].nunique()
    # diff_g1 = len(vars[group == 1]) - gradients[0][group == 1].nunique()
   
    # if no group found with matching gradients, mark the smaller group as malicious
    if diff_g0 == diff_g1:
        # get the minority label
        try:
            bad_label = (mode(group) + 1) % 2
        except StatisticsError:
            # equally sized groups, select the first group to keep.
            bad_label = 0
    elif diff_g0 < diff_g1:
        bad_label = 1
    elif diff_g0 > diff_g1:
        bad_label = 0
    else:
        assert False

    # see which indexes match the minority label
    predict_poi = [n for n, l in enumerate(group) if l == bad_label]

    return predict_poi


def fltrust(grads):
    """
    gradients: list of gradients. The last one is the trusted server bootstrap update.
    """

    n = len(grads) - 1
    
    # use the last gradient (server update) as the trusted source
    baseline = grads[-1]
    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
    for each_param_list in grads:
        each_param_array = np.array(each_param_list).squeeze()
        _cos = np.dot(baseline, each_param_array) / (np.linalg.norm(baseline) + 1e-9) / (np.linalg.norm(each_param_array) + 1e-9)
        cos_sim.append(_cos)
    cos_sim = np.stack(cos_sim)[:-1]
    cos_sim = np.maximum(cos_sim, 0) # relu
    normalized_weights = cos_sim / (np.sum(cos_sim) + 1e-9) # weighted trust score

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(grads[i] * normalized_weights[i] / (np.linalg.norm(grads[i]) + 1e-9) * np.linalg.norm(baseline))

    # update the global model
    global_update = np.sum(new_param_list, axis=0)
    assert global_update.shape == grads[-1].shape
  
    return global_update

def unflatten_grads(flat_param, param_example):

    new_params = copy.deepcopy(param_example)
    param_order = new_params.keys()

    i = 0
    for param in param_order:
        n_flat = len(new_params[param].flatten())
        new_params[param] = torch.tensor(flat_param[i:i+n_flat].reshape(new_params[param].shape)).to(device='cuda')
        i = i + n_flat
    return new_params

def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend([grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs

def flatten_param(gradients, params):

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in params:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend([grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs
        

if __name__ == "__main__":
    import pickle
    grads_1 = pickle.load(open("../sf_debug_grads.pickle", "rb"))

    # a = fltrust(grads_1)

    import time

    def timeit_1arg(def_function, grad_1, number):
        timings = []
        for _ in range(number):
            start_time = time.perf_counter()
            def_function(grad_1)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
        return timings

    def timeit_2arg(def_function, grad_1, n_poi, number):
        timings = []
        for _ in range(number):
            start_time = time.perf_counter()
            def_function(grad_1, n_poi)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
        return timings    

    n_runs = 100

    timing_dict = {}
    
    # t = timeit_1arg(mandera_detect, grads_1, n_runs)
    # timing_dict['mandera'] = t

    # t = timeit_1arg(median, grads_1, n_runs)
    # timing_dict['median'] = t

    # t = timeit_2arg(tr_mean, grads_1, 30, n_runs)
    # timing_dict['tr_mean'] = t

    # t = timeit_2arg(multi_krum, grads_1, 30, n_runs)
    # timing_dict['multi_krum'] = t

    # t = timeit_2arg(bulyan, grads_1, 30, n_runs)
    # timing_dict['bulyan'] = t

    t = timeit_1arg(fltrust, grads_1, n_runs)
    timing_dict['fltrust'] = t

    print(timing_dict)

    pickle.dump(timing_dict, open("timings_dict_fltrust.pickle", "wb"))


    # Quick tests in ipython with %timeit

    # # 232 ms ± 3.47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # new_param = median(grads_1)

    # # 225 ms ± 1.06 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # new_param = tr_mean(grads_1, 10)

    # # 2.34 s ± 11.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # _, index = multi_krum(grads_1, 10, False)
    # print(index)

    # # 1min 1s ± 206 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # _, index = bulyan(grads_1, 10)
    # print(index)

    # # 805 ms ± 6.77 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # index = mandera_detect(grads_1)
    # print(index)