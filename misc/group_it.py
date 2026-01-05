import torch

def group(Xs : torch.Tensor, 
          angle_basis : torch.Tensor, 
          weight_threshold = 0.003, 
          angle_threshold = 15.0,
          max_group_number = 2,
          fiber_fraction_threshold = .0001):
    
    """ 
    Inputs
    ---------
    
    1. Xs <-> reso_ratio_fold: post normalization fiber signals
    2. Angle Basis
    3. Weight Threshold
    4. Angle Threshold
    5. Max Group Number
    6. Fiber Weight Threshold (We do this after the fact, I suppose)

    """
    ith_fiber_weights = torch.zeros((Xs.shape[0],max_group_number, Xs.shape[1])) 
    Xs_fiber = Xs[:, 0:angle_basis.shape[0]]
    Xs_fiber_cpy = Xs_fiber.clone() 
    Xs_isotropic = Xs[:, angle_basis.shape[0]:]

    if max_group_number > 1:
        for number_group in range(max_group_number):
            current_max_weight, current_max_index = torch.max(Xs_fiber_cpy, dim = 1)
            if all(current_max_weight < weight_threshold):
                break
            else:
                current_direction    = angle_basis[current_max_index,:]      
                crossing_angles      = torch.clamp(torch.einsum('bi, ji -> bj', current_direction, angle_basis), min = -.9999, max = .9999) # clamp for numerical stability of torch.arccos
                crossing_angles      = torch.rad2deg(torch.arccos(crossing_angles))
                angle_neighbors      = torch.logical_or(crossing_angles <= angle_threshold, crossing_angles >= 180-angle_threshold)
                
                weights              = Xs_fiber_cpy > weight_threshold
                selected_neighbors   = torch.logical_and(angle_neighbors, weights)     
                
                ith_fiber_weights[:, number_group, 0:angle_basis.shape[0]][selected_neighbors] = Xs_fiber[selected_neighbors]         
                Xs_fiber_cpy[selected_neighbors] = 0.
       
        ith_fiber_weights[:, :, angle_basis.shape[0]:] = Xs_isotropic[:, None, :]
        
        for number_group in reversed(range(max_group_number)):
            """
            Recurrsivly traverse up the grouped fiber fractions to eliminate trivial fibers. 
            """
            if number_group == 0: 
                break
            else:
                ith_grouped_fiber_fraction = ith_fiber_weights[:, number_group, 0:angle_basis.shape[0]].sum(dim = 1)
                merge_map = ith_grouped_fiber_fraction < fiber_fraction_threshold
                ith_fiber_weights[merge_map, number_group - 1, 0:angle_basis.shape[0]] += ith_fiber_weights[merge_map, number_group, 0:angle_basis.shape[0]]
                ith_fiber_weights[merge_map, number_group, 0:angle_basis.shape[0]] = 0.
            
        return ith_fiber_weights.reshape(-1, Xs.shape[1])
            
    else:
        return Xs