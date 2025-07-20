import numpy as np
from src.network.tensor import Tensor

DTYPE = 'float64' 
RNG = np.random.default_rng()
if DTYPE=='float64':
    EPS, ATOL, RTOL = 1e-6, 1e-5, 1e-3
else:
    EPS, ATOL, RTOL = 1e-4, 1e-4, 1e-2
K = 20

''' auto-grad testing suite
    - test all of the auto-grad primatives, 
    - test using central differences
    - test by modifying each parameter individually i.e. only do scalar pertubations
'''

def compute_central_diff_error(test_fn, test_input, 
            other_inputs, eps, perturbed_idx, tols):
    '''verify auto-grad of funciton f: R^n -> R'''
    atol, rtol = tols

    # rescale epsilon and convert to tensor
    perturbed_val = test_input.data[perturbed_idx]
    eps = eps * (1 + abs(perturbed_val))
    pertubation_tensor = np.zeros_like(test_input.data, dtype=DTYPE)
    pertubation_tensor[perturbed_idx] += eps 
    pertubation_tensor = Tensor(pertubation_tensor)

    # Compute grad
    for tensor in [test_input, *other_inputs]:
        tensor.zero_grad()
    clean_out = test_fn(test_input, other_inputs)
    clean_out.backward()
    auto_grad = test_input.grad[perturbed_idx]

    # Compute central diff Grad approximaiton
    test_forward = test_input + pertubation_tensor
    forward_out = test_fn(test_forward, other_inputs).item()
    test_back = test_input - pertubation_tensor
    back_out = test_fn(test_back, other_inputs).item()
    approx_grad = (forward_out - back_out) / (2*eps)


    abs_err = abs(approx_grad - auto_grad)
    rel_err = abs_err / (abs(auto_grad) + atol)
    is_close = abs_err <= atol + rtol*abs(auto_grad)

    return is_close, abs_err, rel_err, clean_out.item(), forward_out, back_out

# need to generate inputs, compute cd err and output/format test result, to log file maybe?
def test_fn_random_inputs(test_fn, test_shape, other_shapes=[], input_bounds=(-5, 5),
                          num_samples=K, eps=EPS, tols=(ATOL, RTOL)):
    
    test_input = Tensor.random(test_shape, input_bounds, requires_grad=True)
    other_inputs = [Tensor.random(shape, input_bounds) for shape in other_shapes]

    num_samples = min(test_input.size, num_samples)
    pertubation_nums = RNG.choice(test_input.size, size=num_samples, replace=False)
    pretubation_idxs = np.unravel_index(pertubation_nums, test_shape)

    all_close = True
    log = ''
    log += f'test input \n {test_input.data} \nother inputs \n'
    for other_input in other_inputs:
        log += f' {other_input.data} \n'
    for sample_i in range(num_samples):
        perturbed_idx = tuple(int(pert_dim[sample_i]) for pert_dim in pretubation_idxs)
        is_close, abs_err, rel_err, clean_out, forward_out, back_out = compute_central_diff_error(
                                        test_fn, test_input, other_inputs, eps, perturbed_idx, tols)
        log += f'test {'passed' if is_close else 'failed'}: abs err = {abs_err:.4f}, rel err = {rel_err:.4f}, perturbed idx = {perturbed_idx} \n'
        log += f'clean_out: {clean_out} forward_out: {forward_out} back_out: {back_out} \n'
        if not is_close:
            all_close = False

    return all_close, log
        
bin_ufuncs = {'add' : lambda test_inp, other_inps: (test_inp+other_inps[0]).sum(),
              'radd': lambda test_inp, other_inps: (other_inps[0]+test_inp).sum(),
              'sub' : lambda test_inp, other_inps: (test_inp-other_inps[0]).sum(),
              'rsub': lambda test_inp, other_inps: (other_inps[0]-test_inp).sum(),
              'mul' : lambda test_inp, other_inps: (test_inp*other_inps[0]).sum(),
              'rmul': lambda test_inp, other_inps: (other_inps[0]*test_inp).sum(),
              'pow' : lambda test_inp, other_inps: (test_inp**other_inps[0]).sum(),
              'rpow': lambda test_inp, other_inps: (other_inps[0]**test_inp).sum(),
              'truediv' : lambda test_inp, other_inps: (test_inp/other_inps[0]).sum(),
              'rtruediv': lambda test_inp, other_inps: (other_inps[0]/test_inp).sum(),}

matmul_fns = {'matmul': lambda test_inp, other_inps: (test_inp@other_inps[0]).sum(),
              'rmatmul': lambda test_inp, other_inps: (other_inps[0]@test_inp).sum(),}

unary_ufunc = {'relu': lambda test_inp, other_inps: (test_inp.relu()).sum(),
            'log': lambda test_inp, other_inps: (test_inp.log()).sum(),
            'exp': lambda test_inp, other_inps: (test_inp.exp()).sum(),
            'sum': lambda test_inp, other_inps: test_inp.sum(),
            'mean': lambda test_inp, other_inps: test_inp.mean(),}

if __name__ == '__main__':
      
    for func_name, test_fn in unary_ufunc.items():
        test_shape, other_shapes = (2, 3), [(3,2)]
        input_bounds = (1, 10) if func_name == 'log' else (-5, 5)
        all_close, log = test_fn_random_inputs(test_fn, test_shape, other_shapes, input_bounds=input_bounds)
        print(f'function: {func_name} {'passed' if all_close else 'failed'}')
        if not all_close:
            print(log)

    for func_name, test_fn in matmul_fns.items():
        test_shape = (2, 3) if func_name == 'matmul' else (3, 2)
        other_shapes = [(3, 2)] if func_name == 'matmul' else [(2, 3)]
        all_close, log = test_fn_random_inputs(test_fn, test_shape, other_shapes, input_bounds=input_bounds)
        print(f'function: {func_name} {'passed' if all_close else 'failed'}')
        if not all_close:
            print(log)

    for func_name, test_fn in bin_ufuncs.items():
        test_shape, other_shapes = (2, 3), [(2,3)]
        input_bounds = (1, 5) if (func_name == 'pow' or func_name == 'rpow') else (-5, 5)
        all_close, log = test_fn_random_inputs(test_fn, test_shape, other_shapes, input_bounds=input_bounds)
        print(f'function: {func_name} {'passed' if all_close else 'failed'}')
        if not all_close:
            print(log)