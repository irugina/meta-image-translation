# https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py

import copy
import torch
from torch.autograd import grad
import traceback
from utils.gan_loss import *
from utils.common_helpers import *

def clone_parameters(param_list):
    return [p.clone() for p in param_list]


def clone_module(module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                cloned = module._parameters[param_key].clone()
                clone._parameters[param_key] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                clone._buffers[buffer_key] = module._buffers[buffer_key].clone()

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(module._modules[module_key])

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    clone = clone._apply(lambda x: x)
    return clone


def detach_module(module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Detaches all parameters/buffers of a previously cloned module from its computational graph.

    Note: detach works in-place, so it does not return a copy.

    **Arguments**

    * **module** (Module) - Module to be detached.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    detach_module(clone)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate on clone, not net.
    ~~~
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            detached = module._parameters[param_key].detach_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key])


def update_module(module, updates=None):
    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Updates the parameters of a module in-place, in a way that preserves differentiability.

    The parameters of the module are swapped with their update values, according to:
    \[
    p \gets p + u,
    \]
    where \(p\) is the parameter, and \(u\) is its corresponding update.


    **Arguments**

    * **module** (Module) - The module to update.
    * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the tensors in .update attributes.

    **Example**
    ~~~python
    error = loss(model(X), y)
    grads = torch.autograd.grad(
        error,
        model.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(model, updates=updates)
    ~~~
    """
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            module._parameters[param_key] = p + p.update

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            module._buffers[buffer_key] = buff + buff.update

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(module._modules[module_key],
                                                    updates=None)

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    module._apply(lambda x: x)
    return module

def compare_state_dict(dict1, dict2):
    """
    input: dict1, dict2 - OrderedDict
    {string} -> {tensor}
    return True if they're the same
    """
    if dict1.keys() != dict2.keys(): return False
    for key in dict1.keys():
        t1, t2 = dict1[key],dict2[key]
        if not (torch.equal(t1, t2)): return False
    return True

def compute_updates(module, grads, opt):
    """modifies module in-place
    sets .update attributes for all its params"""
    params = list(module.parameters())
    for p, g in zip(params, grads):
        if g is not None:
            p.update = - opt.inner_lr * g

def compute_gradients(module, loss, opt):
    if opt.allow_nograd:
        # Compute relevant gradients
        diff_params = [p for p in module.parameters() if p.requires_grad]
        grad_params = grad(loss,
                           diff_params,
                           retain_graph=not opt.first_order,
                           create_graph=not opt.first_order,
                           allow_unused=opt.allow_unused)
        gradients = []
        grad_counter = 0
        # Handles gradients for non-differentiable parameters
        for param in module.parameters():
            if param.requires_grad:
                gradient = grad_params[grad_counter]
                grad_counter += 1
            else:
                gradient = None
            gradients.append(gradient)
    else:
        try:
            gradients = grad(loss,
                             module.parameters(),
                             retain_graph= not opt.first_order,
                             create_graph= not opt.first_order,
                             allow_unused=opt.allow_unused)
        except RuntimeError:
            traceback.print_exc()
            print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')
    return gradients



def adapt_reconstruction(model, data, opt):
    loss_fn = torch.nn.L1Loss()
    # clone model
    learner = clone_module(model)
    # unpack task and put on cuda
    support_real_A, support_real_B, query_real_A , query_real_B = unpack_data(data, opt)
    # adapt learner on support frames
    for i in range(opt.inner_steps):
        # compute fake_B's for support split
        support_fake_B = learner(support_real_A)
        loss = loss_fn(support_real_B, support_fake_B)
        # SGD update
        grads = compute_gradients(learner, loss, opt)
        compute_updates(learner, grads, opt)
        update_module(learner)
    query_fake_B = learner(query_real_A)
    loss = loss_fn(query_real_B, query_fake_B)
    return loss, learner

def adapt_adversarial(model, opt, data):
    criterionGAN = GANLoss('vanilla').to(opt.device)
    criterionL1 = torch.nn.L1Loss()
    # clone G and D
    generator, discriminator = model
    G_learner = clone_module(generator)
    D_learner = clone_module(discriminator)
    # adaptation on support set
    support_real_A, support_real_B, query_real_A , query_real_B = unpack_data(data, opt)
    for i in range(opt.inner_steps):
        support_fake_B = G_learner(support_real_A)
        resized_support_real_A = nn.Upsample(scale_factor=2, mode='bilinear') (support_real_A)
        loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1 = compute_GAN_loss(
                resized_support_real_A, 
                support_real_B, 
                support_fake_B, 
                D_learner, 
                criterionGAN, 
                criterionL1,
                opt.lambda_L1)
        # SGD inner loop for generator
        loss_G = loss_G_GAN + loss_G_L1
        grads = compute_gradients(G_learner, loss_G, opt)
        compute_updates(G_learner, grads, opt)
        update_module(G_learner)
        if not opt.fix_inner_loop_discriminator: # SGD inner loop for discriminator
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            grads = compute_gradients(D_learner, loss_D, opt)
            compute_updates(D_learner, grads, opt)
            update_module(D_learner)
    # compute loss on query set
    resized_query_real_A = nn.Upsample(scale_factor=2, mode='bilinear') (query_real_A)
    query_fake_B = G_learner(query_real_A)  # G(A)
    loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1 = compute_GAN_loss(
            resized_query_real_A, 
            query_real_B, 
            query_fake_B, 
            D_learner, 
            criterionGAN, 
            criterionL1,
            opt.lambda_L1)
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_G = loss_G_GAN + loss_G_L1

    return loss_D, loss_G, D_learner, G_learner
