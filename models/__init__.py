import torch
import numpy as np
import tqdm

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

def mat_by_vec(M, v):
    vshape = v.shape[2]
    if len(v.shape) > 3: vshape = vshape * v.shape[3]
    return torch.matmul(M, v.view(v.shape[0] * v.shape[1], vshape,
                    1)).view(v.shape[0], v.shape[1], M.shape[0])

def vec_to_image(v, img_dim):
    return v.view(v.shape[0], v.shape[1], img_dim, img_dim)

def invert_diag(M):
    M_inv = torch.zeros_like(M)
    M_inv[M != 0] = 1 / M[M != 0]
    return M_inv


@torch.no_grad()
def general_anneal_Langevin_dynamics(H, y_0, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, c_begin = 0, sigma_0 = 1):
    U, singulars, V = torch.svd(H, some=False)
    V_t = V.transpose(0, 1)

    ZERO = 1e-3
    singulars[singulars < ZERO] = 0

    Sigma = torch.zeros_like(H)
    for i in range(singulars.shape[0]): Sigma[i, i] = singulars[i]
    S_1, S_n = singulars[0], singulars[-1]

    S_S_t = torch.zeros_like(U)
    for i in range(singulars.shape[0]): S_S_t[i, i] = singulars[i] ** 2

    num_missing = V.shape[0] - torch.count_nonzero(singulars)

    s0_2_I = ((sigma_0 ** 2) * torch.eye(U.shape[0])).to(x_mod.device)

    V_t_x = mat_by_vec(V_t, x_mod)
    U_t_y = mat_by_vec(U.transpose(0,1), y_0)

    img_dim = x_mod.shape[2]

    images = []

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='general annealed Langevin sampling'):

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (c + c_begin)
            labels = labels.long()
            step_size = step_lr * ((sigma / sigmas[-1]) ** 2)
            step_size_new = step_lr * ((1 / sigmas[-1]) ** 2)

            falses = torch.zeros(V_t_x.shape[2] - singulars.shape[0], dtype=torch.bool, device=x_mod.device)
            cond_before_lite = singulars * sigma > sigma_0
            cond_after_lite = singulars * sigma < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            step_vector = torch.zeros_like(V_t_x)
            step_vector[:, :, :] = step_size_new * (sigma**2)
            step_vector[:, :, cond_before] = step_size_new * ((sigma**2) - (sigma_0 / singulars[cond_before_lite])**2)
            step_vector[:, :, cond_after] = step_size_new * (sigma**2) * (1 - (singulars[cond_after_lite] * sigma / sigma_0)**2)

            for s in range(n_steps_each):
                grad = torch.zeros_like(V_t_x)
                score = mat_by_vec(V_t, scorenet(x_mod, labels))

                diag_mat = S_S_t * (sigma ** 2) - s0_2_I
                diag_mat[cond_after_lite, cond_after_lite] = diag_mat[cond_after_lite, cond_after_lite] * (-1)

                first_vec = U_t_y - mat_by_vec(Sigma, V_t_x)
                cond_grad = mat_by_vec(invert_diag(diag_mat), first_vec)
                cond_grad = mat_by_vec(Sigma.transpose(0,1), cond_grad)
                grad = torch.zeros_like(cond_grad)
                grad[:, :, cond_before] = cond_grad[:, :, cond_before]
                grad[:, :, cond_after] = cond_grad[:, :, cond_after] + score[:, :, cond_after]
                grad[:, :, -num_missing:] = score[:, :, -num_missing:]

                noise = torch.randn_like(V_t_x)
                V_t_x = V_t_x + step_vector * grad + noise * torch.sqrt(step_vector * 2)
                x_mod = vec_to_image(mat_by_vec(V, V_t_x), img_dim)

                if not final_only:
                    images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images