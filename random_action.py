max_force = 10
        forces_torch = max_force*(1.0-2.0 * torch.rand((num_actions*num_envs), dtype=torch.float32, device=device))