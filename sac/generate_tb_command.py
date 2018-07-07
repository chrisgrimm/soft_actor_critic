import os

def produce_tensorboard_command(run_dir, common_path, run_names):
    run_paths = [f'{run_name}:{os.path.join(run_dir, run_name, common_path)}' for run_name in run_names]
    return f"tensorboard --logdir={','.join(run_paths)}"



run_names = (
    #[f'nc_future{i+1}' for i in range(0, 6)] +
    #[f'nc_future_10_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w256_d3_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w128_d2_rs0.1_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w128_d2_rs0.005_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w256_d3_rs0.01_lr1e-5_{i+1}' for i in range(0, 6)]
    #[f'nc_future_10_{i+1}' for i in range(0, 6)] +
    #[f'nc_future256_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w256_d3_{i+1}' for i in range(0, 6)]
    #[f'centered_run{i+1}' for i in range(0, 6)] +
    #[f'future_centered_run{i+1}' for i in range(0, 4)] +
    #[f'nc_future5_w128_d2_rs0.1_{i+1}' for i in range(0,6)]
    #[f'nc_future5_w256_d2_lr5e-5_clip100_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w256_d2_lr5e-5_clip10_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w128_d2_lr5e-5_mixed_{i+1}' for i in range(0, 6)] +
    #[f'nc_future5_w128_d2_lr5e-5_clip100_rs0.005_{i+1}' for i in range(0, 3)] +
    #[f'nc_future5_w128_d2_lr5e-5_clip100_rs0.0075_{i+1}'for i in range(0, 3)]
    [f'nc_future_ent_mult_{i}' for i in range(6)] +
    [f'nc_final_{i}' for i in range(6)] +
    [f'nc_future_mixed_actions_{i}' for i in range(6)]
)

print(produce_tensorboard_command('runs', 'data', run_names))