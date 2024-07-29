import subprocess
import os
import itertools

params = {
    'R': [64, 96],
    'Lb': [64, 128, 256, 384],
    'use_cagra_graph': [False],
    'QD' : [192]
}

params_cagra = {
    'R': [64, 96],
    'use_cagra_graph': [True],
    'cagra_graph_degree': [32, 64, 96],
    'cagra_intermediate_graph_degree': [64, 96, 128],
    'QD' : [192]
}

# Define the constraints
constraints = [
    lambda x: x['R'] <= x['Lb'],
    # lambda x: x['R'] == 96 or (x['R'] == 64 and x['Lb'] == 384)
]

cagra_constraints = [
    lambda x: x['cagra_graph_degree'] <= x['R'] and x['cagra_graph_degree'] <= x['cagra_intermediate_graph_degree']    
]

output_dir = '/datasets/tarangj/datasets/wiki_all_1M/DiskANNSSD'
ssd_builder_path = os.path.join("/home/nfs/tarangj/DiskANN/build", "apps", "build_disk_index")
vectors_bin_path = '/datasets/tarangj/datasets/wiki_all_1M/base.1M.fbin'

# Define a function to generate the output file name
def generate_output_file(dir, params):
    if not params['use_cagra_graph']:
        print("here in generate_output_file")
        output_file = f"diskann.R{params['R']}.Lb{params['Lb']}.QD{params['QD']}.use_cagra_graphFalse"
    else:
        output_file = f"diskann.R{params['R']}.QD{params['QD']}.use_cagra_graphTrue.cagra_graph_degree{params['cagra_graph_degree']}.cagra_intermediate_graph_degree{params['cagra_intermediate_graph_degree']}"
    print("output_file", output_file)
    return os.path.join(dir, output_file)

count = 0
# Run the grid search
for combo in itertools.product(*params.values()):
    # if count > 0:
    #     continue
    # Create a dictionary from the combination
    combo_dict = dict(zip(params.keys(), combo))
    
    # Check the constraints
    if all(constraint(combo_dict) for constraint in constraints):
        # Generate the output file name
        output_file = generate_output_file(output_dir, combo_dict)

        args = [
            ssd_builder_path,
            "--data_type", "float",
            "--dist_fn", "l2",
            "--data_path", vectors_bin_path,
            "--index_path_prefix", output_file,
            "-R", str(combo_dict["R"]),
            "-L", str(combo_dict["Lb"]),
            "--QD", str(combo_dict["QD"]),
            "--search_DRAM_budget", "100",
            "--build_DRAM_budget", "100",
            "--num_threads", "80",
            "--build_PQ_bytes", str(combo_dict["QD"]),
            # "--use_cuvs_cagra_graph", "false"
        ]

        completed = subprocess.run(args, timeout=3600)

        output_file

        if completed.returncode != 0:
            command_run = " ".join(args)
            raise Exception(f"Unable to build a disk index with the command: '{command_run}'\ncompleted_process: {completed}\nstdout: {completed.stdout}\nstderr: {completed.stderr}")
    # count += 1

# Run the grid search
count = 0
for combo in itertools.product(*params_cagra.values()):
    # if count > 0:
    #     continue
    print("here")
    # Create a dictionary from the combination
    combo_dict = dict(zip(params_cagra.keys(), combo))
    
    # Check the constraints
    if all(constraint(combo_dict) for constraint in cagra_constraints):
        # Generate the output file name
        output_file = generate_output_file(output_dir, combo_dict)

        args = [
            ssd_builder_path,
            "--data_type", "float",
            "--dist_fn", "l2",
            "--data_path", vectors_bin_path,
            "--index_path_prefix", output_file,
            "-R", str(combo_dict["R"]),
            "-L", "128",
            "--QD", str(combo_dict["QD"]),
            "--search_DRAM_budget", "100",
            "--build_DRAM_budget", "100",
            "--num_threads", "80",
            "--build_PQ_bytes", str(combo_dict["QD"]),
            "--use_cuvs_cagra_graph", "true"
        ]

        completed = subprocess.run(args, timeout=3600)

        output_file

        if completed.returncode != 0:
            command_run = " ".join(args)
            raise Exception(f"Unable to build a disk index with the command: '{command_run}'\ncompleted_process: {completed}\nstdout: {completed.stdout}\nstderr: {completed.stderr}")
    
    # count += 1