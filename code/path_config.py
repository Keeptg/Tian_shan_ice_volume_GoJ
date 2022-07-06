import os 
import sys 
import socket 


host_name = socket.gethostname()
script_dir = sys.path[0]
if host_name == 'lifei-laptop':
    root_dir = '/home/lifei/Data/Study_in_Innsbruck/'
    data_dir = os.path.join(root_dir, 'Data', 'Shpfile', 'Tienshan_data')
    work_dir = os.path.join(root_dir, 'Data', 'model_output')
    out_dir = os.path.join(root_dir, 'Data', 'model_output')
    cluster_dir = os.path.join(root_dir, 'Data', 'cluster_output')
    for dir_ in [root_dir, data_dir, work_dir, out_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_in_cluster = False
elif host_name in ['login01', 'login02'] or 'node' in host_name:
    root_dir = '/home/users/lifei/'
    data_dir = os.path.join(root_dir, 'Script_laptop', 'Tienshan_data')
    work_dir = os.environ['WORKDIR']
    out_dir = '/home/www/lifei/run_output'
    for dir_ in [root_dir, data_dir, work_dir, out_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_in_cluster = True
elif host_name == 'DESKTOP-3N46MRQ':
    root_dir = '/home/lifei/My-OGGM-script'
    data_dir = os.path.join(root_dir, 'Tienshan_data')
    work_dir = os.path.join(root_dir, 'test_working_dir')
    out_dir = os.path.join(root_dir, 'out_dir')
    for dir_ in [root_dir, data_dir, work_dir, out_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_in_cluster = False
elif host_name == 'WGJ-Group':
    root_dir = '/home/lifei/Data/code_project/Tienshan_paper/'
    data_dir = os.path.join(root_dir, 'Tienshan_data')
    work_dir = os.path.join(root_dir, 'model_output')
    out_dir = os.path.join(root_dir, 'model_output')
    cluster_dir = os.path.join(root_dir, 'cluster_output')
    for dir_ in [root_dir, data_dir, work_dir, out_dir, cluster_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_in_cluster = False
else:
    raise ValueError(f"Get unexcepted hostname: {host_name}!")