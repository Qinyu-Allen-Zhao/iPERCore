import os
import os.path as osp
import platform

# the gpu ids
gpu_ids = "0"

# the image size
image_size = 512

# the default number of source images, it will be updated if the actual number of sources <= num_source
num_source = 2

# the assets directory. This is very important, please download it from `one_drive_url` firstly.
assets_dir = "./assets"

# the output directory.
output_dir = "./results"

# the model id of this case. This is a random model name.
# model_id = "model_" + str(time.time())

# # This is a specific model name, and it will be used if you do not change it.
# model_id = "axing_1"

# symlink from the actual assets directory to this current directory
work_asserts_dir = os.path.join("./assets")
if not os.path.exists(work_asserts_dir):
    os.symlink(osp.abspath(assets_dir), osp.abspath(work_asserts_dir),
               target_is_directory=(platform.system() == "Windows"))

cfg_path = osp.join(work_asserts_dir, "configs", "deploy.toml")

model_id = "synthesis_%d" % 8

src_path = "\"path?=./assets/samples/sources/syn_subjects/%d.jpg,name?=syn_subjects_%d\"" % (8, 8)

ref_path = "\"path?=./assets/samples/references/akun_1.mp4," \
               "name?=akun_1," \
               "pose_fc?=400\""

command = f"python -m iPERCore.services.run_imitator" + \
              f" --gpu_ids {gpu_ids}" + \
              f" --num_source {num_source}" + \
              f" --image_size {image_size}" + \
              f" --output_dir {output_dir}" + \
              f" --model_id {model_id}" + \
              f" --cfg_path {cfg_path}" + \
              f" --src_path {src_path}" + \
              f" --ref_path {ref_path}" + \
              f" --sub_id {8}"
os.system(command)
