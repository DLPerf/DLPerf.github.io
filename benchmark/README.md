## Benchmark and Asessment
We reproduce PPs on the machine with a 16-core Intel i7-7820X CPU (3.60GHz), NVIDIA TITAN Xp GPU, 128GB RAM and 1TB SSD with TensorFlow Docker images.
The symptoms of PPs may change slightly in different hardware settings.

Our benchmark and assessment statistics could be downloaded [here](https://github.com/DLPerf/DLPerf.github.io/tree/main/benchmark/benchmark.csv).
The assessment of three performance analysis techniques are listed in "profiler_status", "xla_status" and "doc_status". The meaning of status values:
- `-1` : Not Applied.
- `0` : Not Solved.
- `1` : Partially solved for Tensorflow Profiler and XLA, applied for Tensorflow Document.

## Steps to Reproduce the Benchmark
### Download Benchmark Data
Install git lfs following the [instruction](https://git-lfs.github.com/) and clone this repo to get the benchmark and data [here](https://github.com/DLPerf/DLPerf.github.io/tree/main/benchmark/download).
Concat the splitted tars with `cat benchmark.tar* > test1.tar.gz` and `cat benchmark_data* > test2.tar.gz`. Unzip them with `tar -xvf test1.tar.gz` and `tar -xvf test2.tar.gz`.

PPs requiring different Tensorflow versions are in different folders. 

### Install Tensorflow Docker
1. Install NVIDIA docker driver and Docker Engine on the host machine, [link](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver).
2. Install Tensorflow docker images with the command `docker run --gpus all -v [benchmark_path]:/tf/mydata -it --privileged=true tensorflow/tensorflow:[tf_docker_version]`. We use the following TensorFlow images:
   - tensorflow/tensorflow:2.5.0-gpu-jupyter
   - tensorflow/tensorflow:2.4.3-gpu-jupyter
   - tensorflow/tensorflow:2.3.2-gpu-jupyter
   - tensorflow/tensorflow:2.2.3-gpu-jupyter
   - tensorflow/tensorflow:2.0.0-gpu-py3-jupyter
   - tensorflow/tensorflow:1.15.5-gpu-jupyter
   - tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
   - tensorflow/tensorflow:1.13.1-gpu-jupyter
   - tensorflow/tensorflow:1.12.3-gpu-py3


### Reproduce the benchmark
In the docker container, enter a PP directory with `cd /tf/mydata/[tf_version]/[PP_dir]`. There are `README`, `buggy.py`, `fixed.py` in each PP directory, `profile/` or `tf_xla.py` if it is applied for Tensorflow Profiler or XLA.
There may be multiple pairs of buggy file and fixed file if there are multiple PPs extracted from the same post, or multiple variants of the same PP.
The environment configuration, performance change after fixing, and reproduction steps are recorded in the `README`.

Follow the next steps to run PPs and assess them with TensorFlow Profiler or XLA:
1. Check the environment requirements in the `README`, install them with `pip install`.
2. Run `python buggy.py` or `python fixed.py` to reproduce symptoms of buggy and fixed version. 
3. If there exists `profile/`, run  `cd ./profile`,  `python buggy_profile.py`, `python fixed_profile.py` to generating profiling data. To visualize the profiling data, you should install TensorBoard with `pip install tensorboard`, and then run `tensorboard --logdir=logs/ --port=6006 --load_fast=false --bind_all`.
4. If there exists `tf_xla.py`, run  `python tf_xla.py` to reproduce the results of XLA.

PP root causes abbreviate in names of `PP_dir`:
- Not Using Efficient API: API_NUE
- Not Using Batch API: API_NUB
- Ineffient API Usage: API_IAU
- Confusion with Computation Graph: Model_CCG
- Inefficient Model Structure: Model_IMS
- Improper Model Parameter: Model_IMP
- Improper Hyper Parameter: Model_IHP
- Buggy Library Version: Library_LB
- Inefficient Data Transmission: Data_IDT
- Inefficient Data Preprocessing: Data_IDP
- Improper Data Inputs: Data_IDI

