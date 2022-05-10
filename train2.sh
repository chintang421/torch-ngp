CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/dnerf/mutant --workspace exp_dnerf/mutant -O --tcnn --mode blender --scale 1.0 --dnerf --iters 250000
CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/dnerf/hook --workspace exp_dnerf/hook -O --tcnn --mode blender --scale 1.0 --dnerf --iters 250000
CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/dnerf/jumpingjacks --workspace exp_dnerf/jumpingjacks -O --tcnn --mode blender --scale 1.0 --dnerf --iters 250000
