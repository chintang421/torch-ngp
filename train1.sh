CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/dnerf/lego --workspace exp_dnerf/lego -O --tcnn --mode blender --scale 1.0 --dnerf --iters 250000
CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/dnerf/bouncingballs --workspace exp_dnerf/bouncingballs -O --tcnn --mode blender --scale 1.0 --dnerf --iters 250000
CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/dnerf/hellwarrior --workspace exp_dnerf/hellwarrior -O --tcnn --mode blender --scale 1.0 --dnerf --iters 250000
