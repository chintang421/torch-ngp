python -m pdb main_nerf.py data/dnerf/jumpingjacks/ --workspace trial_nerf -O --tcnn --mode blender --gui --scale 1.0
python -m pdb main_nerf.py data/dnerf/lego --workspace trial_nerff -O --tcnn --mode blender --scale 1.0 --tnerf --iters 2000
CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/dnerf/lego --workspace exp_lego_dnerf -O --tcnn --mode blender --scale 1.0 --dnerf --iters 2000