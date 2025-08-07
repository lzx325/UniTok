#!/usr/bin
set -e
source setup.sh
if false; then
	torchrun \
	--nnodes=${nnodes} \
	--node_rank=${node_rank} \
	--master_addr=${master_addr} \
	--master_port=${master_port} \
	--nproc_per_node=${nproc_per_node} \
	main.py \
	--local_bs 64 \
	--vae_local_bs 64 \
	--vocab_size 32768 \
	--num_codebooks 8 \
	--report_wandb True \
	--model 'vitamin_large' \
	--exp_name 'unitok_pretrain' \
	--vis_img_dir 'assets/vis_imgs/' \
	"$@"
fi

if true; then
	nnodes=1
	nproc_per_node=1
	train_data="$datadr/datacomp/bulk_data/small/shards/{00000000..00000010}.tar"
	output_dir="./bulk_data/output_dir-train_debug"
	torchrun \
	--nnodes=${nnodes} \
	--nproc_per_node=${nproc_per_node} \
	main.py \
	--local_bs 1 \
	--vae_local_bs 1 \
	--vocab_size 32768 \
	--num_codebooks 8 \
	--report_wandb True \
	--model 'vitamin_large' \
	--exp_name 'unitok_pretrain' \
	--train_data "$train_data" \
	--output_dir "$output_dir" \
	--vis_img_dir "assets/vis_imgs/" \
	"$@"
fi

if false; then
	src_dir="assets/vis_imgs"
	rec_dir="bulk_data/rec_img_dir"
	mkdir -p "$rec_dir"

	# for img in "$src_dir"/*.jpg; do
	# 	img_name=$(basename "$img")
	# 	python inference.py \
	# 	--ckpt_path bulk_data/checkpoints/unitok_tokenizer.pth \
	# 	--src_img "$img" \
	# 	--rec_img "$rec_dir/$img_name"
	# done

	python inference.py \
    --ckpt_path bulk_data/checkpoints/unitok_tokenizer.pth \
    --src_img assets/vis_imgs/v0.jpg \
	--rec_img bulk_data/rec_img_dir/v0.jpg
fi
