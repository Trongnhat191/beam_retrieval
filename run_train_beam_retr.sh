CUDA_VISIBLE_DEVICES=0 \
python train_beam_retriever.py \
--do_train \
--prefix retr_hotpot_beam_size2_large \
--model_name chandar-lab/NeoBERT \
--tokenizer_path chandar-lab/NeoBERT \
--dataset_type hotpot \
--train_file datasets/mrc/hotpotqa/hotpot_train_v1.1.json \
--predict_file datasets/mrc/hotpotqa/hotpot_dev_distractor_v1.json \
--train_batch_size 4 \
--learning_rate 2e-5 \
--fp16 \
--beam_size 2 \
--predict_batch_size 1 \
--warmup-ratio 0.1 \
--num_train_epochs 1 \
--mean_passage_len 250 \
--log_period_ratio 0.01 \
--accumulate_gradients 4 \
--eval_period_ratio 0.3
