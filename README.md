obtain steering vector from GoEmotion --> generation&classification test --> benchmark to see if improve performance

1. `python src/data_loader_emo.py --emotion anger --split train`

2. `python src/recorder.py --emotion anger --split train --model_name gpt2 --layer -1 --batch_size 2 --device cuda`

3. `python src/analyzer.py --emotion anger --split train --layer -1 --top_k 50`

4. `python src/injector.py --emotion anger --split train --layer -1 --vector_type mean`

5. `python src/evaluator.py --model_name gpt2 --vector_path data/processed/anger/steering_vector/train_layer-1_mean_k50.npy --layer -1 --device cpu --max_length 10 "The meeting ended and" "I just read the news and"`

6. `python src/data_loader_truthful.py --splits validation --output_dir data/benchmarks`


