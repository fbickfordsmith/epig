python main.py --multirun rng.seed="range(20)" data=mnist/curated_pool experiment_name=mnist_curated acquisition.objective=random
python main.py --multirun rng.seed="range(20)" data=mnist/curated_pool experiment_name=mnist_curated acquisition.objective=bald
python main.py --multirun rng.seed="range(20)" data=mnist/curated_pool experiment_name=mnist_curated acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/curated_approx experiment_name=mnist_curated acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/curated_pool experiment_name=mnist_curated acquisition.objective=marg_entropy
python main.py --multirun rng.seed="range(20)" data=mnist/curated_pool experiment_name=mnist_curated acquisition.objective=badge acquisition.batch_size=10 data.batch_sizes.pool=1
python main.py --multirun rng.seed="range(20)" data=mnist/unbalanced_pool experiment_name=mnist_unbalanced acquisition.objective=random
python main.py --multirun rng.seed="range(20)" data=mnist/unbalanced_pool experiment_name=mnist_unbalanced acquisition.objective=bald
python main.py --multirun rng.seed="range(20)" data=mnist/unbalanced_pool experiment_name=mnist_unbalanced acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/unbalanced_test experiment_name=mnist_unbalanced acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/unbalanced_approx experiment_name=mnist_unbalanced acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/unbalanced_pool experiment_name=mnist_unbalanced acquisition.objective=marg_entropy
python main.py --multirun rng.seed="range(20)" data=mnist/unbalanced_pool experiment_name=mnist_unbalanced acquisition.objective=badge acquisition.batch_size=10 data.batch_sizes.pool=1
python main.py --multirun rng.seed="range(20)" data=mnist/redundant_pool experiment_name=mnist_redundant acquisition.objective=random
python main.py --multirun rng.seed="range(20)" data=mnist/redundant_pool experiment_name=mnist_redundant acquisition.objective=bald
python main.py --multirun rng.seed="range(20)" data=mnist/redundant_pool experiment_name=mnist_redundant acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/redundant_test experiment_name=mnist_redundant acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/redundant_approx experiment_name=mnist_redundant acquisition.objective=epig
python main.py --multirun rng.seed="range(20)" data=mnist/redundant_pool experiment_name=mnist_redundant acquisition.objective=marg_entropy
python main.py --multirun rng.seed="range(20)" data=mnist/redundant_pool experiment_name=mnist_redundant acquisition.objective=badge acquisition.batch_size=10 data.batch_sizes.pool=1