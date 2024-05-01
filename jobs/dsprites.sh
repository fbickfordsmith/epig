# Supervised
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_supervised  data=dsprites/image  model=pytorch_conv_net_burgess_mcdo  trainer=pytorch_neural_net_classif_mcdo  trainer.n_optim_steps_min=5_000  acquisition.method=random
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_supervised  data=dsprites/image  model=pytorch_conv_net_burgess_mcdo  trainer=pytorch_neural_net_classif_mcdo  trainer.n_optim_steps_min=5_000  acquisition.method=bald
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_supervised  data=dsprites/image  model=pytorch_conv_net_burgess_mcdo  trainer=pytorch_neural_net_classif_mcdo  trainer.n_optim_steps_min=5_000  acquisition.method=epig

# Semi-supervised
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_semi  data=dsprites/embedding  model=sklearn_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=random
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_semi  data=dsprites/embedding  model=sklearn_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=bald
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_semi  data=dsprites/embedding  model=sklearn_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=epig
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_semi  data=dsprites/embedding  model=sklearn_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=greedy_k_centers  acquisition.batch_size=10
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_semi  data=dsprites/embedding  model=sklearn_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=k_means           acquisition.batch_size=10
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_semi  data=dsprites/embedding  model=sklearn_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=probcover         acquisition.batch_size=10
python  main.py  --multirun  rng.seed=range(20)  experiment_name=dsprites_semi  data=dsprites/embedding  model=sklearn_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=typiclust         acquisition.batch_size=10