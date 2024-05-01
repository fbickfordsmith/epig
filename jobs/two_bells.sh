python  main.py  --multirun  rng.seed=range(100)  experiment_name=two_bells_0.1k  data=two_bells/0.1k  model=gpytorch_variational  trainer=gpytorch_classif_bernoulli  acquisition.method=bald
python  main.py  --multirun  rng.seed=range(100)  experiment_name=two_bells_1k    data=two_bells/1k    model=gpytorch_variational  trainer=gpytorch_classif_bernoulli  acquisition.method=bald
python  main.py  --multirun  rng.seed=range(100)  experiment_name=two_bells_10k   data=two_bells/10k   model=gpytorch_variational  trainer=gpytorch_classif_bernoulli  acquisition.method=bald
python  main.py  --multirun  rng.seed=range(100)  experiment_name=two_bells_100k  data=two_bells/100k  model=gpytorch_variational  trainer=gpytorch_classif_bernoulli  acquisition.method=bald
python  main.py  --multirun  rng.seed=range(100)  experiment_name=two_bells_100k  data=two_bells/100k  model=gpytorch_variational  trainer=gpytorch_classif_bernoulli  acquisition.method=random
python  main.py  --multirun  rng.seed=range(100)  experiment_name=two_bells_100k  data=two_bells/100k  model=gpytorch_variational  trainer=gpytorch_classif_bernoulli  acquisition.method=epig