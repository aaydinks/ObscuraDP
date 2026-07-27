cd "/home/user/ObscuraDP/Trial 07/27"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/user/ObscuraDP/obscura_env/bin/python run_multiseed.py --seeds 46 --skip-existing
/home/user/ObscuraDP/obscura_env/bin/python run_frontier.py
