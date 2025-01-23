from parse import args
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer, filter_test_data, sample_dataset
from methods.supervised import run_supervised_experiment
from methods.supervised import run_supervised_experiment_EM
from methods.detectgpt import run_perturbation_experiments
from methods.gptzero import run_gptzero_experiment
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, \
    run_GLTR_experiment

if __name__ == '__main__':

    import random
    import torch

    random.seed(0)
    seeds = [random.randint(0, 100000000) for _ in range(100)]

    random.seed(seeds[args.iter])
    torch.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed_all(seeds[args.iter])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    print(f'Loading dataset {args.dataset}...')
    data = dataset_loader.load(args.dataset, detectLLM=args.detectLLM)
    # data = sample_dataset(data, 50, 10)
    # data = filter_test_data(data, max_length=25)

    base_model_name = args.base_model_name.replace('/', '_')
    SAVE_PATH = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.dataset}-{args.detectLLM}"
    if (len(args.load_path) > 0):
        temp = args.load_path.split("-")
        if (len(temp) == 2):
            train_model = temp[1]
        else:
            train_model = "-".join(temp[1:])
    else:
        train_model = args.detectLLM
    args.train_model = train_model
    if (len(args.load_path) > 0):
        args.load_path_criterion = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.load_path}/{train_model}_{args.method}_benchmark_results_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}.pkl"
        if(args.enhance):
            args.load_path_criterion = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.load_path}/{train_model}_{args.method}_benchmark_results_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}_1.pkl"
    if (len(args.load_path) > 0):
        args.load_path = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.load_path}/{args.method}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(SAVE_PATH + '/%s' % args.method):
        os.makedirs(SAVE_PATH + '/%s' % args.method)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    load_base_model(base_model, DEVICE)


    # def phd_criterion(text): return get_phd(
    #     text, base_model, base_tokenizer, DEVICE)

    def ll_criterion(text):
        return get_ll(
            text, base_model, base_tokenizer, DEVICE)


    def rank_criterion(text):
        return -get_rank(text,
                         base_model, base_tokenizer, DEVICE, log=False)


    def logrank_criterion(text):
        return -get_rank(text,
                         base_model, base_tokenizer, DEVICE, log=True)


    def entropy_criterion(text):
        return get_entropy(
            text, base_model, base_tokenizer, DEVICE)


    def GLTR_criterion(text):
        return get_rank_GLTR(
            text, base_model, base_tokenizer, DEVICE)


    outputs = []
    # outputs.append(run_threshold_experiment(data, phd_criterion, "phd"))

    # method_list = ["Log-Likelihood", "Rank", "Log-Rank", "Entropy", "GLTR", "OpenAI-D", "ConDA", "ChatGPT-D", "LM-D", "DetectGPT", "LRR", "NPR", "GPTZero"]

    if args.method == "Log-Likelihood":
        outputs.append(run_threshold_experiment(
            data, ll_criterion, "likelihood", args.load_path_criterion))
    elif args.method == "Rank":
        outputs.append(run_threshold_experiment(data, rank_criterion, "rank", args.load_path_criterion))
    elif args.method == "Log-Rank":
        outputs.append(run_threshold_experiment(
            data, logrank_criterion, "log_rank", args.load_path_criterion))
    elif args.method == "Entropy":
        outputs.append(run_threshold_experiment(
            data, entropy_criterion, "entropy", args.load_path_criterion))
    elif args.method == "GLTR":
        outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", args.load_path_criterion))

    elif args.method == "OpenAI-D":
        outputs.append(
            run_supervised_experiment(
                data,
                model='roberta-base-openai-detector',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/OpenAI-D",
                load_path=args.load_path))
    elif args.method == "OpenAI-STK":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='OpenAI-STK',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/OpenAI-STK",
                load_path=args.load_path))
    elif args.method == "ChatGPT-STK":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='ChatGPT-STK',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/ChatGPT-STK",
                load_path=args.load_path))
    elif args.method == "ChatGPT-D":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='ChatGPT-D',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/ChatGPT-D",
                load_path=args.load_path))
    elif args.method == "MPU":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='MPU',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/MPU",
                load_path=args.load_path))
    elif args.method == "MPU-STK":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='MPU-STK',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/MPU-STK",
                load_path=args.load_path))
    elif args.method == "MPUEM_soft":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='MPUEM_soft',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/MPUEM_soft",
                load_path=args.load_path))
    elif args.method == "Fast":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='Fast',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/Fast",
                load_path=args.load_path))
    elif args.method == "FastEM":
        outputs.append(
            run_supervised_experiment_EM(
                data,
                model='FastEM',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/FastEM",
                load_path=args.load_path))

    # # run GPTZero: pleaze specify your gptzero_key in the args
    elif args.method == "GPTZero":
        outputs.append(run_gptzero_experiment(data, api_key=args.gptzero_key))

    # save results
    import pickle as pkl

    if (args.enhance):
        with open(os.path.join(SAVE_PATH,
                               f"{train_model}_{args.method}_benchmark_results_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}_1.pkl"),
                  "wb") as f:
            pkl.dump(outputs, f)
    else:
        with open(os.path.join(SAVE_PATH,
                               f"{train_model}_{args.method}_benchmark_results_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}.pkl"),
                  "wb") as f:
            pkl.dump(outputs, f)

    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    with open("logs/performance.csv", "a") as wf:
        for row in outputs:
            wf.write(
                f"{args.dataset},{args.detectLLM},{args.base_model_name},{args.method},{json.dumps(row['general'])}\n")

    print("Finish")
