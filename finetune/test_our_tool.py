import json
import torch
import pandas as pd
import os

from test import cal_metrics, analyze_test_results, align_test_metrics, compare_test_metrics

def tool_run_tests():
    base_model_name_or_path = "/home/Data/xac/nas/models/models--codellama--CodeLlama-13b-Instruct-hf/snapshots/745795438019e47e4dad1347a0093e11deee4c68"
    # 2k, 4k, 8k
    model_max_length = 1024 * 8

    peft_output_list = [
        None,  # the off-the-shelf LLM
        "/home/Data/xac/merge/llm_one4all/finetuning/model/checkpoint-80",  # finetuning output
    ]

    data_path = "data4ft.json"

    data_path_list = data_path
    
    output_path_list = [
        None,
        # "sliced"
        "xxx",
        # "xxx_sliced"
    ]

    df = pd.DataFrame({
        "peft_output": peft_output_list,
        "output_path": output_path_list,
        "data_path": data_path_list,
    })
    df["base_model_name_or_path"] = base_model_name_or_path
    df["model_max_length"] = model_max_length
    df["max_length"] = model_max_length

    df["peft_output"] = df["peft_output"].map(lambda x: (MODEL_PATH + x) if x is not None else None)
    df["output_path"] = df["output_path"].map(lambda x: OUTPUT_PATH + SLICE_PATH + (f"out_{model_name}_{model_max_length}_{x}_result.json" if x is not None else f"out_{model_name}_{model_max_length}_result.json"))
    df["data_path"] = df["data_path"].map(lambda x: DATA_PATH + SLICE_PATH + x)

    print("#test task num:", len(df))

    avail_cuda = [0]
    num_avail_cuda = len(avail_cuda)

    cmd_param_base = "--base_model_name_or_path '{base_model_name_or_path}' --data_path '{data_path}' --output_path '{output_path}' --model_max_length {model_max_length} --max_length {model_max_length} --device '{device}'"
    for idx, task in enumerate(df.to_dict(orient="records")):
        print(f"========== task {idx} ==========")

        task["device"] = f"cuda:{avail_cuda[idx % num_avail_cuda]}"

        cmd_param = cmd_param_base
        if task["peft_output"] is not None:
            cmd_param += " --peft_output '{peft_output}'"

        cmd_param = cmd_param.format_map(task)
        print(cmd_param)

        cmd = f"nohup python predict.py {cmd_param} >test.log 2>&1 &"
        os.system(cmd)