import time
from multiprocessing import Process

model_name = "mnist_sep_act_m6_9628"

NUM_PROCESS = 25
TIMEOUT = 600

if __name__ == "__main__":
    from utils.pyct_attack_exp import pyct_random_1_4_8_16_32_limit, pyct_shap_1_4_8_16_32_limit, run_multi_attack
     
    # from utils.pyct_attack_exp import pyct_shap_1_test, pyct_shap_1_test_20_3tak    
    # exp test shap 1 - idx 7, 261, 352, 420, 443, 559 will attack succesfully
    # but pyct can only attack 18
    # inputs = pyct_shap_1_test_20_3tak(model_name)
    
    inputs = pyct_random_1_4_8_16_32_limit(model_name, first_n_img=400, limit=70)
    inputs.extend(
        pyct_shap_1_4_8_16_32_limit(model_name, first_n_img=400, limit=70))
    
    print("#"*40, f"number of inputs: {len(inputs)}", "#"*40)
    time.sleep(3)

    ########## 分派input給各個subprocesses ##########    
    all_subprocess_tasks = [[] for _ in range(NUM_PROCESS)]
    cursor = 0
    for task in inputs:    
        all_subprocess_tasks[cursor].append(task)    
       
        cursor+=1
        if cursor == NUM_PROCESS:
            cursor = 0


    running_processes = []
    for sub_tasks in all_subprocess_tasks:
        if len(sub_tasks) > 0:
            p = Process(target=run_multi_attack, args=(sub_tasks, TIMEOUT, ))
            p.start()
            running_processes.append(p)
            time.sleep(1) # subprocess start 的間隔時間
       
    for p in running_processes:
        p.join()


    print('done')
