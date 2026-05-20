import json
from openai import AsyncOpenAI
from datetime import datetime
import time
import asyncio
from tqdm import tqdm
import argparse
import os
from pathlib import Path


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please put it in .env or export it in the shell.")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.siliconflow.cn/v1",
)

# openai.base_url = "https://api.chatanywhere.tech/v1"
# MODEL_NAME = "gpt-4"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "gpt-4o"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

MAX_CONCURRENT_REQUESTS = 10


async def run_openai_api_async(system_prompt: str, user_prompt: str, index: int) -> tuple:
    retries = 3
    for attempt in range(retries):
        try:
            rsp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            preference_text = rsp.choices[0].message.content.strip()
            return index, preference_text, None
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
            else:
                return index, None, f"GENERATION_FAILED: {e}"


def construct_intention_prompt(history_data: list, target_day_type: str, target_time_of_day: str,
                               specific_time: str) -> (str, str):
    history_str_parts = [f"[{item['day_type']} {item['time_of_day']}] {item['category_name']}" for item in history_data]
    user_history_str = " -> ".join(history_str_parts)

    # System Prompt 定义模型的角色和总体任务
    system_prompt = ("You are an expert in human behavioral dynamics. Your task is to infer a user's next "
                     "macro-level behavioral intention based on their historical trajectory and a specific time context.")

    # User Prompt 提供具体数据和指令
    user_prompt = f"""
        [User History]: {user_history_str}
        [Current Target Time]: {target_day_type} {target_time_of_day} ({specific_time})

        [Analytical Framework]: To formulate your response, please consider the following analytical steps internally:
            1.  **Thematic History Analysis**: What is the dominant theme or pattern in the user's history (e.g., 'health-conscious', 'work-focused', 'socially active')?
            2.  **Contextual Contrast**: How does this historical theme contrast with the general public's typical behavior at the [Current Target Time]? Highlight any conflict or alignment (e.g., "The user's disciplined weekday pattern conflicts with a typical relaxed weekend evening").
            3.  **Inference of Intentional Shift**: Based on the contrast, infer the user's likely psychological shift. Are they likely to continue their pattern or break from it?

        [Output Format]: Synthesize your analysis from the framework above into a **single, fluid paragraph**, under 50 words. 
            - **Do not** use bullet points, numbered lists, or labels like "Step 1".
            - The tone should be that of an expert providing a concise, narrative summary.
            - Start by acknowledging the user's pattern and the current context.
            - Conclude by stating the inferred intention and the most likely macro-categories of interest (e.g., Food, Entertainment, Shopping).
        """

    return system_prompt, user_prompt


def load_data_and_prepare_input(json_filepath: str, index: int) -> dict:
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    if not (0 <= index < len(data)):
        print(f"错误：索引 {index} 超出范围。文件中有 {len(data)} 个片段。")
        return None

    segment = data[index]

    # 解析 result 中的 timestamp
    result_timestamp_str = segment['result']['timestamp']
    # fromisoformat 可以直接处理 "2012-04-10T16:21:48+00:00" 格式
    dt_object = datetime.fromisoformat(result_timestamp_str)

    # 将datetime对象格式化为 "Weekday HH:MM" 的形式
    # %A 表示完整的星期几名称 (e.g., "Tuesday")
    # %H 表示24小时制的小时, %M 表示分钟
    specific_time_str = dt_object.strftime("%A %H:%M")

    return {
        "history_data": segment['history'],
        "target_day_type": segment['result']['day_type'],
        "target_time_of_day": segment['result']['time_of_day'],
        "specific_time": specific_time_str
    }


async def generate_cot_parallel(filename: str):
    with open(filename, "r", encoding='utf-8') as f:
        fp = json.load(f)

    # 1. 筛选出所有需要处理的条目及其原始索引
    items_to_process = [(i, v) for i, v in enumerate(fp) if not v.get("preference")]

    if not items_to_process:
        print("所有条目均已处理完毕！")
        return

    total_to_process = len(items_to_process)
    print(f"共有 {len(fp)} 条记录，其中 {total_to_process} 条需要生成。")
    print(f"使用安全并发数: {MAX_CONCURRENT_REQUESTS}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def controlled_api_call(system_prompt, user_prompt, index):
        # 在进入API调用前，先获取一个信号量
        async with semaphore:
            # 现在，实际并发运行的协程数量不会超过 MAX_CONCURRENT_REQUESTS
            return await run_openai_api_async(system_prompt, user_prompt, index)

    # 2. 为每个需要处理的条目创建异步任务
    tasks = []
    for i, v in items_to_process:
        result_timestamp_str = v['result']['timestamp']
        dt_object = datetime.fromisoformat(result_timestamp_str)
        specific_time_str = dt_object.strftime("%A %H:%M")

        system_prompt, user_prompt = construct_intention_prompt(
            history_data=v['history'],
            target_day_type=v['result']['day_type'],
            target_time_of_day=v['result']['time_of_day'],
            specific_time=specific_time_str
        )
        # 将调用API的协程加入任务列表
        tasks.append(controlled_api_call(system_prompt, user_prompt, i))

    # 3. 使用tqdm作为进度条，并行执行所有任务
    print("开始并行生成，请稍候...")
    start_time = time.time()

    fail_count = 0
    completed_count = 0
    SAVE_INTERVAL = 100

    # asyncio.as_completed 会在任务完成时立即返回结果，非常适合与tqdm配合
    for future in tqdm(asyncio.as_completed(tasks), total=total_to_process, desc="Generating CoT"):
        # 等待下一个完成的任务
        original_index, preference_text, error_message = await future
        # 将返回的结果更新回原始数据列表
        if error_message is None:
            fp[original_index]["preference"] = preference_text
        else:
            print(f"Error: {error_message}")
            fail_count += 1

        completed_count += 1
        if completed_count % SAVE_INTERVAL == 0:
            with open(filename, "w", encoding='utf-8') as wf:
                json.dump(fp, wf, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"\n所有任务处理完成，耗时: {end_time - start_time:.2f} 秒。")

    print("\n" + "=" * 50)
    print("--- 任务总结 ---")
    num_success = total_to_process - fail_count
    print(f"成功: {num_success} 条")
    print(f"失败: {fail_count} 条")

    # 4. 所有任务完成后，统一保存一次文件
    print(f"正在将全部 {len(fp)} 条结果保存至 '{filename}'...")
    with open(filename, "w", encoding='utf-8') as wf:
        json.dump(fp, wf, indent=4, ensure_ascii=False)

    print("保存完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    args = parser.parse_args()
    dataset = args.dataset_name

    asyncio.run(generate_cot_parallel(f'../datasets/{dataset}/dataIntegration/val_set.json'))
    asyncio.run(generate_cot_parallel(f'../datasets/{dataset}/dataIntegration/test_set.json'))
    asyncio.run(generate_cot_parallel(f'../datasets/{dataset}/dataIntegration/train_set.json'))
